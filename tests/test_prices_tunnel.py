"""Prices end to end: fake gateway <-WebSocket-> TunnelClient, with config.toml on disk (capability models.prices)."""
import asyncio
import logging
import tomllib

import pytest_asyncio

from socketllama import client as client_module
from socketllama import protocol
from socketllama.client import TunnelClient
from socketllama.config import ModelPrice, load_config
from socketllama.config_settings import ConfigSettings

from .conftest import FAST_RECONNECT_SECONDS, STARTUP_TIMEOUT_SECONDS, wait_until
from .fakes import INSTANCE, MODEL, SIGNING_SECRET, TOKEN, FakeBackend, FakeGateway

OTHER_MODEL = 'qwen3:14b'
# Shared in the configuration, but not among the models the backend offers
NOT_OFFERED = 'later:1b'
OWNER_ONLY = 0o600
READ_ONLY = 0o400
QUIET_SECONDS = 0.2
SHORT_RESEND_SECONDS = 0.1
SHORT_REFRESH_SECONDS = 0.3
PRICES_CAPABLE = [protocol.CAPABILITY_MODELS_PRICES]
FILE_PRICE = ModelPrice(input=0.8, output=2.0)
TOPTIC_PRICE = ModelPrice(input=0.4, output=1.4)
FILE_PRICE_LINE = f'"{MODEL}" = {{ input = 0.80, output = 2.00 }}   # my own'


def config_text(gateway_url: str, backend_url: str) -> str:
    return f'''# written by hand
[gateway]
url = "{gateway_url}"
instance = "{INSTANCE}"
token = "{TOKEN}"
signing_secret = "{SIGNING_SECRET}"
allow_insecure = true

[client]
reconnect_min = {FAST_RECONNECT_SECONDS}
reconnect_max = {FAST_RECONNECT_SECONDS}

[[backends]]
name = "ollama"
kind = "ollama"
base_url = "{backend_url}"
models = ["{MODEL}", "{OTHER_MODEL}", "{NOT_OFFERED}"]

[backends.prices_per_1m_tokens]
{FILE_PRICE_LINE}
'''


def settings_frame(model: str, price: ModelPrice) -> dict:
    return {'type': protocol.FRAME_MODELS_SETTINGS, 'models': [
        {'backend': 'ollama', 'model': model, 'price': {'input': price.input, 'output': price.output}}]}


def file_prices(path) -> dict:
    return tomllib.loads(path.read_text())['backends'][0]['prices_per_1m_tokens']


@pytest_asyncio.fixture
async def backend():
    async with FakeBackend(more_models=[OTHER_MODEL]) as fake:
        yield fake


class Running:
    """A client started from a config.toml in ``tmp_path``; stopped at the end."""

    def __init__(self, tmp_path, gateway: FakeGateway, backend: FakeBackend, mode: int = OWNER_ONLY):
        self.path = tmp_path / 'config.toml'
        self.path.write_text(config_text(gateway.url, backend.base_url))
        self.path.chmod(mode)
        self.client = TunnelClient(load_config(self.path))
        self._gateway = gateway

    async def __aenter__(self):
        self.task = asyncio.create_task(self.client.run_forever())
        await wait_until(lambda: self._gateway.connections)
        return self

    async def __aexit__(self, *exc_info):
        self.client.stop()
        await asyncio.wait_for(self.task, STARTUP_TIMEOUT_SECONDS)


async def acknowledged_report(gateway: FakeGateway) -> dict:
    report = await gateway.next_frame(protocol.FRAME_MODELS)
    await gateway.send({'type': protocol.FRAME_MODELS_ACK})
    return report


def models_of(report: dict) -> dict:
    return {model['name']: model for backend in report['backends'] for model in backend['models']}


async def report_with(gateway: FakeGateway, price: dict) -> dict:
    """The next models report whose MODEL carries ``price`` (an earlier one may be a resend)."""
    while True:
        report = await gateway.next_frame(protocol.FRAME_MODELS)
        if models_of(report)[MODEL].get(protocol.CONFIG_PRICE_FIELD) == price:
            return report


async def test_the_client_says_it_takes_prices(tmp_path, gateway, backend):
    async with Running(tmp_path, gateway, backend):
        pass
    assert protocol.CAPABILITY_MODELS_PRICES in gateway.auth_frames[0]['capabilities']


async def test_a_gateway_that_takes_prices_is_told_the_price_of_the_file(tmp_path, backend):
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, Running(tmp_path, gateway, backend):
        models = models_of(await gateway.next_frame(protocol.FRAME_MODELS))
    assert models[MODEL][protocol.CONFIG_PRICE_FIELD] == {'input': FILE_PRICE.input, 'output': FILE_PRICE.output}
    # the backend's own default stays as it was; a model without a line has no price to report
    assert 'input_cost_per_token' not in models[MODEL]
    assert protocol.CONFIG_PRICE_FIELD not in models[OTHER_MODEL]


async def test_a_gateway_without_prices_is_told_none_and_none_it_sends_are_written(tmp_path, gateway, backend, caplog):
    async with Running(tmp_path, gateway, backend) as running:
        report = await acknowledged_report(gateway)
        await gateway.send(settings_frame(MODEL, TOPTIC_PRICE))
        await asyncio.sleep(QUIET_SECONDS)
        await gateway.drop_connection()
        await wait_until(lambda: len(gateway.connections) == 2)
        await asyncio.sleep(QUIET_SECONDS)
    assert protocol.CONFIG_PRICE_FIELD not in models_of(report)[MODEL]
    assert file_prices(running.path)[MODEL] == FILE_PRICE.as_table()
    # said once for the run, not at every connection
    assert caplog.text.count('does not take prices from your machine') == 1


async def test_a_price_sent_before_a_report_was_acknowledged_is_ignored(tmp_path, backend):
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await gateway.next_frame(protocol.FRAME_MODELS)
        await gateway.send(settings_frame(MODEL, TOPTIC_PRICE))
        await asyncio.sleep(QUIET_SECONDS)
        assert file_prices(running.path)[MODEL] == FILE_PRICE.as_table()
        # once toptic.ai has seen the file's prices, the price it sends counts
        await gateway.send({'type': protocol.FRAME_MODELS_ACK})
        await gateway.send(settings_frame(MODEL, TOPTIC_PRICE))
        await wait_until(lambda: file_prices(running.path)[MODEL] == TOPTIC_PRICE.as_table())


async def test_a_written_price_is_kept_in_the_line_and_reported_soon(tmp_path, backend, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', SHORT_RESEND_SECONDS)
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await acknowledged_report(gateway)
        await gateway.send(settings_frame(MODEL, TOPTIC_PRICE))
        # well before the next refresh: what toptic.ai was told of the file is out of date
        await report_with(gateway, {'input': TOPTIC_PRICE.input, 'output': TOPTIC_PRICE.output})
    assert f'"{MODEL}" = {{ input = 0.4, output = 1.4 }}   # my own' in running.path.read_text()
    assert f'toptic.ai set the price of ollama/{MODEL} to $0.40 in, $1.40 out per 1M tokens' in caplog.text


async def test_after_a_reconnect_a_price_is_taken_at_once_for_a_model_reported_earlier_in_the_run(tmp_path, backend):
    """The second connection sends no report of its own within the gateway's minute (the report is unchanged
    and was just acknowledged): what the first connection's acknowledged report listed still counts."""
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await acknowledged_report(gateway)
        await gateway.drop_connection()
        await wait_until(lambda: len(gateway.connections) == 2)
        await gateway.send(settings_frame(MODEL, TOPTIC_PRICE))
        await wait_until(lambda: file_prices(running.path)[MODEL] == TOPTIC_PRICE.as_table())
        # taken without a report or an acknowledgement on the second connection
        assert [frame for frame in gateway.drain_frames() if frame['type'] == protocol.FRAME_MODELS] == []


async def test_a_price_for_a_model_the_acknowledged_report_did_not_list_is_not_written(tmp_path, backend):
    """The model is shared but was not offered when the report was made: toptic.ai has not seen its line."""
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await acknowledged_report(gateway)
        await gateway.send(settings_frame(NOT_OFFERED, TOPTIC_PRICE))
        await gateway.send(settings_frame(OTHER_MODEL, TOPTIC_PRICE))  # one it did list, as a marker
        await wait_until(lambda: OTHER_MODEL in file_prices(running.path))
    assert NOT_OFFERED not in file_prices(running.path)


async def test_a_price_for_a_model_without_a_line_adds_one(tmp_path, backend):
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await acknowledged_report(gateway)
        await gateway.send(settings_frame(OTHER_MODEL, TOPTIC_PRICE))
        await wait_until(lambda: OTHER_MODEL in file_prices(running.path))
    assert file_prices(running.path) == {MODEL: FILE_PRICE.as_table(), OTHER_MODEL: TOPTIC_PRICE.as_table()}


async def test_a_file_that_cannot_be_written_keeps_its_price_reported(tmp_path, backend, monkeypatch, caplog):
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', SHORT_RESEND_SECONDS)
    monkeypatch.setattr(client_module, 'MODEL_REFRESH_INTERVAL_SECONDS', SHORT_REFRESH_SECONDS)
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend, mode=READ_ONLY) as running:
        await acknowledged_report(gateway)
        await gateway.send(settings_frame(MODEL, TOPTIC_PRICE))
        await wait_until(lambda: 'read-only' in caplog.text)
        later = await gateway.next_frame(protocol.FRAME_MODELS)
    assert models_of(later)[MODEL][protocol.CONFIG_PRICE_FIELD] == {'input': FILE_PRICE.input,
                                                                     'output': FILE_PRICE.output}
    assert file_prices(running.path)[MODEL] == FILE_PRICE.as_table()


async def test_a_price_changed_by_hand_while_the_client_runs_is_not_written_over(tmp_path, backend, caplog):
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await acknowledged_report(gateway)
        hand_edited = running.path.read_text().replace('input = 0.80', 'input = 0.90')
        running.path.write_text(hand_edited)
        await gateway.send(settings_frame(MODEL, TOPTIC_PRICE))
        await wait_until(lambda: 'Restart the client' in caplog.text)
    assert running.path.read_text() == hand_edited


async def test_a_failure_while_writing_does_not_end_the_connection(tmp_path, backend, monkeypatch, caplog):
    def broken(self, frame, acknowledged):
        raise RuntimeError('editor bug')
    monkeypatch.setattr(ConfigSettings, 'take_settings', broken)
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await acknowledged_report(gateway)
        await gateway.send(settings_frame(MODEL, TOPTIC_PRICE))
        await wait_until(lambda: 'editor bug' in caplog.text)
        await gateway.send({'type': protocol.FRAME_PING})  # anything: the connection still serves
        await asyncio.sleep(QUIET_SECONDS)
        assert running.client.connections_established == 1


async def test_the_start_shows_the_prices_of_the_file(tmp_path, gateway, backend, caplog):
    caplog.set_level(logging.INFO)
    async with Running(tmp_path, gateway, backend):
        pass
    assert f'Prices in config.toml: ollama/{MODEL} $0.80 in, $2.00 out per 1M tokens' in caplog.text
