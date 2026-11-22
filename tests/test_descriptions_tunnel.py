"""Descriptions end to end: fake gateway <-WebSocket-> TunnelClient, with config.toml on disk (capability
models.descriptions, beside models.prices)."""
import asyncio
import logging
import tomllib

import pytest_asyncio

from socketllama import client as client_module
from socketllama import protocol
from socketllama.client import TunnelClient
from socketllama.config import ModelPrice, load_config

from .conftest import FAST_RECONNECT_SECONDS, STARTUP_TIMEOUT_SECONDS, wait_until
from .fakes import INSTANCE, MODEL, SIGNING_SECRET, TOKEN, FakeBackend, FakeGateway

OTHER_MODEL = 'qwen3:14b'
OWNER_ONLY = 0o600
QUIET_SECONDS = 0.2
SHORT_RESEND_SECONDS = 0.1
BOTH_CAPABLE = [protocol.CAPABILITY_MODELS_PRICES, protocol.CAPABILITY_MODELS_DESCRIPTIONS]
PRICES_CAPABLE = [protocol.CAPABILITY_MODELS_PRICES]
FILE_DESCRIPTION = 'Quick answers'
TOPTIC_DESCRIPTION = 'Quick answers to short questions, "in English" and Grüezi'
FILE_PRICE = ModelPrice(input=0.8, output=2.0)
TOPTIC_PRICE = ModelPrice(input=0.4, output=1.4)


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
models = ["{MODEL}", "{OTHER_MODEL}"]

[backends.descriptions]
"{MODEL}" = "{FILE_DESCRIPTION}"   # my own

[backends.prices_per_1m_tokens]
"{MODEL}" = {{ input = 0.80, output = 2.00 }}
'''


def settings_frame(model: str, description: str | None = None, price: ModelPrice | None = None) -> dict:
    entry = {'backend': 'ollama', 'model': model}
    if description is not None:
        entry['description'] = description
    if price is not None:
        entry['price'] = {'input': price.input, 'output': price.output}
    return {'type': protocol.FRAME_MODELS_SETTINGS, 'models': [entry]}


def in_file(path, table: str) -> dict:
    return tomllib.loads(path.read_text())['backends'][0].get(table, {})


def file_descriptions(path) -> dict:
    return in_file(path, 'descriptions')


@pytest_asyncio.fixture
async def backend():
    async with FakeBackend(more_models=[OTHER_MODEL]) as fake:
        yield fake


class Running:
    """A client started from a config.toml in ``tmp_path``; stopped at the end."""

    def __init__(self, tmp_path, gateway: FakeGateway, backend: FakeBackend):
        self.path = tmp_path / 'config.toml'
        self.path.write_text(config_text(gateway.url, backend.base_url))
        self.path.chmod(OWNER_ONLY)
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


async def test_the_client_says_it_takes_descriptions(tmp_path, gateway, backend):
    async with Running(tmp_path, gateway, backend):
        pass
    assert protocol.CAPABILITY_MODELS_DESCRIPTIONS in gateway.auth_frames[0]['capabilities']


async def test_a_gateway_that_takes_descriptions_is_told_the_description_of_the_file(tmp_path, backend):
    async with FakeGateway(capabilities=BOTH_CAPABLE) as gateway, Running(tmp_path, gateway, backend):
        models = models_of(await gateway.next_frame(protocol.FRAME_MODELS))
    assert models[MODEL][protocol.CONFIG_DESCRIPTION_FIELD] == FILE_DESCRIPTION
    assert models[MODEL][protocol.CONFIG_PRICE_FIELD] == {'input': FILE_PRICE.input, 'output': FILE_PRICE.output}
    # a model without a line has no such field at all
    assert protocol.CONFIG_DESCRIPTION_FIELD not in models[OTHER_MODEL]


async def test_a_gateway_that_takes_prices_only_is_told_no_descriptions_and_none_it_sends_is_written(
        tmp_path, backend, caplog):
    async with FakeGateway(capabilities=PRICES_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        report = await acknowledged_report(gateway)
        await gateway.send(settings_frame(MODEL, TOPTIC_DESCRIPTION, TOPTIC_PRICE))
        # the price beside it is taken
        await wait_until(lambda: in_file(running.path, 'prices_per_1m_tokens')[MODEL] == TOPTIC_PRICE.as_table())
        await gateway.drop_connection()
        await wait_until(lambda: len(gateway.connections) == 2)
        await asyncio.sleep(QUIET_SECONDS)
    assert protocol.CONFIG_DESCRIPTION_FIELD not in models_of(report)[MODEL]
    assert file_descriptions(running.path) == {MODEL: FILE_DESCRIPTION}
    # said once for the run, not at every connection
    assert caplog.text.count('does not take descriptions from your machine') == 1
    assert 'does not take prices' not in caplog.text


async def test_a_description_sent_before_a_report_was_acknowledged_is_ignored(tmp_path, backend):
    async with FakeGateway(capabilities=BOTH_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await gateway.next_frame(protocol.FRAME_MODELS)
        await gateway.send(settings_frame(MODEL, TOPTIC_DESCRIPTION))
        await asyncio.sleep(QUIET_SECONDS)
        assert file_descriptions(running.path) == {MODEL: FILE_DESCRIPTION}
        # once toptic.ai has seen what the file holds, the description it sends counts
        await gateway.send({'type': protocol.FRAME_MODELS_ACK})
        await gateway.send(settings_frame(MODEL, TOPTIC_DESCRIPTION))
        await wait_until(lambda: file_descriptions(running.path)[MODEL] == TOPTIC_DESCRIPTION)


async def test_a_price_and_a_description_are_written_and_reported_soon(tmp_path, backend, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', SHORT_RESEND_SECONDS)
    async with FakeGateway(capabilities=BOTH_CAPABLE) as gateway, \
            Running(tmp_path, gateway, backend) as running:
        await acknowledged_report(gateway)
        await gateway.send(settings_frame(MODEL, TOPTIC_DESCRIPTION, TOPTIC_PRICE))
        # well before the next refresh: what toptic.ai was told of the file is out of date
        while True:
            model = models_of(await gateway.next_frame(protocol.FRAME_MODELS))[MODEL]
            if model.get(protocol.CONFIG_DESCRIPTION_FIELD) == TOPTIC_DESCRIPTION:
                break
    assert model[protocol.CONFIG_PRICE_FIELD] == {'input': TOPTIC_PRICE.input, 'output': TOPTIC_PRICE.output}
    text = running.path.read_text()
    assert f'"{MODEL}" = "Quick answers to short questions, \\"in English\\" and Grüezi"   # my own' in text
    assert f'"{MODEL}" = {{ input = 0.4, output = 1.4 }}' in text
    assert f'toptic.ai set the description of ollama/{MODEL} to "{TOPTIC_DESCRIPTION}"' in caplog.text
    assert len(list(tmp_path.glob('config.toml.backup-*'))) == 1


async def test_the_start_shows_the_models_with_a_description(tmp_path, gateway, backend, caplog):
    caplog.set_level(logging.INFO)
    async with Running(tmp_path, gateway, backend):
        pass
    assert f'Descriptions in config.toml: ollama/{MODEL}\n' in caplog.text
