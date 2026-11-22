"""What the running client shows its owner, end to end: one line per event that matters, at INFO."""
import asyncio
import json
import logging

import httpx

import pytest
import pytest_asyncio
from websockets.exceptions import ConnectionClosed

from socketllama import client as client_module
from socketllama import events, models_memory, protocol
from socketllama.client import TunnelClient
from socketllama.relay import Relay

from .conftest import STARTUP_TIMEOUT_SECONDS, make_config, ollama_backend, wait_until
from .fakes import MODEL, NONCE, SIGNING_SECRET, STREAM_PARTS, FakeBackend, FakeGateway

CLIENT_LOGGER = 'socketllama'
USER_NAME = 'Alice Example'
PROMPT_TEXT = 'What should we offer in the merger talks?'
CHAT = {'model': MODEL, 'stream': True, 'messages': [{'role': 'system', 'content': 'Be brief.'},
                                                     {'role': 'user', 'content': PROMPT_TEXT}]}
CHAT_SIZE = f'2 messages, {len("Be brief.") + len(PROMPT_TEXT)} characters'
USAGE = {'prompt_tokens': 812, 'completion_tokens': 304, 'total_tokens': 1116}
QUIET_SECONDS = 0.2
SHORT_INTERVAL_SECONDS = 0.05
MEMORY_CAPABLE = [protocol.CAPABILITY_MODELS_MEMORY]


def lines(caplog, level: int = logging.INFO) -> list[str]:
    """What the client showed at exactly ``level``."""
    return [record.getMessage() for record in caplog.records
            if record.name == CLIENT_LOGGER and record.levelno == level]


def lines_of_request(caplog, number: int) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if record.name == CLIENT_LOGGER and record.getMessage().startswith(f'#{number} ')]


class Running:
    def __init__(self, gateway: FakeGateway, backend: FakeBackend, **backend_options):
        self.client = TunnelClient(make_config(gateway.url, [ollama_backend(backend, **backend_options)]))
        self._gateway = gateway

    async def __aenter__(self):
        self.task = asyncio.create_task(self.client.run_forever())
        await wait_until(lambda: self._gateway.connections)
        return self.client

    async def __aexit__(self, *exc_info):
        self.client.stop()
        await asyncio.wait_for(self.task, STARTUP_TIMEOUT_SECONDS)


@pytest.fixture(autouse=True)
def info_logged(caplog):
    caplog.set_level(logging.INFO, logger=CLIENT_LOGGER)


@pytest_asyncio.fixture
async def counting_backend():
    async with FakeBackend(usage=USAGE) as fake:
        yield fake


# ----- the client says it can take the name -----

async def test_the_client_tells_the_gateway_it_shows_who_asks(gateway, backend):
    async with Running(gateway, backend):
        pass
    assert protocol.CAPABILITY_REQUEST_USER in gateway.auth_frames[0]['capabilities']


# ----- a prompt and its answer -----

async def test_a_prompt_and_its_end_are_one_line_each(gateway, counting_backend, caplog):
    async with Running(gateway, counting_backend):
        await gateway.send_request('req-1', CHAT, **{protocol.REQUEST_USER_FIELD: USER_NAME})
        await gateway.collect_response('req-1')
        await wait_until(lambda: len(lines_of_request(caplog, 1)) == 2)
    assert lines_of_request(caplog, 1)[0] == f'#1 prompt from {USER_NAME} for ollama/{MODEL}: {CHAT_SIZE}'
    assert lines_of_request(caplog, 1)[1].startswith('#1 done after ')
    assert lines_of_request(caplog, 1)[1].endswith(' s, 812 prompt tokens, 304 answer tokens')


async def test_without_a_name_the_prompt_is_from_an_unknown_user(gateway, backend, caplog):
    """toptic.ai sends no name where it does not take request.user."""
    async with Running(gateway, backend):
        await gateway.send_request('req-1', CHAT)
        await gateway.collect_response('req-1')
        await wait_until(lambda: len(lines_of_request(caplog, 1)) == 2)
    prompt, end = lines_of_request(caplog, 1)
    assert prompt == f'#1 prompt from unknown user for ollama/{MODEL}: {CHAT_SIZE}'
    # this backend gives no token counts
    assert end.startswith('#1 done after ') and end.endswith(' s')


async def test_an_answer_that_is_not_streamed_has_its_token_counts_too(gateway, counting_backend, caplog):
    async with Running(gateway, counting_backend):
        await gateway.send_request('req-1', {**CHAT, 'stream': False})
        await gateway.collect_response('req-1')
        await wait_until(lambda: len(lines_of_request(caplog, 1)) == 2)
    assert lines_of_request(caplog, 1)[1].endswith(' s, 812 prompt tokens, 304 answer tokens')


async def test_prompts_are_numbered_across_reconnects(gateway, backend, caplog):
    async with Running(gateway, backend):
        await gateway.send_request('req-1', CHAT)
        await gateway.collect_response('req-1')
        await gateway.drop_connection()
        await wait_until(lambda: len(gateway.connections) == 2)
        await gateway.send_request('req-2', CHAT)
        await gateway.collect_response('req-2')
        await wait_until(lambda: len(lines_of_request(caplog, 2)) == 2)
    assert lines_of_request(caplog, 2)[0].startswith('#2 prompt from ')


async def test_neither_the_prompt_nor_the_answer_is_shown(gateway, counting_backend, caplog):
    caplog.set_level(logging.DEBUG)
    async with Running(gateway, counting_backend):
        await gateway.send_request('req-1', CHAT, **{protocol.REQUEST_USER_FIELD: USER_NAME})
        await gateway.collect_response('req-1')
        await wait_until(lambda: len(lines_of_request(caplog, 1)) == 2)
    shown = '\n'.join(record.getMessage() for record in caplog.records if record.name.startswith(CLIENT_LOGGER))
    for content in (PROMPT_TEXT, 'Be brief', STREAM_PARTS[0] + STREAM_PARTS[1]):
        assert content not in shown


async def test_a_backend_error_ends_the_answer_as_an_error_with_its_status_and_kind(gateway, backend, caplog):
    async with Running(gateway, backend):
        await gateway.send_request('req-1', {**CHAT, 'model': 'missing'})
        await gateway.collect_response('req-1')
        await wait_until(lambda: lines(caplog, logging.WARNING))
    [end] = [line for line in lines(caplog, logging.WARNING) if line.startswith('#1 ')]
    assert end.startswith('#1 error after ') and end.endswith(' s: HTTP 404 from ollama (not_found_error)')


async def test_the_backends_error_message_is_only_a_detail(gateway, backend, caplog):
    """It may quote the request, so it is shown with -v only."""
    caplog.set_level(logging.DEBUG, logger=CLIENT_LOGGER)
    async with Running(gateway, backend):
        await gateway.send_request('req-1', {**CHAT, 'model': 'missing'})
        await gateway.collect_response('req-1')
        await wait_until(lambda: lines(caplog, logging.DEBUG) and lines(caplog, logging.WARNING))
    assert not any('model not found' in line for line in lines(caplog, logging.WARNING))
    assert '#1 the backend said: model not found' in lines(caplog, logging.DEBUG)


async def test_an_unreachable_backend_is_one_warning(gateway, caplog):
    dead = {'name': 'ollama', 'kind': 'ollama', 'base_url': 'http://127.0.0.1:9/v1', 'models': []}
    client = TunnelClient(make_config(gateway.url, [dead]))
    task = asyncio.create_task(client.run_forever())
    await wait_until(lambda: gateway.connections)
    await gateway.send_request('req-1', CHAT)
    await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    client.stop()
    await asyncio.wait_for(task, STARTUP_TIMEOUT_SECONDS)
    [end] = [line for line in lines(caplog, logging.WARNING) if line.startswith('#1 ')]
    assert end.startswith('#1 error after ') and f': {protocol.ERROR_BACKEND_UNREACHABLE}, ' in end


async def test_a_refused_request_is_one_warning_without_a_number(gateway, backend, caplog):
    async with Running(gateway, backend):
        await gateway.send_request('req-1', CHAT, secret='not-the-secret')
        await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    assert [line for line in lines(caplog, logging.WARNING) if 'request' in line] == [
        f'Refused a request: {protocol.ERROR_BAD_SIGNATURE}, Request signature does not match the signing secret']
    assert lines_of_request(caplog, 1) == []


# ----- cancel, pause, resume -----

async def test_a_cancelled_answer_ends_as_cancelled_by_toptic(gateway, backend, caplog):
    async with Running(gateway, backend):
        backend.hold_stream.clear()
        await gateway.send_request('req-1', CHAT)
        await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
        await gateway.send({'type': protocol.FRAME_CANCEL, 'id': 'req-1'})
        await wait_until(lambda: len(lines_of_request(caplog, 1)) == 2)
        backend.hold_stream.set()
    end = lines_of_request(caplog, 1)[1]
    assert end.startswith('#1 cancelled after ') and end.endswith(f' s: {events.STOPPED_BY_GATEWAY}')


async def test_an_answer_cut_off_by_the_connection_ends_as_cancelled_for_that_reason(gateway, backend, caplog):
    async with Running(gateway, backend):
        backend.hold_stream.clear()
        await gateway.send_request('req-1', CHAT)
        await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
        await gateway.drop_connection()
        await wait_until(lambda: len(lines_of_request(caplog, 1)) == 2)
        backend.hold_stream.set()
    assert lines_of_request(caplog, 1)[1].endswith(f' s: {events.CONNECTION_ENDED}')


async def test_pause_and_resume_are_shown_once_each(gateway, backend, caplog):
    async with Running(gateway, backend):
        backend.hold_stream.clear()
        await gateway.send_request('req-1', CHAT)
        await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
        for _ in range(2):
            await gateway.send({'type': protocol.FRAME_PAUSE, 'id': 'req-1'})
        await asyncio.sleep(QUIET_SECONDS)
        for _ in range(2):
            await gateway.send({'type': protocol.FRAME_RESUME, 'id': 'req-1'})
        backend.hold_stream.set()
        await gateway.next_frame(protocol.FRAME_RESPONSE_END)
        await wait_until(lambda: len(lines_of_request(caplog, 1)) == 4)
    assert lines_of_request(caplog, 1)[1:3] == [events.paused(1), events.resumed(1)]


async def test_after_the_first_pause_and_resume_the_others_are_details(gateway, backend, caplog):
    """Backpressure may stop and start a long answer many times."""
    caplog.set_level(logging.DEBUG, logger=CLIENT_LOGGER)
    async with Running(gateway, backend):
        backend.hold_stream.clear()
        await gateway.send_request('req-1', CHAT)
        await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
        for _ in range(2):
            await gateway.send({'type': protocol.FRAME_PAUSE, 'id': 'req-1'})
            await gateway.send({'type': protocol.FRAME_RESUME, 'id': 'req-1'})
        await wait_until(lambda: len([record for record in caplog.records if record.name == CLIENT_LOGGER
                                      and record.getMessage().startswith('#1 ')]) >= 5)
        backend.hold_stream.set()
        await gateway.next_frame(protocol.FRAME_RESPONSE_END)
    assert lines(caplog).count(events.paused(1)) == 1 and lines(caplog).count(events.resumed(1)) == 1
    assert lines(caplog, logging.DEBUG).count(events.paused(1)) == 1


# ----- the connection and the models -----

async def test_connecting_and_the_models_report_are_shown(gateway, backend, caplog):
    async with Running(gateway, backend):
        await gateway.next_frame(protocol.FRAME_MODELS)
        await gateway.send({'type': protocol.FRAME_MODELS_ACK, 'created': [f'tester-home/ollama/{MODEL}']})
        await wait_until(lambda: any(line.startswith('toptic.ai registered') for line in lines(caplog)))
    shown = lines(caplog)
    assert events.connected(gateway.url, 'tester-home') in shown
    assert 'Reported to toptic.ai: ollama 1 model' in shown
    assert 'toptic.ai registered the 1 model reported: 1 new' in shown


async def test_a_report_that_did_not_change_is_not_shown_again(gateway, backend, caplog, monkeypatch):
    monkeypatch.setattr(client_module, 'MODEL_REFRESH_INTERVAL_SECONDS', SHORT_INTERVAL_SECONDS)
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', SHORT_INTERVAL_SECONDS / 2)
    async with Running(gateway, backend):
        for _ in range(2):
            await gateway.next_frame(protocol.FRAME_MODELS)
            await gateway.send({'type': protocol.FRAME_MODELS_ACK})
        await asyncio.sleep(QUIET_SECONDS)
    assert len([line for line in lines(caplog) if line.startswith('Reported to toptic.ai')]) == 1
    assert len([line for line in lines(caplog) if line.startswith('toptic.ai registered')]) == 1


async def test_a_refused_authentication_says_why(backend):
    refusal = {'type': protocol.FRAME_AUTH_FAILED, 'reason': protocol.AUTH_FAILED_TOKEN}
    async with FakeGateway(auth_answer=refusal) as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
        with pytest.raises(client_module.FatalGatewayError) as refused:
            await asyncio.wait_for(client.run_forever(), STARTUP_TIMEOUT_SECONDS)
    assert str(refused.value) == events.auth_refused(protocol.AUTH_FAILED_TOKEN)


async def test_a_change_of_the_models_in_memory_is_shown_once(backend, caplog, monkeypatch):
    monkeypatch.setattr(models_memory, 'MODELS_MEMORY_POLL_SECONDS', SHORT_INTERVAL_SECONDS)
    monkeypatch.setattr(models_memory, 'MODELS_MEMORY_MIN_INTERVAL_SECONDS', SHORT_INTERVAL_SECONDS)
    async with FakeGateway(capabilities=MEMORY_CAPABLE) as gateway:
        async with Running(gateway, backend):
            await gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            backend.in_memory[MODEL] = None
            await gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            await gateway.drop_connection()
            await wait_until(lambda: len(gateway.connections) == 2)
            await gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            await asyncio.sleep(QUIET_SECONDS)
    in_memory = [line for line in lines(caplog) if line.startswith('In memory now')]
    assert in_memory == ['In memory now: ollama none', f'In memory now: ollama {MODEL}']


async def test_a_model_to_load_for_a_prompt_is_shown(backend, caplog):
    async with FakeGateway(capabilities=[protocol.CAPABILITY_RESPONSE_PROGRESS]) as gateway:
        async with Running(gateway, backend):
            # the answer waits as long as a model takes to load
            backend.hold_headers.clear()
            await gateway.send_request('req-1', CHAT)
            await gateway.next_frame(protocol.FRAME_RESPONSE_PROGRESS)
            await wait_until(lambda: len(lines_of_request(caplog, 1)) == 2)
            backend.hold_headers.set()
            await gateway.collect_response('req-1')
            await wait_until(lambda: len(lines_of_request(caplog, 1)) == 3)
    assert lines_of_request(caplog, 1)[1] == events.model_not_in_memory(1, 'ollama', MODEL)


async def test_the_name_is_not_signed_it_is_only_shown(gateway, backend, caplog):
    """A name added after signing changes nothing the client does: the request is relayed as signed."""
    async with Running(gateway, backend):
        frame = gateway.request_frame('req-1', CHAT)
        frame[protocol.REQUEST_USER_FIELD] = USER_NAME
        await gateway.send(frame)
        _, _, final = await gateway.collect_response('req-1')
    assert final['type'] == protocol.FRAME_RESPONSE_END


# ----- the end line comes however the answer ends -----

class FailingSend:
    """The send of a connection that closes once the answer has begun."""

    def __init__(self):
        self.frames = []

    async def __call__(self, frame: dict) -> None:
        if frame['type'] == protocol.FRAME_RESPONSE_CHUNK:
            raise ConnectionClosed(None, None)
        self.frames.append(frame)


def _relay(backend: FakeBackend, http: httpx.AsyncClient, send) -> Relay:
    return Relay(make_config('ws://127.0.0.1:9/ws/', [ollama_backend(backend)]), http, send, NONCE)


def _signed(request_id: str) -> dict:
    raw = json.dumps(CHAT).encode()
    return {'type': protocol.FRAME_REQUEST, 'id': request_id, 'backend': 'ollama', 'method': 'POST',
            'path': '/chat/completions', 'body': protocol.encode_body(raw),
            'signature': protocol.sign_request(SIGNING_SECRET, NONCE, request_id, 'ollama', 'POST',
                                               '/chat/completions', raw)}


async def test_a_connection_that_closes_while_the_answer_is_sent_ends_it_as_cancelled(backend, caplog):
    send = FailingSend()
    async with httpx.AsyncClient() as http:
        relay = _relay(backend, http, send)
        await relay.handle_request(_signed('req-1'), 0)
        await wait_until(lambda: relay.in_flight() == 0)
    end = lines_of_request(caplog, 1)[-1]
    assert end.startswith('#1 cancelled after ') and end.endswith(f': {events.CONNECTION_ENDED}')
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING
                and record.name == CLIENT_LOGGER]
    # nobody is left to answer with an error frame
    assert protocol.FRAME_RESPONSE_ERROR not in [frame['type'] for frame in send.frames]


async def test_a_request_cancelled_before_it_ran_still_ends(backend, caplog):
    async with httpx.AsyncClient() as http:
        relay = _relay(backend, http, FailingSend())
        await relay.handle_request(_signed('req-1'), 0)
        relay.handle_cancel({'type': protocol.FRAME_CANCEL, 'id': 'req-1'})
        await wait_until(lambda: relay.in_flight() == 0)
    assert lines_of_request(caplog, 1)[-1].startswith('#1 cancelled after ')
    assert lines_of_request(caplog, 1)[-1].endswith(f': {events.STOPPED_BY_GATEWAY}')
