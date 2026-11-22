"""What the running client tells Ollama about its memory: after answers through the tunnel, and at start."""
import asyncio
import logging
import time

import pytest

from socketllama import client as client_module
from socketllama import protocol
from socketllama.client import FatalGatewayError, TunnelClient

from .conftest import STARTUP_TIMEOUT_SECONDS, litellm_backend, make_config, ollama_backend, wait_until
from .fakes import EMBEDDING_MODEL, HTTP_NOT_FOUND, HTTP_OK, MODEL, FakeBackend, FakeGateway

CHAT = {'model': MODEL, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}
KEEP_LOADED = '2h'
SECOND_MODEL = 'qwen3:14b'
GENERATE = '/api/generate'
CHAT_COMPLETIONS = '/v1/chat/completions'
PROCESS_LIST = '/api/ps'
MODEL_LIST = '/v1/models'
KEEP_MODEL_LOADED = {'model': MODEL, 'keep_alive': KEEP_LOADED}
QUIET_SECONDS = 0.2
CLIENT_LOGGER = 'socketllama'
PROGRESS_CAPABLE = [protocol.CAPABILITY_RESPONSE_PROGRESS, protocol.CAPABILITY_MODELS_UNCHECKED]
# Watching this many of the (shortened) intervals between reports for one that should not come
REPORT_INTERVALS_WATCHED = 2
# How much later than the client sent it the first report reaches the gateway, which stores it then:
# less than the (shortened) interval in which the client waits for the acknowledgement
SLOW_ARRIVAL_SECONDS = QUIET_SECONDS / 2
MODELS_ACK = {'type': protocol.FRAME_MODELS_ACK, 'created': [MODEL]}
REPORT_IGNORED = {'type': protocol.FRAME_ERROR,
                  'message': 'Models report ignored: reports are accepted at most every 60 seconds'}


def paths(backend: FakeBackend) -> list:
    return [request['path'] for request in backend.requests]


def sent(backend: FakeBackend, path: str) -> list:
    return [request['json'] for request in backend.requests if request['path'] == path]


class Running:
    """A client connected to a fake gateway with one backend, stopped at the end."""

    def __init__(self, gateway: FakeGateway, backend_config: dict):
        self.client = TunnelClient(make_config(gateway.url, [backend_config]))
        self._gateway = gateway

    async def __aenter__(self):
        self.task = asyncio.create_task(self.client.run_forever())
        await wait_until(lambda: self._gateway.connections)
        await self._gateway.next_frame(protocol.FRAME_MODELS)
        return self

    async def __aexit__(self, *exc_info):
        self.client.stop()
        await asyncio.wait_for(self.task, STARTUP_TIMEOUT_SECONDS)


# ----- after an answer -----

@pytest.mark.parametrize('stream', [True, False])
async def test_after_an_answer_ollama_is_told_how_long_the_model_stays(gateway, stream):
    async with FakeBackend(loads_on_chat=True) as backend:
        async with Running(gateway, ollama_backend(backend, keep_loaded=KEEP_LOADED)):
            await gateway.send_request('req-1', {**CHAT, 'stream': stream})
            _, _, final = await gateway.collect_response('req-1')
            assert final['type'] == protocol.FRAME_RESPONSE_END
            await wait_until(lambda: backend.in_memory == {MODEL: KEEP_LOADED})
    assert sent(backend, GENERATE) == [KEEP_MODEL_LOADED]
    # once the relayed request was over: the request Ollama got last decides
    assert paths(backend).index(CHAT_COMPLETIONS) < paths(backend).index(GENERATE)
    assert sent(backend, CHAT_COMPLETIONS) == [{**CHAT, 'stream': stream}]  # the relayed request is untouched


async def test_the_answer_does_not_wait_for_it(gateway):
    async with FakeBackend(loads_on_chat=True) as backend:
        backend.hold_generate.clear()
        async with Running(gateway, ollama_backend(backend, keep_loaded=KEEP_LOADED)):
            await gateway.send_request('req-1', CHAT)
            _, _, final = await gateway.collect_response('req-1')
            assert final['type'] == protocol.FRAME_RESPONSE_END
            await wait_until(lambda: GENERATE in paths(backend))
            assert backend.in_memory == {MODEL: None}  # Ollama has not answered the extra request yet
            await gateway.send_request('req-2', CHAT)   # and the client relays on meanwhile
            assert (await gateway.collect_response('req-2'))[2]['type'] == protocol.FRAME_RESPONSE_END
            backend.hold_generate.set()


async def test_an_answer_cut_short_by_a_cancel_still_leaves_the_model_with_the_configured_time(gateway):
    async with FakeBackend(loads_on_chat=True) as backend:
        backend.hold_stream.clear()
        async with Running(gateway, ollama_backend(backend, keep_loaded=KEEP_LOADED)):
            await gateway.send_request('req-1', CHAT)
            await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
            await gateway.send({'type': protocol.FRAME_CANCEL, 'id': 'req-1'})
            await wait_until(lambda: backend.in_memory == {MODEL: KEEP_LOADED})
            backend.hold_stream.set()
    assert sent(backend, GENERATE) == [KEEP_MODEL_LOADED]


async def test_a_request_cancelled_before_ollama_answered_loads_nothing(gateway):
    async with FakeBackend(loads_on_chat=True) as backend:
        backend.hold_headers.clear()
        async with Running(gateway, ollama_backend(backend, keep_loaded=KEEP_LOADED)):
            await gateway.send_request('req-1', CHAT)
            await wait_until(lambda: CHAT_COMPLETIONS in paths(backend))
            await gateway.send({'type': protocol.FRAME_CANCEL, 'id': 'req-1'})
            await asyncio.sleep(QUIET_SECONDS)
    assert GENERATE not in paths(backend)


async def test_a_request_ollama_refused_loads_nothing(gateway):
    async with FakeBackend(loads_on_chat=True) as backend:
        async with Running(gateway, ollama_backend(backend, keep_loaded=KEEP_LOADED)):
            await gateway.send_request('req-1', {**CHAT, 'model': 'missing'})
            start, _, _ = await gateway.collect_response('req-1')
            assert start['status'] == HTTP_NOT_FOUND
            await asyncio.sleep(QUIET_SECONDS)
    assert GENERATE not in paths(backend)


async def test_a_model_ollama_unloaded_right_after_the_answer_is_not_loaded_again(gateway):
    """Ollama set to unload at once, or another model took its place: the fake does not load on a chat."""
    async with FakeBackend() as backend:
        async with Running(gateway, ollama_backend(backend, keep_loaded=KEEP_LOADED)):
            await gateway.send_request('req-1', CHAT)
            await gateway.collect_response('req-1')
            await wait_until(lambda: paths(backend).count(PROCESS_LIST) == 2)  # discovery, and this
            await asyncio.sleep(QUIET_SECONDS)
    assert GENERATE not in paths(backend)


async def test_a_model_that_is_not_shared_is_neither_asked_nor_kept(gateway):
    async with FakeBackend(loads_on_chat=True) as backend:
        async with Running(gateway, ollama_backend(backend, models=[MODEL], keep_loaded=KEEP_LOADED)):
            await gateway.send_request('req-1', {**CHAT, 'model': EMBEDDING_MODEL})
            assert (await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR))['code'] == protocol.ERROR_MODEL_NOT_ALLOWED
            await asyncio.sleep(QUIET_SECONDS)
    assert GENERATE not in paths(backend) and CHAT_COMPLETIONS not in paths(backend)


async def test_without_the_setting_ollama_gets_what_it_always_got(gateway):
    async with FakeBackend(loads_on_chat=True) as backend:
        async with Running(gateway, ollama_backend(backend)):
            await gateway.send_request('req-1', CHAT)
            await gateway.collect_response('req-1')
            await asyncio.sleep(QUIET_SECONDS)
    assert GENERATE not in paths(backend)
    assert paths(backend).count(PROCESS_LIST) == 1  # discovery only
    assert backend.in_memory == {MODEL: None}


async def test_other_kinds_of_backends_get_what_they_always_got(gateway):
    async with FakeBackend(loads_on_chat=True) as backend:
        async with Running(gateway, litellm_backend(backend)):
            await gateway.send_request('req-1', CHAT, backend='litellm')
            await gateway.collect_response('req-1')
            await asyncio.sleep(QUIET_SECONDS)
    assert GENERATE not in paths(backend) and PROCESS_LIST not in paths(backend)


async def test_two_answers_of_one_model_at_a_time_both_tell_ollama(gateway):
    async with FakeBackend(loads_on_chat=True) as backend:
        backend.hold_stream.clear()
        async with Running(gateway, ollama_backend(backend, keep_loaded=KEEP_LOADED)):
            await gateway.send_request('req-1', CHAT)
            await gateway.send_request('req-2', CHAT)
            await wait_until(lambda: paths(backend).count(CHAT_COMPLETIONS) == 2)
            backend.hold_stream.set()
            await wait_until(lambda: sent(backend, GENERATE) == [KEEP_MODEL_LOADED] * 2)
            await wait_until(lambda: backend.in_memory == {MODEL: KEEP_LOADED})


async def test_ollama_refusing_the_extra_request_costs_no_answer_and_one_line_in_the_log(gateway, caplog):
    async with FakeBackend(loads_on_chat=True, models_that_do_not_fit=[MODEL]) as backend:
        async with Running(gateway, ollama_backend(backend, keep_loaded=KEEP_LOADED)):
            with caplog.at_level(logging.WARNING, logger=CLIENT_LOGGER):
                for request_id in ('req-1', 'req-2', 'req-3'):
                    await gateway.send_request(request_id, CHAT)
                    start, _, final = await gateway.collect_response(request_id)
                    assert (start['status'], final['type']) == (HTTP_OK, protocol.FRAME_RESPONSE_END)
                await wait_until(lambda: paths(backend).count(GENERATE) == 3)
                await asyncio.sleep(QUIET_SECONDS)
    told = [record for record in caplog.records if 'could not tell Ollama' in record.getMessage()]
    assert len(told) == 1


# ----- at start -----

async def test_the_models_listed_are_loaded_once_the_first_report_is_on_its_way(gateway, caplog):
    async with FakeBackend(more_models=[SECOND_MODEL]) as backend:
        config = ollama_backend(backend, models=[MODEL, SECOND_MODEL], keep_loaded=KEEP_LOADED, load_at_start=True)
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            async with Running(gateway, config):
                await wait_until(lambda: backend.in_memory == {MODEL: KEEP_LOADED, SECOND_MODEL: KEEP_LOADED})
    assert sent(backend, GENERATE) == [KEEP_MODEL_LOADED, {'model': SECOND_MODEL, 'keep_alive': KEEP_LOADED}]
    # the report was built from what discovery saw before anything was loaded
    assert paths(backend).index(MODEL_LIST) < paths(backend).index(GENERATE)
    assert f'Backend ollama: {SECOND_MODEL} is in memory after' in caplog.text


async def test_once_they_are_loaded_the_gateway_gets_a_fresh_report(gateway, monkeypatch):
    """Not only at the next refresh, minutes later: the context window Ollama gave a loaded model is known now."""
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', QUIET_SECONDS)
    async with FakeBackend() as backend:
        async with Running(gateway, ollama_backend(backend, models=[MODEL], load_at_start=True)):
            await gateway.send(MODELS_ACK)  # no resend for want of it
            fresh = await gateway.next_frame(protocol.FRAME_MODELS)  # well before MODEL_REFRESH_INTERVAL_SECONDS
            await gateway.send(MODELS_ACK)
            await asyncio.sleep(QUIET_SECONDS * REPORT_INTERVALS_WATCHED)
            assert [frame for frame in gateway.drain_frames() if frame['type'] == protocol.FRAME_MODELS] == []
    assert fresh['backends'][0]['models']
    # built after the loading, from what Ollama has in memory then
    last_discovery = max(index for index, path in enumerate(paths(backend)) if path == MODEL_LIST)
    assert last_discovery > paths(backend).index(GENERATE)


async def test_the_fresh_report_reaches_the_gateway_once_it_stores_one_again(gateway, monkeypatch):
    """The gateway times its limit from when a report arrived, on its own clock: from the moment the
    client sent the last one, a report that took longer to get there would look too early."""
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', QUIET_SECONDS)
    async with FakeBackend() as backend:
        async with Running(gateway, ollama_backend(backend, models=[MODEL], load_at_start=True)):
            await asyncio.sleep(SLOW_ARRIVAL_SECONDS)
            stored_at = time.monotonic()
            await gateway.send(MODELS_ACK)
            await gateway.next_frame(protocol.FRAME_MODELS)
            assert time.monotonic() - stored_at >= QUIET_SECONDS  # what the gateway's limit asks for


async def test_a_fresh_report_the_gateway_ignored_is_sent_once_more(gateway, monkeypatch):
    """Like the first report of a connection: one that is not acknowledged is sent again, once."""
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', QUIET_SECONDS)
    async with FakeBackend() as backend:
        async with Running(gateway, ollama_backend(backend, models=[MODEL], load_at_start=True)):
            await gateway.send(MODELS_ACK)
            await gateway.next_frame(protocol.FRAME_MODELS)
            await gateway.send(REPORT_IGNORED)
            resent = await gateway.next_frame(protocol.FRAME_MODELS)  # well before MODEL_REFRESH_INTERVAL_SECONDS
            await asyncio.sleep(QUIET_SECONDS * REPORT_INTERVALS_WATCHED)
            assert [frame for frame in gateway.drain_frames() if frame['type'] == protocol.FRAME_MODELS] == []
    assert resent['backends'][0]['models']


async def test_without_load_at_start_no_fresh_report_follows(gateway, monkeypatch):
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', QUIET_SECONDS)
    async with FakeBackend() as backend:
        async with Running(gateway, ollama_backend(backend, models=[MODEL], keep_loaded=KEEP_LOADED)):
            await gateway.send({'type': protocol.FRAME_MODELS_ACK, 'created': [MODEL]})
            await asyncio.sleep(QUIET_SECONDS * REPORT_INTERVALS_WATCHED)
            assert [frame for frame in gateway.drain_frames() if frame['type'] == protocol.FRAME_MODELS] == []


async def test_a_reconnect_does_not_load_them_again(gateway):
    async with FakeBackend() as backend:
        async with Running(gateway, ollama_backend(backend, models=[MODEL], load_at_start=True)):
            await wait_until(lambda: MODEL in backend.in_memory)
            await gateway.drop_connection()
            await wait_until(lambda: len(gateway.connections) == 2)
            await asyncio.sleep(QUIET_SECONDS)
    assert sent(backend, GENERATE) == [{'model': MODEL}]


async def test_nothing_is_loaded_for_a_client_the_gateway_turns_away():
    refusal = {'type': protocol.FRAME_AUTH_FAILED, 'reason': 'invalid_token'}
    async with FakeBackend() as backend, FakeGateway(auth_answer=refusal) as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend, models=[MODEL], load_at_start=True)]))
        with pytest.raises(FatalGatewayError):
            await asyncio.wait_for(client.run_forever(), STARTUP_TIMEOUT_SECONDS)
    assert backend.requests == []


async def test_stopping_does_not_wait_for_a_model_that_is_still_loading(gateway):
    async with FakeBackend() as backend:
        backend.hold_generate.clear()
        async with Running(gateway, ollama_backend(backend, models=[MODEL], load_at_start=True)) as running:
            await wait_until(lambda: GENERATE in paths(backend))
        # stopped within the time allowed for it, while Ollama had not answered the load
        assert running.task.done() and backend.in_memory == {}


async def test_requests_are_relayed_while_models_are_still_loading(gateway):
    async with FakeBackend() as backend:
        backend.hold_generate.clear()
        async with Running(gateway, ollama_backend(backend, models=[MODEL], load_at_start=True)):
            await wait_until(lambda: GENERATE in paths(backend))
            await gateway.send_request('req-1', CHAT)
            assert (await gateway.collect_response('req-1'))[2]['type'] == protocol.FRAME_RESPONSE_END


async def test_a_model_loaded_at_start_is_reported_as_in_memory_when_a_request_comes():
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, ollama_backend(backend, models=[MODEL], load_at_start=True)):
            await wait_until(lambda: MODEL in backend.in_memory)
            backend.hold_headers.clear()
            await gateway.send_request('req-1', CHAT)
            await asyncio.sleep(QUIET_SECONDS)
            backend.hold_headers.set()
            stages = []
            while (frame := await gateway.next_frame())['type'] != protocol.FRAME_RESPONSE_START:
                stages.append(frame['stage'])
    assert stages == [protocol.PROGRESS_STAGE_RECEIVED]


async def test_without_load_at_start_nothing_is_loaded(gateway):
    async with FakeBackend() as backend:
        async with Running(gateway, ollama_backend(backend, models=[MODEL], keep_loaded=KEEP_LOADED)):
            await asyncio.sleep(QUIET_SECONDS)
    assert GENERATE not in paths(backend)
