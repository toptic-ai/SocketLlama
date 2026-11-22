"""A model that takes long to load: the client has Ollama finish a load whose request was closed, waits for
the start of an answer as long as the gateway does, and tells the gateway when Ollama refuses a prompt
because it is still loading."""
import asyncio
import json
import logging

import pytest

from socketllama import events, protocol
from socketllama import ollama_memory
from socketllama.client import TunnelClient
from socketllama.config import DEFAULT_BACKEND_READ_TIMEOUT_SECONDS, KIND_OLLAMA, Backend
from socketllama.ollama_memory import (
    OLLAMA_LOADING_ANSWER_MAX_BYTES, OLLAMA_LOADING_ANSWER_MAX_SECONDS, says_model_is_loading,
)
from socketllama.relay import MEMORY_CHECK_TIMEOUT_SECONDS
from socketllama.units import SECONDS_PER_HOUR

from .conftest import STARTUP_TIMEOUT_SECONDS, litellm_backend, make_config, ollama_backend, wait_until
from .fakes import (
    CONTEXT_LENGTH, HTTP_NOT_FOUND, HTTP_OK, HTTP_SERVER_ERROR, INSTANCE, JSON_CONTENT_TYPE, MODEL, STREAM_PARTS,
    FakeBackend, FakeGateway, ManualTimeouts, sse_text,
)

CHAT = {'model': MODEL, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}
OLLAMA = 'ollama'
LITELLM = 'litellm'
LITELLM_MODEL = 'gpt-proxy'
KEEP_LOADED = '2h'
UNLOAD_AT_ONCE = 0
GENERATE = '/api/generate'
CHAT_COMPLETIONS = '/v1/chat/completions'
PROCESS_LIST = '/api/ps'
REQUEST = 'req-1'
SECOND_REQUEST = 'req-2'
# The id of a request the client refuses at once (see frames_so_far)
MARKER_REQUEST = 'marker'
NOT_THE_SECRET = 'not-the-signing-secret'
# The numbers the client gives the first and the second request it relays
FIRST = events.FIRST_REQUEST_NUMBER
SECOND = FIRST + 1
OUTCOMES = (events.OUTCOME_DONE, events.OUTCOME_CANCELLED, events.OUTCOME_ERROR)
REFUSED_LINE_START = 'Refused a request'
IN_MEMORY_AFTER = ' is in memory after '
# The waits of relaying and of the client's own loads run out when a test says so (ManualTimeouts), so these
# are the seconds of real use: a
# backend's read_timeout unless its configuration says otherwise, what toptic.ai says it waits while a model
# loads, more than the client goes along with, and the short time an owner gave a backend
READ_TIMEOUT = DEFAULT_BACKEND_READ_TIMEOUT_SECONDS
ANNOUNCED_SECONDS = 900
BEYOND_THE_LONGEST_SECONDS = protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS + ANNOUNCED_SECONDS
OWNERS_SHORT_READ_TIMEOUT = 60
# What a gateway may say that is no wait, and the start of the warning about it
NOT_A_WAIT = 'soon'
IGNORED_WAIT = 'Ignored how long toptic.ai says it waits'
# How often a test has the gateway drop the connection, to see what is said again
RECONNECTS = 2
CLIENT_LOGGER = 'socketllama'
PROGRESS_CAPABLE = [protocol.CAPABILITY_RESPONSE_PROGRESS]
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_CONFLICT = 409
TEXT_CONTENT_TYPE = 'text/plain'
# What Ollama says while a model is loading, on its OpenAI-compatible endpoint and on its own endpoints
LOADING_MESSAGE = 'unexpected server status: llm server loading model'
LOADING_BODY = json.dumps({'error': {'message': LOADING_MESSAGE, 'type': 'api_error', 'param': None,
                                     'code': None}}).encode()
LOADING_BODY_OF_OLLAMAS_OWN_ENDPOINTS = json.dumps({'error': LOADING_MESSAGE}).encode()
OTHER_FAILURE_BODY = json.dumps({'error': {'message': 'model requires more system memory (12.3 GiB) than is '
                                                      'available (7.9 GiB)', 'type': 'api_error'}}).encode()
# The loading message in a body too large to be the answer that is meant
OVERSIZED_LOADING_BODY = json.dumps({'error': {'message': LOADING_MESSAGE},
                                     'padding': 'x' * OLLAMA_LOADING_ANSWER_MAX_BYTES}).encode()
NOT_JSON_LOADING_BODY = f'<html>{LOADING_MESSAGE}</html>'.encode()
# Where a body is cut to send it in parts
PART_BYTES = 16


def in_parts(body: bytes) -> list[bytes]:
    """``body`` cut in three, as a server sends it that writes it piece by piece."""
    return [body[:PART_BYTES], body[PART_BYTES:PART_BYTES * 2], body[PART_BYTES * 2:]]


def paths(backend: FakeBackend) -> list:
    return [request['path'] for request in backend.requests]


def loads(backend: FakeBackend) -> list:
    """The JSON bodies of the load requests the fake Ollama got."""
    return [request['json'] for request in backend.requests if request['path'] == GENERATE]


def shown(caplog, level: int = logging.INFO) -> list[str]:
    """What the client showed at exactly ``level``, but for requests it refused (see frames_so_far)."""
    return [record.getMessage() for record in caplog.records
            if record.name == CLIENT_LOGGER and record.levelno == level
            and not record.getMessage().startswith(REFUSED_LINE_START)]


def end_lines(caplog, number: int) -> list[str]:
    """The line that says how request ``number`` ended, once it has."""
    starts = tuple(f'{events.REQUEST_NUMBER_MARK}{number} {outcome} after ' for outcome in OUTCOMES)
    return [record.getMessage() for record in caplog.records
            if record.name == CLIENT_LOGGER and record.getMessage().startswith(starts)]


async def ended(caplog, number: int) -> str:
    """Wait until request ``number`` is over for the client; its end line. Whatever the client does about
    how a request ended, it has done or started by then."""
    await wait_until(lambda: end_lines(caplog, number))
    return end_lines(caplog, number)[0]


async def frames_so_far(gateway: FakeGateway, request_id: str) -> list:
    """The frames the client sent about a request that the gateway has not read yet, all of them: read up
    to the client's answer to a request sent now, which it refuses at once and answers after them."""
    await gateway.send_request(MARKER_REQUEST, CHAT, secret=NOT_THE_SECRET)
    frames = []
    while (frame := await gateway.next_frame()).get('id') != MARKER_REQUEST:
        if frame.get('id') == request_id:
            frames.append(frame)
    return frames


def announcing(seconds: object, capabilities: tuple = ()) -> FakeGateway:
    """A gateway whose auth_ok says how long it waits while a model loads."""
    return FakeGateway(auth_answer={'type': protocol.FRAME_AUTH_OK, 'instance': INSTANCE,
                                    'protocol': protocol.PROTOCOL_VERSION, 'capabilities': list(capabilities),
                                    protocol.MODEL_LOADING_TIMEOUT_FIELD: seconds})


class Running:
    """A client connected to a fake gateway with one backend, stopped at the end."""

    def __init__(self, gateway: FakeGateway, backend_config: dict, timeouts: ManualTimeouts | None = None):
        """``timeouts``: what bounds the waits of relaying, for a test that says when they run out."""
        options = {} if timeouts is None else {'timeout': timeouts}
        self.client = TunnelClient(make_config(gateway.url, [backend_config]), **options)
        self._gateway = gateway

    async def __aenter__(self):
        self.task = asyncio.create_task(self.client.run_forever())
        await wait_until(lambda: self._gateway.connections)
        await self._gateway.next_frame(protocol.FRAME_MODELS)
        return self

    async def __aexit__(self, *exc_info):
        self.client.stop()
        await asyncio.wait_for(self.task, STARTUP_TIMEOUT_SECONDS)

    def loads_on_their_way(self) -> dict:
        """The loads the client has asked its memory component for that are not over."""
        return self.client._memory._finishing


def shared(backend: FakeBackend, **settings) -> dict:
    """An Ollama backend that shares MODEL alone (sharing every model is a warning at start)."""
    return ollama_backend(backend, models=[MODEL], **settings)


@pytest.fixture(autouse=True)
def info_logged(caplog):
    caplog.set_level(logging.INFO, logger=CLIENT_LOGGER)


async def waiting_for_its_model(gateway: FakeGateway, fake: FakeBackend, request_id: str = REQUEST,
                                stage: str = protocol.PROGRESS_STAGE_MODEL_NOT_IN_MEMORY, payload: dict = CHAT,
                                **request):
    """Send a prompt the backend does not answer (as while a model loads): up to its last progress ``stage``,
    and until the backend has the request."""
    fake.hold_headers.clear()
    chats = paths(fake).count(CHAT_COMPLETIONS)
    await gateway.send_request(request_id, payload, **request)
    while (await gateway.next_frame(protocol.FRAME_RESPONSE_PROGRESS))['stage'] != stage:
        pass
    await wait_until(lambda: paths(fake).count(CHAT_COMPLETIONS) > chats)


async def cancel(gateway: FakeGateway, request_id: str = REQUEST):
    await gateway.send({'type': protocol.FRAME_CANCEL, 'id': request_id})


# ----- a load whose request was closed is finished by the client -----

async def test_a_request_cancelled_while_its_model_loads_has_the_client_load_it_and_say_so(caplog):
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)) as running:
            await waiting_for_its_model(gateway, backend)
            await cancel(gateway)
            await wait_until(lambda: any(IN_MEMORY_AFTER in line for line in shown(caplog)))
            assert await frames_so_far(gateway, REQUEST) == []  # the cancelled request is answered nothing
            assert not running.loads_on_their_way()
    assert loads(backend) == [{'model': MODEL}] and backend.in_memory == {MODEL: None}
    assert paths(backend).index(CHAT_COMPLETIONS) < paths(backend).index(GENERATE)
    assert events.load_goes_on(FIRST, OLLAMA, MODEL) in shown(caplog)
    [result] = [line for line in shown(caplog) if IN_MEMORY_AFTER in line]
    assert result.startswith(f'{OLLAMA}/{MODEL}{IN_MEMORY_AFTER}')
    assert result.endswith(events.load_finished(FIRST, OLLAMA, MODEL, 0.0).split(' s: ')[-1])
    assert shown(caplog, logging.WARNING) == []


async def test_what_is_in_memory_is_asked_before_the_prompt_goes_to_ollama():
    """Ollama lists a model from the moment it starts to load it: asked meanwhile, the model this very
    prompt makes it load could pass for one that was in memory."""
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)):
            asked_before = paths(backend).count(PROCESS_LIST)
            await gateway.send_request(REQUEST, CHAT)
            await gateway.collect_response(REQUEST)
    of_the_request = paths(backend)[len(paths(backend)) - paths(backend)[::-1].index(CHAT_COMPLETIONS) - 1:]
    assert paths(backend).count(PROCESS_LIST) == asked_before + 1 and PROCESS_LIST not in of_the_request


async def test_an_ollama_that_does_not_tell_what_is_in_memory_holds_no_prompt_up_for_long(caplog):
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            backend.hold_process_list.clear()
            await gateway.send_request(REQUEST, CHAT)
            await timeouts.run_out(MEMORY_CHECK_TIMEOUT_SECONDS)
            assert CHAT_COMPLETIONS not in paths(backend)
            stages = []
            while (frame := await gateway.next_frame())['type'] == protocol.FRAME_RESPONSE_PROGRESS:
                stages.append(frame['stage'])
            assert frame['type'] == protocol.FRAME_RESPONSE_START and frame['status'] == HTTP_OK
            assert (await ended(caplog, FIRST)).startswith(f'{events.REQUEST_NUMBER_MARK}{FIRST} {events.OUTCOME_DONE}')
    assert stages == [protocol.PROGRESS_STAGE_RECEIVED]  # nothing is claimed about a memory that is unknown
    assert shown(caplog, logging.WARNING) == []


async def test_the_model_is_loaded_to_stay_as_long_as_the_backend_keeps_models():
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend, keep_loaded=KEEP_LOADED)):
            await waiting_for_its_model(gateway, backend)
            await cancel(gateway)
            await wait_until(lambda: backend.in_memory == {MODEL: KEEP_LOADED})
    assert loads(backend) == [{'model': MODEL, 'keep_alive': KEEP_LOADED}]


async def test_a_backend_that_lets_no_model_stay_has_none_loaded(caplog):
    """With keep_loaded = 0 the same request would unload the model."""
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend, keep_loaded=UNLOAD_AT_ONCE)) as running:
            await waiting_for_its_model(gateway, backend)
            await cancel(gateway)
            await ended(caplog, FIRST)
            assert not running.loads_on_their_way()
    assert loads(backend) == []


async def test_a_connection_that_ends_while_the_model_loads_has_the_client_load_it_too(caplog):
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)):
            await waiting_for_its_model(gateway, backend)
            await gateway.drop_connection()
            await wait_until(lambda: backend.in_memory == {MODEL: None})
    assert loads(backend) == [{'model': MODEL}]
    assert events.load_goes_on(FIRST, OLLAMA, MODEL) in shown(caplog)


async def test_a_load_that_fails_is_a_warning_with_ollamas_reason_and_nothing_more(caplog):
    async with FakeBackend(models_that_do_not_fit=[MODEL]) as backend, \
            FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)):
            await waiting_for_its_model(gateway, backend)
            await cancel(gateway)
            await wait_until(lambda: shown(caplog, logging.WARNING))
            backend.hold_headers.set()
            await gateway.send_request(SECOND_REQUEST, CHAT)  # the client relays on
            _, _, final = await gateway.collect_response(SECOND_REQUEST)
            assert final['type'] == protocol.FRAME_RESPONSE_END
    [warning] = shown(caplog, logging.WARNING)
    assert warning.startswith(events.load_failed(FIRST, OLLAMA, MODEL, '').split(': ')[0])
    assert 'requires more system memory (12.3 GiB) than is available (7.9 GiB)' in warning
    assert not [record for record in caplog.records if record.levelno > logging.WARNING]


async def test_a_load_that_outlasts_the_clients_wait_says_which_setting_to_raise(caplog):
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend, read_timeout=OWNERS_SHORT_READ_TIMEOUT), timeouts):
            backend.hold_generate.clear()
            await waiting_for_its_model(gateway, backend)
            await cancel(gateway)
            # the load is on its way, bounded as a whole by the backend's read_timeout
            await wait_until(lambda: GENERATE in paths(backend))
            await timeouts.run_out(OWNERS_SHORT_READ_TIMEOUT)
            await wait_until(lambda: shown(caplog, logging.WARNING))
            backend.hold_generate.set()
    [warning] = shown(caplog, logging.WARNING)
    assert f'did not answer within {OWNERS_SHORT_READ_TIMEOUT:g} s' in warning and 'read_timeout' in warning


async def test_a_model_that_was_in_memory_is_not_loaded(caplog):
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH) as backend, \
            FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)) as running:
            # the backend has the prompt: the client asked what is in memory before it sent it
            await waiting_for_its_model(gateway, backend, stage=protocol.PROGRESS_STAGE_RECEIVED)
            await cancel(gateway)
            await ended(caplog, FIRST)
            assert not running.loads_on_their_way()
    assert loads(backend) == []


async def test_a_request_cancelled_once_ollama_began_its_answer_loads_nothing(caplog):
    """The model is in memory then, whatever was found when the request came."""
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)) as running:
            backend.hold_stream.clear()
            await waiting_for_its_model(gateway, backend)
            backend.hold_headers.set()
            await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
            await cancel(gateway)
            await ended(caplog, FIRST)
            assert not running.loads_on_their_way()
            backend.hold_stream.set()
    assert events.model_not_in_memory(FIRST, OLLAMA, MODEL) in shown(caplog)
    assert loads(backend) == []


async def test_a_backend_of_another_kind_has_nothing_loaded(caplog):
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, litellm_backend(backend, models=[LITELLM_MODEL])) as running:
            await waiting_for_its_model(gateway, backend, stage=protocol.PROGRESS_STAGE_RECEIVED,
                                        payload={**CHAT, 'model': LITELLM_MODEL}, backend=LITELLM)
            await cancel(gateway)
            await ended(caplog, FIRST)
            assert not running.loads_on_their_way()
    assert GENERATE not in paths(backend) and PROCESS_LIST not in paths(backend)


async def test_without_knowing_what_was_in_memory_nothing_is_loaded(gateway, caplog):
    """A gateway that takes no progress: the client did not ask Ollama what it has in memory."""
    async with FakeBackend() as backend:
        async with Running(gateway, shared(backend)) as running:
            backend.hold_headers.clear()
            await gateway.send_request(REQUEST, CHAT)
            await wait_until(lambda: CHAT_COMPLETIONS in paths(backend))
            await cancel(gateway)
            await ended(caplog, FIRST)
            assert not running.loads_on_their_way()
    assert loads(backend) == []


async def test_two_cancelled_requests_for_one_model_are_one_load(caplog):
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)) as running:
            backend.hold_generate.clear()
            await waiting_for_its_model(gateway, backend)
            await cancel(gateway)
            await wait_until(lambda: GENERATE in paths(backend))
            await waiting_for_its_model(gateway, backend, SECOND_REQUEST)
            await cancel(gateway, SECOND_REQUEST)
            await ended(caplog, SECOND)
            assert events.load_already_going_on(SECOND, OLLAMA, MODEL) in shown(caplog)
            backend.hold_generate.set()
            await wait_until(lambda: not running.loads_on_their_way())
    assert loads(backend) == [{'model': MODEL}]
    assert len([line for line in shown(caplog) if IN_MEMORY_AFTER in line]) == 1


async def test_stopping_the_client_ends_a_load_it_asked_for(caplog):
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)) as running:
            backend.hold_generate.clear()
            await waiting_for_its_model(gateway, backend)
            await cancel(gateway)
            await wait_until(lambda: GENERATE in paths(backend))
        # stopped within the time allowed for it, while Ollama had not answered the load
        assert running.task.done() and backend.in_memory == {}
        assert not running.loads_on_their_way()  # dropped: nothing is left that could report an end
    assert not any(IN_MEMORY_AFTER in line for line in shown(caplog))
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


async def test_stopping_the_client_while_a_model_loads_for_a_request_starts_no_load(caplog):
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)):
            await waiting_for_its_model(gateway, backend)
        await ended(caplog, FIRST)
    assert events.load_goes_on(FIRST, OLLAMA, MODEL) not in shown(caplog)
    assert loads(backend) == []


# ----- the client's own wait for the start of an answer -----

async def test_a_request_the_client_gives_up_waiting_for_has_its_model_loaded_like_a_cancelled_one(caplog):
    """Closing the request ends Ollama's load whoever closes it."""
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            await waiting_for_its_model(gateway, backend)
            await timeouts.run_out(READ_TIMEOUT)
            frame = await gateway.next_frame()
            await wait_until(lambda: backend.in_memory == {MODEL: None})
    assert frame == {'type': protocol.FRAME_RESPONSE_ERROR, 'id': REQUEST, 'code': protocol.ERROR_BACKEND_TIMEOUT,
                     'message': f'Backend {OLLAMA} did not begin to answer within {READ_TIMEOUT:g} s'}
    assert events.load_goes_on(FIRST, OLLAMA, MODEL) in shown(caplog)
    assert loads(backend) == [{'model': MODEL}]


async def test_giving_up_on_a_model_that_was_in_memory_loads_nothing(caplog):
    timeouts = ManualTimeouts()
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH) as backend, \
            FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts) as running:
            await waiting_for_its_model(gateway, backend, stage=protocol.PROGRESS_STAGE_RECEIVED)
            await timeouts.run_out(READ_TIMEOUT)
            assert (await gateway.next_frame())['code'] == protocol.ERROR_BACKEND_TIMEOUT
            await ended(caplog, FIRST)
            assert not running.loads_on_their_way()
    assert loads(backend) == []


async def test_for_a_model_that_was_not_in_memory_the_client_waits_as_long_as_the_gateway_says_it_does(caplog):
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, announcing(ANNOUNCED_SECONDS, PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            await waiting_for_its_model(gateway, backend)
            assert timeouts.running() == [ANNOUNCED_SECONDS]  # not the backend's shorter read_timeout
            await timeouts.run_out(ANNOUNCED_SECONDS)
            frame = await gateway.next_frame()
    assert frame['code'] == protocol.ERROR_BACKEND_TIMEOUT
    assert frame['message'] == f'Backend {OLLAMA} did not begin to answer within {ANNOUNCED_SECONDS:g} s'
    assert events.gateway_waits_for_loading(ANNOUNCED_SECONDS) in shown(caplog)


async def test_for_a_model_that_was_in_memory_the_backends_read_timeout_stays_whatever_the_gateway_says():
    """The gateway waits longer only once it was told that the model is not in memory."""
    timeouts = ManualTimeouts()
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH) as backend, \
            announcing(ANNOUNCED_SECONDS, PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            await waiting_for_its_model(gateway, backend, stage=protocol.PROGRESS_STAGE_RECEIVED)
            assert timeouts.running() == [READ_TIMEOUT]


async def test_a_backend_of_another_kind_keeps_the_time_its_owner_gave_it():
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, announcing(ANNOUNCED_SECONDS, PROGRESS_CAPABLE) as gateway:
        config = litellm_backend(backend, models=[LITELLM_MODEL], read_timeout=OWNERS_SHORT_READ_TIMEOUT)
        async with Running(gateway, config, timeouts):
            await waiting_for_its_model(gateway, backend, stage=protocol.PROGRESS_STAGE_RECEIVED,
                                        payload={**CHAT, 'model': LITELLM_MODEL}, backend=LITELLM)
            assert timeouts.running() == [OWNERS_SHORT_READ_TIMEOUT]


async def test_without_knowing_what_was_in_memory_the_backends_read_timeout_stays():
    """A gateway that takes no progress is told nothing about memory, so it does not wait longer either."""
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, announcing(ANNOUNCED_SECONDS) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            backend.hold_headers.clear()
            await gateway.send_request(REQUEST, CHAT)
            await wait_until(lambda: CHAT_COMPLETIONS in paths(backend))
            assert timeouts.running() == [READ_TIMEOUT]


async def test_a_gateway_that_says_nothing_gets_the_backends_read_timeout_as_ever(caplog):
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            await waiting_for_its_model(gateway, backend)
            assert timeouts.running() == [READ_TIMEOUT]
    assert not any('toptic.ai' in line and 'for a model to load' in line for line in shown(caplog))


@pytest.mark.parametrize('announced', [0, -900, True, 900.0, '900', None, [900]])
async def test_a_wait_that_is_no_positive_whole_number_of_seconds_is_ignored_with_a_warning(caplog, announced):
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, announcing(announced, PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            await waiting_for_its_model(gateway, backend)
            assert timeouts.running() == [READ_TIMEOUT]
    [warning] = [line for line in shown(caplog, logging.WARNING) if 'toptic.ai says it waits' in line]
    assert warning.isprintable() and 'read_timeout' in warning


async def test_a_wait_longer_than_the_client_goes_along_with_counts_as_the_longest_with_a_warning(caplog):
    """Not ignored: a gateway set to wait very long must not leave every machine with its read_timeout."""
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, announcing(BEYOND_THE_LONGEST_SECONDS, PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            await waiting_for_its_model(gateway, backend)
            assert timeouts.running() == [protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS]
    assert shown(caplog, logging.WARNING) == [
        events.loading_wait_shortened(BEYOND_THE_LONGEST_SECONDS, protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS)]


async def test_once_the_answer_has_begun_the_backends_read_timeout_applies_between_its_parts():
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, announcing(ANNOUNCED_SECONDS, PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            backend.hold_stream.clear()
            await waiting_for_its_model(gateway, backend)
            backend.hold_headers.set()
            start = await gateway.next_frame(protocol.FRAME_RESPONSE_START)
            chunk = await gateway.next_frame()
            await timeouts.run_out(READ_TIMEOUT)
            final = await gateway.next_frame()
            backend.hold_stream.set()
    assert start['status'] == HTTP_OK and STREAM_PARTS[0].encode() in protocol.decode_body(chunk['data'])
    assert final == {'type': protocol.FRAME_RESPONSE_ERROR, 'id': REQUEST, 'code': protocol.ERROR_BACKEND_TIMEOUT,
                     'message': f'Backend {OLLAMA} sent nothing more for {READ_TIMEOUT:g} s'}


async def test_a_pause_however_long_is_not_the_backends_silence(gateway):
    """While the gateway holds an answer back, no time of the backend runs."""
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend:
        async with Running(gateway, shared(backend), timeouts):
            backend.hold_stream.clear()
            await gateway.send_request(REQUEST, CHAT)
            await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
            await gateway.send({'type': protocol.FRAME_PAUSE, 'id': REQUEST})
            assert await frames_so_far(gateway, REQUEST) == []  # and the client has taken the pause
            backend.hold_stream.set()
            # the rest of the answer has come, and the client waits for the gateway alone
            await wait_until(lambda: not timeouts.running())
            assert await frames_so_far(gateway, REQUEST) == []
            await gateway.send({'type': protocol.FRAME_RESUME, 'id': REQUEST})
            _, body, final = await gateway.collect_response(REQUEST)
    assert sse_text(body) == ''.join(STREAM_PARTS[1:]) and final['type'] == protocol.FRAME_RESPONSE_END


async def test_a_backend_that_goes_away_in_the_middle_of_its_answer_is_reported_as_unreachable(gateway, caplog):
    async with FakeBackend() as backend:
        backend.dies_mid_stream = True
        async with Running(gateway, shared(backend)):
            await gateway.send_request(REQUEST, CHAT)
            start, body, final = await gateway.collect_response(REQUEST)
            end = await ended(caplog, FIRST)
    assert start['status'] == HTTP_OK and sse_text(body) == STREAM_PARTS[0]
    assert final['type'] == protocol.FRAME_RESPONSE_ERROR and final['code'] == protocol.ERROR_BACKEND_UNREACHABLE
    assert f' {events.OUTCOME_ERROR} after ' in end


async def test_the_longest_wait_itself_is_taken_as_said_without_a_warning(caplog):
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, \
            announcing(protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS, PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            await waiting_for_its_model(gateway, backend)
            assert timeouts.running() == [protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS]
    assert shown(caplog, logging.WARNING) == []
    assert events.gateway_waits_for_loading(protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS) in shown(caplog)


async def reconnected(gateway: FakeGateway, caplog, times: int) -> None:
    """The gateway drops the connection ``times`` times, and the client is connected again after each."""
    for _ in range(times):
        connections = len(gateway.connections)
        await gateway.drop_connection()
        await wait_until(lambda: len(gateway.connections) > connections)
    await wait_until(lambda: len([line for line in shown(caplog) if line.startswith('Connected')]) > times)


async def test_a_wait_shortened_to_the_longest_is_warned_about_once_not_at_every_reconnect(caplog):
    async with FakeBackend() as backend, announcing(BEYOND_THE_LONGEST_SECONDS) as gateway:
        async with Running(gateway, shared(backend)):
            await reconnected(gateway, caplog, RECONNECTS)
    assert shown(caplog, logging.WARNING).count(
        events.loading_wait_shortened(BEYOND_THE_LONGEST_SECONDS, protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS)) == 1


async def test_an_ignored_wait_is_warned_about_once_not_at_every_reconnect(caplog):
    async with FakeBackend() as backend, announcing(NOT_A_WAIT) as gateway:
        async with Running(gateway, shared(backend)):
            await reconnected(gateway, caplog, RECONNECTS)
    assert len([line for line in shown(caplog, logging.WARNING) if line.startswith(IGNORED_WAIT)]) == 1


async def test_a_gateway_whose_wait_is_ignored_after_one_that_said_it_is_not_said_to_say_nothing(caplog):
    """It said something, which the client does not take: the warning says so, and that read_timeout applies."""
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, announcing(ANNOUNCED_SECONDS, PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            gateway.auth_answer[protocol.MODEL_LOADING_TIMEOUT_FIELD] = NOT_A_WAIT
            await reconnected(gateway, caplog, RECONNECTS)
            await waiting_for_its_model(gateway, backend)
            assert timeouts.running() == [READ_TIMEOUT]
    assert events.loading_wait_not_said() not in shown(caplog)
    [warning] = [line for line in shown(caplog, logging.WARNING) if line.startswith(IGNORED_WAIT)]
    assert 'read_timeout' in warning


async def test_a_gateway_that_says_nothing_after_an_ignored_wait_is_not_said_to_have_changed_anything(caplog):
    async with FakeBackend() as backend, announcing(NOT_A_WAIT) as gateway:
        async with Running(gateway, shared(backend)):
            del gateway.auth_answer[protocol.MODEL_LOADING_TIMEOUT_FIELD]
            await reconnected(gateway, caplog, RECONNECTS)
    assert events.loading_wait_not_said() not in shown(caplog)


async def test_how_long_the_gateway_waits_is_shown_once_not_at_every_reconnect(caplog):
    async with FakeBackend() as backend, announcing(ANNOUNCED_SECONDS) as gateway:
        async with Running(gateway, shared(backend)):
            await gateway.drop_connection()
            await wait_until(lambda: len(gateway.connections) > 1)
            await wait_until(lambda: len([line for line in shown(caplog) if line.startswith('Connected')]) > 1)
    assert shown(caplog).count(events.gateway_waits_for_loading(ANNOUNCED_SECONDS)) == 1


async def test_a_reconnect_to_a_gateway_that_says_it_no_more_is_shown_once_too(caplog):
    timeouts = ManualTimeouts()
    async with FakeBackend() as backend, announcing(ANNOUNCED_SECONDS, PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend), timeouts):
            del gateway.auth_answer[protocol.MODEL_LOADING_TIMEOUT_FIELD]
            for connections in (2, 3):
                await gateway.drop_connection()
                await wait_until(lambda: len(gateway.connections) == connections)
            await waiting_for_its_model(gateway, backend)
            assert timeouts.running() == [READ_TIMEOUT]
    assert shown(caplog).count(events.loading_wait_not_said()) == 1


@pytest.mark.parametrize('read_timeout, announced, waited', [
    (READ_TIMEOUT, None, READ_TIMEOUT),
    (READ_TIMEOUT, ANNOUNCED_SECONDS, ANNOUNCED_SECONDS),
    # a backend given more time than the gateway waits keeps it
    (ANNOUNCED_SECONDS + READ_TIMEOUT, ANNOUNCED_SECONDS, ANNOUNCED_SECONDS + READ_TIMEOUT),
])
def test_the_client_waits_the_longer_of_the_backends_time_and_the_gateways(read_timeout, announced, waited):
    backend = Backend(name=OLLAMA, kind=KIND_OLLAMA, base_url='http://backend.invalid/v1', read_timeout=read_timeout)
    assert backend.first_answer_timeout(announced) == waited


@pytest.mark.parametrize('frame, seconds', [
    ({}, None),
    ({protocol.MODEL_LOADING_TIMEOUT_FIELD: ANNOUNCED_SECONDS}, ANNOUNCED_SECONDS),
    # as it says it: the client makes the longest of it that it goes along with
    ({protocol.MODEL_LOADING_TIMEOUT_FIELD: BEYOND_THE_LONGEST_SECONDS}, BEYOND_THE_LONGEST_SECONDS),
])
def test_what_an_auth_ok_frame_says_about_the_wait_for_a_model(frame, seconds):
    assert protocol.model_loading_timeout(frame) == seconds
    assert protocol.MODEL_LOADING_TIMEOUT_FIELD == 'model_loading_timeout_seconds'  # the gateway works to this text


@pytest.mark.parametrize('value', [0, -1, True, False, 1.0, '900', None, [900], {}])
def test_a_wait_that_is_no_positive_whole_number_is_an_error_that_says_why(value):
    with pytest.raises(ValueError, match='second'):
        protocol.model_loading_timeout({protocol.MODEL_LOADING_TIMEOUT_FIELD: value})


def test_the_longest_wait_taken_from_a_gateway_is_an_hour():
    """The gateway checks its setting against the same number."""
    assert protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS == SECONDS_PER_HOUR


# ----- Ollama refuses a prompt while it loads a model -----

@pytest.mark.parametrize('status, body', [
    (HTTP_SERVER_ERROR, LOADING_BODY),
    (HTTP_SERVICE_UNAVAILABLE, LOADING_BODY),
    (HTTP_SERVER_ERROR, LOADING_BODY_OF_OLLAMAS_OWN_ENDPOINTS),
])
async def test_ollama_saying_it_is_loading_a_model_is_an_error_frame_of_its_own_not_an_answer(
        gateway, caplog, status, body):
    async with FakeBackend(chat_failure=(status, body, JSON_CONTENT_TYPE)) as backend:
        async with Running(gateway, shared(backend)):
            await gateway.send_request(REQUEST, CHAT)
            frame = await gateway.next_frame()
            end = await ended(caplog, FIRST)
            assert await frames_so_far(gateway, REQUEST) == []  # neither response.start nor anything of the body
    assert frame == {'type': protocol.FRAME_RESPONSE_ERROR, 'id': REQUEST,
                     'code': protocol.ERROR_BACKEND_MODEL_LOADING,
                     'message': f'Backend {OLLAMA} is still loading a model'}
    assert protocol.ERROR_BACKEND_MODEL_LOADING == 'backend_model_loading'  # the gateway works to this text
    assert events.backend_loading_model(FIRST, OLLAMA) in shown(caplog)
    assert shown(caplog, logging.WARNING) == [end] and protocol.ERROR_BACKEND_MODEL_LOADING in end
    assert GENERATE not in paths(backend)


async def test_the_loading_answer_is_recognised_when_it_comes_in_several_pieces(gateway):
    failure = (HTTP_SERVER_ERROR, in_parts(LOADING_BODY), JSON_CONTENT_TYPE)
    timeouts = ManualTimeouts()
    async with FakeBackend(chat_failure=failure) as backend:
        async with Running(gateway, shared(backend), timeouts):
            backend.hold_failure_body.clear()
            await gateway.send_request(REQUEST, CHAT)
            for sent in range(1, len(failure[1])):
                # one piece at a time: the next is sent once the client is reading the answer ahead
                await wait_until(lambda: backend.failure_parts_sent == sent)
                await wait_until(lambda: timeouts.running() == [OLLAMA_LOADING_ANSWER_MAX_SECONDS])
                backend.hold_failure_body.set()
                backend.hold_failure_body.clear()
                await wait_until(lambda: backend.failure_parts_sent == sent + 1)
            frame = await gateway.next_frame()
    assert frame['type'] == protocol.FRAME_RESPONSE_ERROR and frame['code'] == protocol.ERROR_BACKEND_MODEL_LOADING


async def test_nothing_of_the_progress_follows_the_error_frame():
    failure = (HTTP_SERVER_ERROR, LOADING_BODY, JSON_CONTENT_TYPE)
    async with FakeBackend(chat_failure=failure) as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)):
            await waiting_for_its_model(gateway, backend)
            backend.hold_headers.set()
            assert (await gateway.next_frame())['type'] == protocol.FRAME_RESPONSE_ERROR
            assert await frames_so_far(gateway, REQUEST) == []


async def test_an_error_answer_that_stalls_is_passed_on_as_far_as_it_came_not_held_back(gateway):
    """Also one that would have turned out to be the loading answer: what the client waits for is short."""
    timeouts = ManualTimeouts()
    async with FakeBackend(chat_failure=(HTTP_SERVER_ERROR, in_parts(LOADING_BODY), JSON_CONTENT_TYPE)) as backend:
        async with Running(gateway, shared(backend), timeouts):
            backend.hold_failure_body.clear()
            await gateway.send_request(REQUEST, CHAT)
            await wait_until(lambda: backend.failure_parts_sent == 1)
            await timeouts.run_out(OLLAMA_LOADING_ANSWER_MAX_SECONDS)
            start = await gateway.next_frame()
            chunk = await gateway.next_frame()
            # the rest has not even been sent by the backend
            assert backend.failure_parts_sent == 1 and await frames_so_far(gateway, REQUEST) == []
            backend.hold_failure_body.set()
            _, more, final = await gateway.collect_response(REQUEST)
    assert start == {'type': protocol.FRAME_RESPONSE_START, 'id': REQUEST, 'status': HTTP_SERVER_ERROR,
                     'headers': {'content-type': JSON_CONTENT_TYPE}}
    assert chunk['type'] == protocol.FRAME_RESPONSE_CHUNK
    assert protocol.decode_body(chunk['data']) + more == LOADING_BODY  # byte for byte, nothing read twice or lost
    assert final == {'type': protocol.FRAME_RESPONSE_END, 'id': REQUEST}


async def test_a_request_cancelled_while_its_error_answer_is_read_ends_with_nothing_sent(caplog):
    failure = (HTTP_SERVER_ERROR, in_parts(LOADING_BODY), JSON_CONTENT_TYPE)
    async with FakeBackend(chat_failure=failure) as backend, FakeGateway(capabilities=PROGRESS_CAPABLE) as gateway:
        async with Running(gateway, shared(backend)) as running:
            backend.hold_failure_body.clear()
            await waiting_for_its_model(gateway, backend)
            backend.hold_headers.set()
            await wait_until(lambda: backend.failure_parts_sent == 1)
            await cancel(gateway)
            end = await ended(caplog, FIRST)
            assert await frames_so_far(gateway, REQUEST) == []
            # Ollama had begun to answer: there is no load of it that the cancel could have ended
            assert not running.loads_on_their_way()
    assert f' {events.OUTCOME_CANCELLED} after ' in end and loads(backend) == []
    assert not [record for record in caplog.records if record.levelno > logging.WARNING]


async def test_a_pause_holds_back_what_was_read_ahead_of_an_error_answer_too(gateway):
    failure = (HTTP_SERVER_ERROR, in_parts(OTHER_FAILURE_BODY), JSON_CONTENT_TYPE)
    async with FakeBackend(chat_failure=failure) as backend:
        async with Running(gateway, shared(backend)):
            backend.hold_failure_body.clear()
            await gateway.send_request(REQUEST, CHAT)
            await wait_until(lambda: backend.failure_parts_sent == 1)
            await gateway.send({'type': protocol.FRAME_PAUSE, 'id': REQUEST})
            backend.hold_failure_body.set()
            start = await gateway.next_frame()
            assert start['type'] == protocol.FRAME_RESPONSE_START
            assert await frames_so_far(gateway, REQUEST) == []  # no chunk while paused
            await gateway.send({'type': protocol.FRAME_RESUME, 'id': REQUEST})
            _, body, final = await gateway.collect_response(REQUEST)
    assert body == OTHER_FAILURE_BODY and final['type'] == protocol.FRAME_RESPONSE_END


@pytest.mark.parametrize('status, body, content_type', [
    (HTTP_SERVER_ERROR, OTHER_FAILURE_BODY, JSON_CONTENT_TYPE),
    (HTTP_SERVER_ERROR, in_parts(OTHER_FAILURE_BODY), JSON_CONTENT_TYPE),
    (HTTP_SERVER_ERROR, OVERSIZED_LOADING_BODY, JSON_CONTENT_TYPE),
    (HTTP_SERVER_ERROR, NOT_JSON_LOADING_BODY, TEXT_CONTENT_TYPE),
    (HTTP_SERVER_ERROR, b'', TEXT_CONTENT_TYPE),
    (HTTP_CONFLICT, LOADING_BODY, JSON_CONTENT_TYPE),
])
async def test_any_other_error_answer_of_ollama_is_relayed_as_it_came(gateway, status, body, content_type):
    async with FakeBackend(chat_failure=(status, body, content_type)) as backend:
        async with Running(gateway, shared(backend)):
            await gateway.send_request(REQUEST, CHAT)
            start, relayed, final = await gateway.collect_response(REQUEST)
    assert start == {'type': protocol.FRAME_RESPONSE_START, 'id': REQUEST, 'status': status,
                     'headers': {'content-type': content_type}}
    assert relayed == (body if isinstance(body, bytes) else b''.join(body))
    assert final == {'type': protocol.FRAME_RESPONSE_END, 'id': REQUEST}


async def test_the_same_words_of_a_backend_of_another_kind_are_relayed_as_they_came(gateway):
    async with FakeBackend(chat_failure=(HTTP_SERVER_ERROR, LOADING_BODY, JSON_CONTENT_TYPE)) as backend:
        async with Running(gateway, litellm_backend(backend, models=[LITELLM_MODEL])):
            await gateway.send_request(REQUEST, {**CHAT, 'model': LITELLM_MODEL}, backend=LITELLM)
            start, relayed, final = await gateway.collect_response(REQUEST)
    assert start['status'] == HTTP_SERVER_ERROR and start['headers'] == {'content-type': JSON_CONTENT_TYPE}
    assert relayed == LOADING_BODY and final['type'] == protocol.FRAME_RESPONSE_END


async def test_an_error_answer_that_was_read_ahead_still_ends_its_line_with_the_backends_error(gateway, caplog):
    """What was read to decide is read along like the rest of an answer."""
    async with FakeBackend(chat_failure=(HTTP_SERVER_ERROR, OTHER_FAILURE_BODY, JSON_CONTENT_TYPE)) as backend:
        async with Running(gateway, shared(backend)):
            await gateway.send_request(REQUEST, CHAT)
            await gateway.collect_response(REQUEST)
            end = await ended(caplog, FIRST)
    assert end.endswith(f': HTTP {HTTP_SERVER_ERROR} from {OLLAMA} (api_error)')


async def test_an_error_that_is_no_server_error_is_not_read_ahead(gateway, backend, running_client):
    await gateway.send_request(REQUEST, {**CHAT, 'model': 'missing'})
    start, relayed, _ = await gateway.collect_response(REQUEST)
    assert start['status'] == HTTP_NOT_FOUND and b'model not found' in relayed


@pytest.mark.parametrize('body, loading', [
    (LOADING_BODY, True),
    (LOADING_BODY_OF_OLLAMAS_OWN_ENDPOINTS, True),
    (OTHER_FAILURE_BODY, False),
    (NOT_JSON_LOADING_BODY, False),
    (b'', False),
    (b'\xff\xfe', False),
    (json.dumps(LOADING_MESSAGE).encode(), False),
    (json.dumps([LOADING_MESSAGE]).encode(), False),
    (json.dumps({'error': {'message': None, 'detail': LOADING_MESSAGE}}).encode(), False),
    (json.dumps({'message': LOADING_MESSAGE}).encode(), False),
])
def test_what_says_that_a_model_is_loading(body, loading):
    assert says_model_is_loading(body) is loading


def test_the_words_looked_for_are_part_of_ollamas_message():
    assert ollama_memory.OLLAMA_LOADING_MODEL_TEXT in LOADING_MESSAGE
