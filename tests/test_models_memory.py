"""What the client tells the gateway about the models Ollama holds in memory (models.memory), and
only a gateway that announced it can take it."""
import asyncio
import gc
import logging
import time

import pytest
import pytest_asyncio

from socketllama import backends as backends_module
from socketllama import models_memory, protocol
from socketllama.client import TunnelClient

from .conftest import STARTUP_TIMEOUT_SECONDS, litellm_backend, make_config, ollama_backend, wait_until
from .fakes import CONTEXT_LENGTH, EMBEDDING_MODEL, HTTP_NOT_FOUND, MODEL, FakeBackend, FakeGateway

MEMORY_CAPABLE = [protocol.CAPABILITY_MODELS_UNCHECKED, protocol.CAPABILITY_MODELS_MEMORY]
CHAT = {'model': MODEL, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}
SECOND_MODEL = 'qwen3:14b'
# A model /v1/models names without a tag; /api/ps names it with the tag Ollama implies
UNTAGGED_MODEL = 'mistral'
UNTAGGED_MODEL_IN_MEMORY = 'mistral:latest'
PROCESS_LIST = '/api/ps'
CHAT_COMPLETIONS = '/v1/chat/completions'
CLIENT_LOGGER = 'socketllama'
QUIET_SECONDS = 0.2
# Shortened intervals: checks without a reason, and the least time between two frames
FAST_POLL_SECONDS = 0.05
SHORT_MIN_INTERVAL_SECONDS = 0.1
# Long enough for two answers to end within it
COALESCING_MIN_INTERVAL_SECONDS = 1.5
# Polls watched for a frame that should not come
POLLS_WATCHED = 5
# A backend's discovery that gives up at once, so it is reported as not checked
FAST_DISCOVERY_TIMEOUT_SECONDS = 0.1
MEMORY_TASK = 'ModelsMemoryReport.run'


def memory_frames(gateway: FakeGateway) -> list:
    return [frame for frame in gateway.drain_frames() if frame['type'] == protocol.FRAME_MODELS_MEMORY]


def process_list_reads(backend: FakeBackend) -> int:
    return sum(request['path'] == PROCESS_LIST for request in backend.requests)


def running_memory_tasks() -> int:
    return sum(task.get_coro().__qualname__ == MEMORY_TASK for task in asyncio.all_tasks() if not task.done())


class Running:
    """A client connected to a fake gateway with the backends given, stopped at the end."""

    def __init__(self, gateway: FakeGateway, *backend_configs: dict):
        self.client = TunnelClient(make_config(gateway.url, list(backend_configs)))
        self._gateway = gateway

    async def __aenter__(self):
        self.task = asyncio.create_task(self.client.run_forever())
        await wait_until(lambda: self._gateway.connections)
        return self

    async def __aexit__(self, *exc_info):
        self.client.stop()
        await asyncio.wait_for(self.task, STARTUP_TIMEOUT_SECONDS)


@pytest_asyncio.fixture
async def memory_gateway():
    async with FakeGateway(capabilities=MEMORY_CAPABLE) as fake:
        yield fake


@pytest.fixture
def short_interval(monkeypatch):
    monkeypatch.setattr(models_memory, 'MODELS_MEMORY_MIN_INTERVAL_SECONDS', SHORT_MIN_INTERVAL_SECONDS)


@pytest.fixture
def fast_poll(monkeypatch):
    monkeypatch.setattr(models_memory, 'MODELS_MEMORY_POLL_SECONDS', FAST_POLL_SECONDS)


@pytest.fixture
def memory_sent_at(monkeypatch) -> list[float]:
    """When the client handed each models.memory frame to its socket, on the clock it times the least
    interval with (time.monotonic). The interval runs from one send to the next: a moment the test takes
    once it has received a frame lies after the send, by however long the frame took to get there."""
    sent_at: list[float] = []
    send = TunnelClient._send

    async def recording_send(socket, frame: dict) -> None:
        if frame['type'] == protocol.FRAME_MODELS_MEMORY:
            sent_at.append(time.monotonic())
        await send(socket, frame)
    monkeypatch.setattr(TunnelClient, '_send', staticmethod(recording_send))
    return sent_at


# ----- the capability -----

@pytest.mark.parametrize('capabilities', [None, [protocol.CAPABILITY_MODELS_UNCHECKED]])
async def test_a_gateway_without_the_capability_gets_no_such_frame(capabilities, fast_poll):
    async with FakeBackend(loads_on_chat=True) as backend, FakeGateway(capabilities=capabilities) as gateway:
        async with Running(gateway, ollama_backend(backend)):
            await gateway.next_frame(protocol.FRAME_MODELS)
            await gateway.send_request('req-1', CHAT)
            await gateway.collect_response('req-1')
            await asyncio.sleep(QUIET_SECONDS)
            assert memory_frames(gateway) == []
    assert process_list_reads(backend) == 1  # discovery only


# ----- what is sent, and when -----

async def test_it_follows_the_first_models_report_of_a_connection(memory_gateway):
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH) as backend:
        async with Running(memory_gateway, ollama_backend(backend)):
            first = await memory_gateway.next_frame()
            memory = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert first['type'] == protocol.FRAME_MODELS
    assert memory == {'type': protocol.FRAME_MODELS_MEMORY, 'backends': [{'backend': 'ollama', 'models': [MODEL]}]}


async def test_a_backend_with_nothing_in_memory_is_listed_without_models(memory_gateway):
    async with FakeBackend() as backend:
        async with Running(memory_gateway, ollama_backend(backend)):
            memory = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert memory['backends'] == [{'backend': 'ollama', 'models': []}]


async def test_an_unchanged_state_is_not_sent_again(memory_gateway, fast_poll, short_interval):
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH) as backend:
        async with Running(memory_gateway, ollama_backend(backend)):
            await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            reads = process_list_reads(backend)
            await wait_until(lambda: process_list_reads(backend) >= reads + POLLS_WATCHED)
            assert memory_frames(memory_gateway) == []


async def test_a_change_is_sent(memory_gateway, fast_poll, short_interval):
    async with FakeBackend(more_models=[SECOND_MODEL]) as backend:
        async with Running(memory_gateway, ollama_backend(backend)):
            assert (await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY))['backends'][0]['models'] == []
            backend.in_memory[SECOND_MODEL] = None  # loaded by somebody else, e.g. the ollama command
            changed = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert changed['backends'] == [{'backend': 'ollama', 'models': [SECOND_MODEL]}]


async def test_after_an_answer_it_is_checked_without_waiting_for_the_next_poll(memory_gateway, short_interval):
    async with FakeBackend(loads_on_chat=True) as backend:
        async with Running(memory_gateway, ollama_backend(backend)):
            await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            await memory_gateway.send_request('req-1', CHAT)
            _, _, final = await memory_gateway.collect_response('req-1')
            assert final['type'] == protocol.FRAME_RESPONSE_END
            # well before MODELS_MEMORY_POLL_SECONDS
            changed = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert changed['backends'] == [{'backend': 'ollama', 'models': [MODEL]}]
    paths = [request['path'] for request in backend.requests]
    last_check = max(index for index, path in enumerate(paths) if path == PROCESS_LIST)
    assert last_check > paths.index(CHAT_COMPLETIONS)


async def test_once_the_models_are_loaded_at_start_it_is_checked(memory_gateway, short_interval):
    async with FakeBackend() as backend:
        backend.hold_generate.clear()
        async with Running(memory_gateway, ollama_backend(backend, models=[MODEL], load_at_start=True)):
            first = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            backend.hold_generate.set()
            # well before MODELS_MEMORY_POLL_SECONDS, and before the fresh models report the loading asks for
            loaded = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert first['backends'] == [{'backend': 'ollama', 'models': []}]
    assert loaded['backends'] == [{'backend': 'ollama', 'models': [MODEL]}]


async def test_changes_within_the_least_interval_end_in_one_frame_with_the_latest_state(memory_gateway, monkeypatch,
                                                                                        memory_sent_at):
    monkeypatch.setattr(models_memory, 'MODELS_MEMORY_MIN_INTERVAL_SECONDS', COALESCING_MIN_INTERVAL_SECONDS)
    async with FakeBackend(loads_on_chat=True, more_models=[SECOND_MODEL]) as backend:
        async with Running(memory_gateway, ollama_backend(backend)):
            await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            for request_id, model in (('req-1', MODEL), ('req-2', SECOND_MODEL)):
                await memory_gateway.send_request(request_id, {**CHAT, 'model': model})
                await memory_gateway.collect_response(request_id)
            # both changes came within it
            assert time.monotonic() - memory_sent_at[0] < COALESCING_MIN_INTERVAL_SECONDS
            latest = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            await asyncio.sleep(QUIET_SECONDS)
            assert memory_frames(memory_gateway) == []
    assert latest['backends'] == [{'backend': 'ollama', 'models': [MODEL, SECOND_MODEL]}]
    # not sent before the least interval had passed since the frame before
    first, second = memory_sent_at
    assert second - first >= COALESCING_MIN_INTERVAL_SECONDS


async def test_after_a_reconnect_it_is_sent_again(memory_gateway, short_interval):
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH) as backend:
        async with Running(memory_gateway, ollama_backend(backend)):
            before = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            await memory_gateway.drop_connection()
            await wait_until(lambda: len(memory_gateway.connections) == 2)
            assert (await memory_gateway.next_frame())['type'] == protocol.FRAME_MODELS
            after = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert after == before


async def test_the_least_interval_spans_a_reconnect(memory_gateway, monkeypatch, memory_sent_at):
    """Timed from the last frame sent, on whichever connection."""
    monkeypatch.setattr(models_memory, 'MODELS_MEMORY_MIN_INTERVAL_SECONDS', COALESCING_MIN_INTERVAL_SECONDS)
    async with FakeBackend() as backend:
        async with Running(memory_gateway, ollama_backend(backend)):
            await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            await memory_gateway.drop_connection()
            await wait_until(lambda: len(memory_gateway.connections) == 2)
            await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    first, second = memory_sent_at
    assert second - first >= COALESCING_MIN_INTERVAL_SECONDS


# ----- what is in it -----

async def test_models_are_named_as_the_models_report_names_them(memory_gateway):
    async with FakeBackend(more_models=[UNTAGGED_MODEL]) as backend:
        backend.in_memory[UNTAGGED_MODEL_IN_MEMORY] = None
        async with Running(memory_gateway, ollama_backend(backend)):
            memory = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert memory['backends'] == [{'backend': 'ollama', 'models': [UNTAGGED_MODEL]}]


async def test_models_that_are_not_reported_are_not_told(memory_gateway):
    """Not shared, or no chat model: the gateway does not know them, and nobody is to learn of them."""
    async with FakeBackend(more_models=[SECOND_MODEL]) as backend:
        backend.in_memory.update({SECOND_MODEL: None, EMBEDDING_MODEL: None, MODEL: None})
        async with Running(memory_gateway, ollama_backend(backend, models=[MODEL, EMBEDDING_MODEL])):
            memory = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert memory['backends'] == [{'backend': 'ollama', 'models': [MODEL]}]


async def test_a_backend_of_another_kind_is_left_out(memory_gateway):
    async with FakeBackend() as ollama, FakeBackend() as litellm:
        async with Running(memory_gateway, ollama_backend(ollama), litellm_backend(litellm)):
            memory = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert [backend['backend'] for backend in memory['backends']] == ['ollama']
    assert process_list_reads(litellm) == 0


async def test_a_backend_whose_memory_cannot_be_read_is_left_out_and_logged_once(memory_gateway, fast_poll,
                                                                              short_interval, caplog):
    async with FakeBackend() as working, FakeBackend(process_list_status=HTTP_NOT_FOUND) as broken:
        with caplog.at_level(logging.WARNING, logger=CLIENT_LOGGER):
            async with Running(memory_gateway, ollama_backend(working), ollama_backend(broken, name='broken')):
                first = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
                reads = process_list_reads(broken)
                await wait_until(lambda: process_list_reads(broken) >= reads + POLLS_WATCHED)
                working.in_memory[MODEL] = None
                changed = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)  # it keeps going
    assert first['backends'] == [{'backend': 'ollama', 'models': []}]
    assert changed['backends'] == [{'backend': 'ollama', 'models': [MODEL]}]
    assert len([record for record in caplog.records
                if record.getMessage().startswith('Backend broken:') and PROCESS_LIST in record.getMessage()]) == 1


async def test_a_backend_reported_as_not_checked_is_left_out(memory_gateway, monkeypatch):
    """Its models are not known to the client: the gateway keeps the ones it had for a while."""
    monkeypatch.setattr(backends_module, 'DISCOVERY_TIMEOUT_SECONDS', FAST_DISCOVERY_TIMEOUT_SECONDS)
    async with FakeBackend() as working, FakeBackend() as busy:
        busy.hold_discovery.clear()
        async with Running(memory_gateway, ollama_backend(working), ollama_backend(busy, name='busy')):
            report = await memory_gateway.next_frame(protocol.FRAME_MODELS)
            memory = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
    assert report['backends'][1][protocol.BACKEND_CHECKED_FIELD] is False
    assert memory['backends'] == [{'backend': 'ollama', 'models': []}]


# ----- the gateway's answers, and the end of a connection -----

async def test_an_error_frame_does_not_end_anything_and_the_state_is_sent_again_once(memory_gateway, fast_poll,
                                                                                    short_interval):
    """The gateway may answer a frame it ignored with an error that does not say which frame it was about."""
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH) as backend:
        async with Running(memory_gateway, ollama_backend(backend)) as running:
            before = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            await memory_gateway.send({'type': protocol.FRAME_ERROR, 'message': 'Frame ignored'})
            again = await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
            reads = process_list_reads(backend)
            await wait_until(lambda: process_list_reads(backend) >= reads + POLLS_WATCHED)
            assert memory_frames(memory_gateway) == []
            assert running.client.connections_established == 1
    assert again == before


async def test_a_disconnect_ends_its_task_cleanly(memory_gateway, caplog):
    """Also while a change waits for the least interval to pass."""
    async with FakeBackend(loads_on_chat=True) as backend:
        with caplog.at_level(logging.DEBUG):
            async with Running(memory_gateway, ollama_backend(backend)):
                await memory_gateway.next_frame(protocol.FRAME_MODELS_MEMORY)
                await memory_gateway.send_request('req-1', CHAT)
                await memory_gateway.collect_response('req-1')
                await wait_until(lambda: process_list_reads(backend) >= 2)  # the change is noticed, and waits
                await memory_gateway.drop_connection()
                await wait_until(lambda: len(memory_gateway.connections) == 2)
                await wait_until(lambda: running_memory_tasks() == 1)  # the new connection's only
            assert running_memory_tasks() == 0
            gc.collect()
    assert 'never retrieved' not in caplog.text
    assert 'Connection helper' not in caplog.text
