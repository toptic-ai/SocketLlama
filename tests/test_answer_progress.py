"""What the client reports before the backend's answer starts (response.progress), and only to a
gateway that announced it can take it."""
import asyncio

import pytest
import pytest_asyncio

from socketllama import protocol
from socketllama.backends import ollama_model_reference
from socketllama.client import TunnelClient

from .conftest import STARTUP_TIMEOUT_SECONDS, litellm_backend, make_config, ollama_backend, wait_until
from .fakes import CONTEXT_LENGTH, HTTP_NOT_FOUND, MODEL, FakeBackend, FakeGateway

CHAT = {'model': MODEL, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}
PROGRESS_CAPABLE = [protocol.CAPABILITY_RESPONSE_PROGRESS, protocol.CAPABILITY_MODELS_UNCHECKED]
QUIET_SECONDS = 0.2


@pytest_asyncio.fixture
async def progress_gateway():
    async with FakeGateway(capabilities=PROGRESS_CAPABLE) as fake:
        yield fake


async def _connected(gateway, backend_config) -> tuple:
    client = TunnelClient(make_config(gateway.url, [backend_config]))
    task = asyncio.create_task(client.run_forever())
    await wait_until(lambda: gateway.connections)
    return client, task


async def _finish(client, task):
    client.stop()
    await asyncio.wait_for(task, STARTUP_TIMEOUT_SECONDS)


async def _stages_before_start(gateway, request_id: str) -> list:
    """Stages of the progress frames of a request, up to its response.start."""
    stages = []
    while True:
        frame = await gateway.next_frame()
        if frame.get('id') != request_id:
            continue
        if frame['type'] != protocol.FRAME_RESPONSE_PROGRESS:
            assert frame['type'] == protocol.FRAME_RESPONSE_START
            return stages
        stages.append(frame['stage'])


async def _stages_while_loading(gateway, backend, backend_config) -> list:
    """Stages reported while the backend holds back its answer, as long as a model takes to load."""
    backend.hold_headers.clear()
    client, task = await _connected(gateway, backend_config)
    await gateway.send_request('req-1', CHAT, backend=backend_config['name'])
    await asyncio.sleep(QUIET_SECONDS)
    backend.hold_headers.set()
    stages = await _stages_before_start(gateway, 'req-1')
    await _finish(client, task)
    return stages


async def test_a_model_not_in_memory_is_reported_after_the_request_was_received(progress_gateway):
    async with FakeBackend(loaded_context_length=None) as backend:
        stages = await _stages_while_loading(progress_gateway, backend, ollama_backend(backend))
    assert stages == [protocol.PROGRESS_STAGE_RECEIVED, protocol.PROGRESS_STAGE_MODEL_NOT_IN_MEMORY]
    assert sum(request['path'] == '/api/ps' for request in backend.requests) == 2  # discovery + this request


async def test_a_model_in_memory_is_only_received(progress_gateway):
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH) as backend:
        stages = await _stages_while_loading(progress_gateway, backend, ollama_backend(backend))
    assert stages == [protocol.PROGRESS_STAGE_RECEIVED]


async def test_the_implied_latest_tag_counts_as_the_same_model(progress_gateway):
    async with FakeBackend(loaded_context_length=CONTEXT_LENGTH, loaded_name='llama3.1:latest') as backend:
        backend.hold_headers.clear()
        client, task = await _connected(progress_gateway, ollama_backend(backend))
        await progress_gateway.send_request('req-1', {**CHAT, 'model': 'llama3.1'})
        await asyncio.sleep(QUIET_SECONDS)
        backend.hold_headers.set()
        stages = await _stages_before_start(progress_gateway, 'req-1')
        await _finish(client, task)
    assert stages == [protocol.PROGRESS_STAGE_RECEIVED]


async def test_an_unknown_list_of_loaded_models_claims_nothing(progress_gateway):
    async with FakeBackend(process_list_status=HTTP_NOT_FOUND) as backend:
        stages = await _stages_while_loading(progress_gateway, backend, ollama_backend(backend))
    assert stages == [protocol.PROGRESS_STAGE_RECEIVED]


async def test_other_backends_are_not_asked_about_loaded_models(progress_gateway):
    async with FakeBackend() as backend:
        stages = await _stages_while_loading(progress_gateway, backend, litellm_backend(backend))
    assert stages == [protocol.PROGRESS_STAGE_RECEIVED]
    assert not any(request['path'] == '/api/ps' for request in backend.requests)


async def test_each_stage_is_sent_once_however_long_the_backend_takes(progress_gateway):
    """Late joiners learn the stage from the gateway, which publishes it again on its own."""
    async with FakeBackend(loaded_context_length=None) as backend:
        stages = await _stages_while_loading(progress_gateway, backend, ollama_backend(backend))
    assert stages == [protocol.PROGRESS_STAGE_RECEIVED, protocol.PROGRESS_STAGE_MODEL_NOT_IN_MEMORY]


async def test_no_progress_after_the_answer_started(progress_gateway):
    async with FakeBackend() as backend:
        backend.hold_stream.clear()
        client, task = await _connected(progress_gateway, ollama_backend(backend))
        await progress_gateway.send_request('req-1', CHAT)
        await _stages_before_start(progress_gateway, 'req-1')
        await asyncio.sleep(QUIET_SECONDS)
        frames = progress_gateway.drain_frames_of('req-1')
        backend.hold_stream.set()
        await _finish(client, task)
    assert all(frame['type'] != protocol.FRAME_RESPONSE_PROGRESS for frame in frames)


async def test_a_gateway_without_the_capability_gets_no_progress(gateway):
    async with FakeBackend(loaded_context_length=None) as backend:
        backend.hold_headers.clear()
        client, task = await _connected(gateway, ollama_backend(backend))
        await gateway.send_request('req-1', CHAT)
        await asyncio.sleep(QUIET_SECONDS)
        backend.hold_headers.set()
        assert await _stages_before_start(gateway, 'req-1') == []
        await _finish(client, task)
    assert sum(request['path'] == '/api/ps' for request in backend.requests) == 1  # discovery only


@pytest.mark.parametrize('name, reference', [
    ('llama3.1', 'llama3.1:latest'),
    ('llama3.1:8b', 'llama3.1:8b'),
    ('hf.co/org/model', 'hf.co/org/model:latest'),
    ('registry.local:5000/org/model', 'registry.local:5000/org/model:latest'),
    ('registry.local:5000/org/model:q4', 'registry.local:5000/org/model:q4'),
])
def test_ollama_model_reference(name, reference):
    assert ollama_model_reference(name) == reference
