import asyncio

import pytest
import pytest_asyncio

from socketllama import cli
from socketllama.client import TunnelClient
from socketllama.config import parse_config

from .fakes import BACKEND_API_KEY, INSTANCE, SIGNING_SECRET, TOKEN, FakeBackend, FakeGateway

FAST_PING_INTERVAL_SECONDS = 0.05
FAST_PONG_TIMEOUT_SECONDS = 0.2
FAST_RECONNECT_SECONDS = 0.01
STARTUP_TIMEOUT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.01
# models = [] in a backend's configuration: every model it offers is shared
SHARE_ALL_MODELS = []


def make_config(gateway_url: str, backends: list, **client_options):
    return parse_config({
        'gateway': {'url': gateway_url, 'instance': INSTANCE, 'token': TOKEN, 'signing_secret': SIGNING_SECRET,
                    'allow_insecure': True},
        'client': {'reconnect_min': FAST_RECONNECT_SECONDS, 'reconnect_max': FAST_RECONNECT_SECONDS,
                   **client_options},
        'backends': backends,
    }, environ={})


def ollama_backend(backend: FakeBackend, **overrides) -> dict:
    return {'name': 'ollama', 'kind': 'ollama', 'base_url': backend.base_url, 'models': SHARE_ALL_MODELS,
            **overrides}


def litellm_backend(backend: FakeBackend, **overrides) -> dict:
    return {'name': 'litellm', 'kind': 'litellm', 'base_url': backend.base_url, 'api_key': BACKEND_API_KEY,
            'models': SHARE_ALL_MODELS, **overrides}


async def wait_until(condition, timeout=STARTUP_TIMEOUT_SECONDS):
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(POLL_INTERVAL_SECONDS)


@pytest.fixture(autouse=True)
def never_the_real_client(monkeypatch):
    """No test starts the real client through the command line, e.g. a setup script that gets out of
    step with the questions and ends in "Start it now? y": the client would try to reach the gateway
    named in the test. Tests that start it use no_real_run, or a TunnelClient against the fakes."""
    async def refuse(config):
        raise AssertionError('a test started the real client')
    monkeypatch.setattr(cli, 'run', refuse)


@pytest_asyncio.fixture
async def backend():
    async with FakeBackend() as fake:
        yield fake


@pytest_asyncio.fixture
async def gateway():
    async with FakeGateway() as fake:
        yield fake


@pytest_asyncio.fixture
async def running_client(gateway, backend):
    """A client connected to the fake gateway with one Ollama backend; stopped after the test."""
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = asyncio.create_task(client.run_forever())
    await wait_until(lambda: gateway.connections)
    yield client
    client.stop()
    await asyncio.wait_for(task, STARTUP_TIMEOUT_SECONDS)


@pytest.fixture
def fast_keepalive():
    return {'ping_interval': FAST_PING_INTERVAL_SECONDS, 'pong_timeout': FAST_PONG_TIMEOUT_SECONDS}
