"""What setup lists for a backend: the models it could share, or why they could not be listed."""
import httpx
import pytest

from socketllama import protocol
from socketllama.backends import DISCOVERY_TIMEOUT_SECONDS, BackendUnavailable, shareable_model_names
from socketllama.config import KIND_LITELLM, KIND_OLLAMA, KIND_OPENAI, Backend

from .fakes import BACKEND_API_KEY, EMBEDDING_MODEL, HTTP_OK, MODEL

HTTP_SERVER_ERROR = 500
# Never contacted: every request of these tests is answered by a mock transport
UNUSED_URL = 'http://backend.invalid/v1'
EVERY_MODEL = Backend(name='ollama', kind=KIND_OLLAMA, base_url=UNUSED_URL)


def _http(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _failing_http(error: Exception) -> httpx.AsyncClient:
    def handler(request):
        raise error
    return _http(handler)


def _connect_error_caused_by(cause: Exception) -> httpx.ConnectError:
    try:
        raise httpx.ConnectError('All connection attempts failed') from cause
    except httpx.ConnectError as error:
        return error


async def test_ollama_lists_its_chat_models(backend):
    async with httpx.AsyncClient() as http:
        names = await shareable_model_names(Backend(name='ollama', kind=KIND_OLLAMA, base_url=backend.base_url), http)
    assert names == [MODEL]  # the embedding model cannot chat


async def test_an_openai_compatible_server_lists_what_it_names(backend):
    async with httpx.AsyncClient() as http:
        names = await shareable_model_names(Backend(name='vllm', kind=KIND_OPENAI, base_url=backend.base_url), http)
    assert names == [MODEL, EMBEDDING_MODEL]


async def test_a_litellm_proxy_is_asked_with_its_key(backend):
    proxy = Backend(name='litellm', kind=KIND_LITELLM, base_url=backend.base_url, api_key=BACKEND_API_KEY)
    async with httpx.AsyncClient() as http:
        assert await shareable_model_names(proxy, http) == ['gpt-proxy']
    assert backend.requests[0]['headers']['authorization'] == f'Bearer {BACKEND_API_KEY}'


async def test_a_refused_connection_says_the_backend_is_down():
    async with _failing_http(_connect_error_caused_by(ConnectionRefusedError())) as http:
        with pytest.raises(BackendUnavailable) as raised:
            await shareable_model_names(EVERY_MODEL, http)
    assert raised.value.is_down and not raised.value.timed_out


async def test_a_timeout_says_after_how_long():
    async with _failing_http(httpx.ReadTimeout('')) as http:
        with pytest.raises(BackendUnavailable, match=f'{DISCOVERY_TIMEOUT_SECONDS:g} s') as raised:
            await shareable_model_names(EVERY_MODEL, http)
    assert raised.value.timed_out and not raised.value.is_down


async def test_an_unusable_answer_carries_the_reason():
    def handler(request):
        return httpx.Response(HTTP_SERVER_ERROR, json={'error': 'busy'})
    async with _http(handler) as http:
        with pytest.raises(BackendUnavailable, match=str(HTTP_SERVER_ERROR)) as raised:
            await shareable_model_names(EVERY_MODEL, http)
    assert raised.value.answered and not raised.value.is_down and not raised.value.timed_out


async def test_a_host_that_cannot_be_reached_did_not_answer():
    async with _failing_http(_connect_error_caused_by(OSError('No route to host'))) as http:
        with pytest.raises(BackendUnavailable) as raised:
            await shareable_model_names(EVERY_MODEL, http)
    assert not raised.value.answered and not raised.value.is_down and not raised.value.timed_out


async def test_an_unusable_answer_is_an_answer():
    def handler(request):
        return httpx.Response(HTTP_OK, text='<html>not the API</html>')
    async with _http(handler) as http:
        with pytest.raises(BackendUnavailable) as raised:
            await shareable_model_names(EVERY_MODEL, http)
    assert raised.value.answered


async def test_an_address_the_http_library_refuses_is_unavailable_not_a_crash():
    async with httpx.AsyncClient() as http:  # refused before anything is sent
        with pytest.raises(BackendUnavailable, match='Invalid') as raised:
            await shareable_model_names(Backend(name='ollama', kind=KIND_OLLAMA, base_url='http://[::1/v1'), http)
    assert not raised.value.answered


async def test_names_the_gateway_would_refuse_are_not_offered(caplog):
    too_long = 'n' * (protocol.MODEL_NAME_MAX_LENGTH + 1)

    def handler(request):
        return httpx.Response(HTTP_OK, json={'data': [{'id': 'fine'}, {'id': too_long}, {'id': 'tab\there'}]})
    async with _http(handler) as http:
        names = await shareable_model_names(Backend(name='vllm', kind=KIND_OPENAI, base_url=UNUSED_URL), http)
    assert names == ['fine']
    assert 'vllm' in caplog.text and 'not reported' in caplog.text
