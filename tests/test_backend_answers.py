"""Whatever a backend answers, discovery neither crashes nor passes its text on to a terminal unescaped."""
import logging

import httpx
import pytest

from socketllama import backends
from socketllama.backends import (
    NOTE_ATTRIBUTE, NOTE_DETAILS_UNREADABLE, NOTE_DOWN, NOTE_ENTRY_IGNORED, NOTE_FAILED, NOTE_NAMES_REFUSED,
    NOTE_TIMED_OUT, NOTE_TOO_MANY_MODELS, BackendUnavailable, build_models_report, discover_backend,
    fit_report_to_gateway, ollama_loaded_models, shareable_model_names,
)
from socketllama.config import KIND_LITELLM, KIND_OLLAMA, KIND_OPENAI, Backend
from socketllama.text import printable

from .fakes import HTTP_FORBIDDEN, HTTP_OK

HTTP_FOUND = 302
HTTP_UNAUTHORIZED = 401
# Never contacted: every request of these tests is answered by a mock transport
UNUSED_URL = 'http://backend.invalid/v1'
KINDS = (KIND_OLLAMA, KIND_LITELLM, KIND_OPENAI)
# A terminal control sequence (U+009B is "control sequence introducer") and a right-to-left override
HOSTILE = 'evil\x9b31m\u202etxet\x07\nsecond line'
# The part of it a Location header can carry (the HTTP library refuses ASCII control characters there)
HOSTILE_LOCATION = 'evil\x9b31m\u202etxet'
NOT_A_LIST = [7, True, 'text', {'id': 'model'}]
SMALL_LIMIT = 1


def backend_of(kind: str) -> Backend:
    return Backend(name='b', kind=kind, base_url=UNUSED_URL)


def answering(payload=None, status=HTTP_OK, headers=None) -> httpx.AsyncClient:
    def handler(request):
        return httpx.Response(status, json=payload, headers=headers)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def notes_of(caplog) -> list:
    return [getattr(record, NOTE_ATTRIBUTE) for record in caplog.records if hasattr(record, NOTE_ATTRIBUTE)]


def test_printable_escapes_what_a_terminal_would_act_on():
    shown = printable(HOSTILE)
    assert shown.isprintable()
    assert shown == 'evil\\x9b31m\\u202etxet\\x07\\nsecond line'
    assert printable('plain text, ünïcödé 🦙') == 'plain text, ünïcödé 🦙'
    assert printable('\ud800').isprintable()


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('data', NOT_A_LIST)
async def test_a_model_list_that_is_no_list_is_an_answer_that_cannot_be_used(kind, data):
    async with answering({'data': data}) as http:
        with pytest.raises(BackendUnavailable, match='not a list') as raised:
            await shareable_model_names(backend_of(kind), http)
    assert raised.value.answered


@pytest.mark.parametrize('kind', KINDS)
async def test_a_model_list_that_is_no_list_leaves_the_backend_unchecked(kind):
    async with answering({'data': True}) as http:
        assert await discover_backend(backend_of(kind), http) is None


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('payload', [{}, {'data': None}, {'data': []}])
async def test_no_model_list_or_an_empty_one_is_a_backend_without_models(kind, payload):
    async with answering(payload) as http:
        assert await shareable_model_names(backend_of(kind), http) == []


@pytest.mark.parametrize('models', NOT_A_LIST)
async def test_a_list_of_loaded_models_that_is_no_list_is_unknown(models):
    async with answering({'models': models}) as http:
        assert await ollama_loaded_models(backend_of(KIND_OLLAMA), http) is None


async def test_a_refusal_carries_its_status():
    async with answering({'error': 'no'}, status=HTTP_UNAUTHORIZED) as http:
        with pytest.raises(BackendUnavailable) as raised:
            await shareable_model_names(backend_of(KIND_OPENAI), http)
    assert raised.value.status == HTTP_UNAUTHORIZED and raised.value.answered


async def test_a_failure_without_an_answer_has_no_status():
    def handler(request):
        raise httpx.ConnectError('no route')
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(BackendUnavailable) as raised:
            await shareable_model_names(backend_of(KIND_OPENAI), http)
    assert raised.value.status is None


async def test_text_of_the_server_in_an_error_is_escaped(caplog):
    """httpx quotes a redirect's Location header in its message."""
    headers = {'location': f'http://elsewhere.invalid/{HOSTILE_LOCATION}'.encode()}
    async with answering({}, status=HTTP_FOUND, headers=headers) as http:
        with pytest.raises(BackendUnavailable) as raised:
            await shareable_model_names(backend_of(KIND_OPENAI), http)
        with caplog.at_level(logging.WARNING):
            assert await discover_backend(backend_of(KIND_OPENAI), http) is None
    assert 'elsewhere.invalid' in str(raised.value) and str(raised.value).isprintable()
    assert raised.value.status == HTTP_FOUND
    assert all(record.getMessage().isprintable() for record in caplog.records)


async def test_a_model_name_of_the_server_in_a_warning_is_escaped(caplog):
    entries = [{'model_name': HOSTILE, 'model_info': 'not an object'}, {'model_name': 'fine', 'model_info': {}}]
    async with answering({'data': entries}) as http:
        with caplog.at_level(logging.WARNING):
            assert await shareable_model_names(backend_of(KIND_LITELLM), http) == ['fine']
    assert caplog.records and all(record.getMessage().isprintable() for record in caplog.records)
    assert [(note.kind, note.names) for note in notes_of(caplog)] == [(NOTE_ENTRY_IGNORED, (HOSTILE,))]


# ----- what discovery logs, it also says in a form a program can read -----

async def test_a_proxy_whose_details_the_key_may_not_read_is_noted(caplog):
    def handler(request):
        if request.url.path.endswith(backends.LITELLM_MODEL_INFO_PATH):
            return httpx.Response(HTTP_FORBIDDEN, json={})
        return httpx.Response(HTTP_OK, json={'data': [{'id': 'gpt-proxy'}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with caplog.at_level(logging.INFO):
            assert await shareable_model_names(backend_of(KIND_LITELLM), http) == ['gpt-proxy']
    assert [(note.kind, note.backend, note.status) for note in notes_of(caplog)] == [
        (NOTE_DETAILS_UNREADABLE, 'b', HTTP_FORBIDDEN)]


def test_models_left_out_of_a_report_are_noted_with_the_reason(monkeypatch, caplog):
    monkeypatch.setattr(backends.protocol, 'MAX_MODELS_PER_BACKEND', SMALL_LIMIT)
    report = [{'name': 'b', 'kind': KIND_OPENAI, 'models': [{'name': 'kept'}, {'name': 'tab\there'},
                                                           {'name': 'one too many'}]}]
    with caplog.at_level(logging.WARNING):
        fit_report_to_gateway(report)
    assert [(note.kind, note.backend, note.names) for note in notes_of(caplog)] == [
        (NOTE_NAMES_REFUSED, 'b', ('tab\there',)), (NOTE_TOO_MANY_MODELS, 'b', ('one too many',))]


@pytest.mark.parametrize('error, kind', [
    (httpx.ReadTimeout(''), NOTE_TIMED_OUT),
    (httpx.ConnectError('TLS handshake failed'), NOTE_FAILED),
])
async def test_a_backend_that_could_not_be_listed_is_noted_with_the_problem(caplog, error, kind):
    def handler(request):
        raise error
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with caplog.at_level(logging.INFO):
            await build_models_report({'b': backend_of(KIND_OPENAI)}, http)
    notes = notes_of(caplog)
    assert [(note.kind, note.backend) for note in notes] == [(kind, 'b')]
    assert isinstance(notes[0].problem, BackendUnavailable)


async def test_a_backend_that_is_down_is_noted(caplog):
    def handler(request):
        try:
            raise httpx.ConnectError('refused') from ConnectionRefusedError()
        except httpx.ConnectError as error:
            raise error
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with caplog.at_level(logging.WARNING):
            assert await discover_backend(backend_of(KIND_OPENAI), http) == []
    assert [note.kind for note in notes_of(caplog)] == [NOTE_DOWN]
