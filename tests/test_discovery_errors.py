"""A failed model discovery is logged so that the owner can tell what went wrong, and reported as unchecked."""
import logging
import socket

import httpx
import pytest

from socketllama.backends import DISCOVERY_TIMEOUT_SECONDS, discover_backend
from socketllama.config import Backend

HTTP_SERVER_ERROR = 500
BACKEND = Backend(name='ollama', kind='ollama', base_url='http://127.0.0.1:11434/v1')


def _failing_http(error: Exception) -> httpx.AsyncClient:
    def handler(request):
        raise error
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_a_timeout_says_after_how_long(caplog):
    async with _failing_http(httpx.ReadTimeout('')) as http:
        with caplog.at_level(logging.WARNING):
            assert await discover_backend(BACKEND, http) is None
    assert f'timed out after {DISCOVERY_TIMEOUT_SECONDS:g} s' in caplog.text


async def test_an_error_without_a_message_is_named_by_its_type(caplog):
    async with _failing_http(httpx.ConnectError('')) as http:
        with caplog.at_level(logging.WARNING):
            assert await discover_backend(BACKEND, http) is None
    assert 'model discovery failed: ConnectError' in caplog.text


async def test_an_error_with_a_message_keeps_it(caplog):
    async with _failing_http(httpx.ConnectError('connection refused')) as http:
        with caplog.at_level(logging.WARNING):
            await discover_backend(BACKEND, http)
    assert 'model discovery failed: ConnectError: connection refused' in caplog.text


def _all_attempts_failed(*attempts: OSError) -> OSError:
    """What the HTTP library's connect raises when a host has several addresses and none could be
    connected (anyio's connect_tcp): one error, caused by the group of the attempts' errors."""
    try:
        raise OSError('All connection attempts failed') from ExceptionGroup('multiple connection attempts failed',
                                                                            list(attempts))
    except OSError as error:
        return error


def _connect_error_caused_by(cause: Exception) -> httpx.ConnectError:
    """A ConnectError as httpx raises it, with the operating system's error in its cause chain."""
    try:
        raise httpx.ConnectError('All connection attempts failed') from cause
    except httpx.ConnectError as error:
        return error


async def test_a_refused_connection_means_the_backend_is_down_and_has_no_models(caplog):
    async with _failing_http(_connect_error_caused_by(ConnectionRefusedError())) as http:
        with caplog.at_level(logging.WARNING):
            assert await discover_backend(BACKEND, http) == []
    assert 'is down' in caplog.text


async def test_a_name_that_cannot_be_resolved_means_the_backend_is_down(caplog):
    async with _failing_http(_connect_error_caused_by(socket.gaierror(socket.EAI_NONAME, 'Name or service not known'))) as http:
        assert await discover_backend(BACKEND, http) == []


async def test_other_connection_failures_leave_the_backend_unchecked():
    async with _failing_http(_connect_error_caused_by(OSError('TLS handshake failed'))) as http:
        assert await discover_backend(BACKEND, http) is None


async def test_an_unusable_answer_leaves_the_backend_unchecked():
    def handler(request):
        return httpx.Response(HTTP_SERVER_ERROR, json={'error': 'busy'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert await discover_backend(BACKEND, http) is None


async def test_a_host_whose_every_address_refuses_is_down():
    """localhost: ::1 and 127.0.0.1, both refused."""
    refused = _all_attempts_failed(ConnectionRefusedError('::1'), ConnectionRefusedError('127.0.0.1'))
    async with _failing_http(_connect_error_caused_by(refused)) as http:
        assert await discover_backend(BACKEND, http) == []


async def test_a_host_with_one_address_that_did_not_refuse_is_unchecked():
    attempts = _all_attempts_failed(ConnectionRefusedError('::1'), TimeoutError('127.0.0.1'))
    async with _failing_http(_connect_error_caused_by(attempts)) as http:
        assert await discover_backend(BACKEND, http) is None


async def test_a_temporary_name_lookup_failure_is_unchecked():
    failure = socket.gaierror(socket.EAI_AGAIN, 'Temporary failure in name resolution')
    async with _failing_http(_connect_error_caused_by(failure)) as http:
        assert await discover_backend(BACKEND, http) is None


@pytest.mark.skipif(not hasattr(socket, 'EAI_NODATA'), reason='the platform has no EAI_NODATA')
async def test_a_name_without_addresses_is_down():
    failure = socket.gaierror(socket.EAI_NODATA, 'No address associated with hostname')
    async with _failing_http(_connect_error_caused_by(failure)) as http:
        assert await discover_backend(BACKEND, http) == []
