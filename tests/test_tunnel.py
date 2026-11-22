"""The client end to end: fake gateway <-WebSocket-> TunnelClient <-HTTP-> fake OpenAI-compatible backend."""
import asyncio
import itertools
import json
import logging

import pytest

from socketllama import __version__, protocol
from socketllama import backends as backends_module
from socketllama import client as client_module
from socketllama.backends import DEFAULT_OLLAMA_CONTEXT_LENGTH
from socketllama.client import (
    MODELS_REPORT_RESEND_SECONDS, REPLACED_RETRY_SECONDS, STABLE_CONNECTION_SECONDS, WIRE_LOGGER_NAME,
    FatalGatewayError, GatewayConnectionLost, TunnelClient,
)
from socketllama.relay import Relay

from .conftest import (
    SHARE_ALL_MODELS, STARTUP_TIMEOUT_SECONDS, litellm_backend, make_config, ollama_backend, wait_until,
)
from .fakes import (
    BACKEND_API_KEY, CONTEXT_LENGTH, EMBEDDING_MODEL, HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_OK, INSTANCE, MODEL, NONCE, SIGNING_SECRET,
    STREAM_PARTS, TOKEN,
    FakeBackend, FakeGateway, sse_text,
)

CHAT = {'model': MODEL, 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}]}
QUIET_SECONDS = 0.2
RUNTIME_CONTEXT_LENGTH = 16384
MODELFILE_NUM_CTX = 8192
CONFIGURED_CONTEXT_LENGTH = 32768
REFUSED_BURST = 8
SMALL_FRAME_LIMIT_BYTES = 4096
# enough leading characters of a base64 body to recognise it in a log line
SECRET_PROBE_CHARS = 24
RECONNECT_MIN = 0.5
RECONNECT_MAX = 1.5
FULL_JITTER = 1.0
# doubling from RECONNECT_MIN, capped at RECONNECT_MAX
EXPECTED_DELAYS = [0.5, 1.0, 1.5, 1.5]
FROZEN_TIME = 0.0
RESEND_WINDOWS_WATCHED = 3


async def start(client: TunnelClient) -> asyncio.Task:
    return asyncio.create_task(client.run_forever())


async def finish(client: TunnelClient, task: asyncio.Task):
    client.stop()
    await asyncio.wait_for(task, STARTUP_TIMEOUT_SECONDS)


# ----- handshake and model report -----

async def test_debug_logging_shows_frames_but_no_credentials_or_content(gateway, backend, caplog):
    caplog.set_level(logging.DEBUG)
    # cli.main() (run by other tests) quiets the wire logger as it does without -v
    caplog.set_level(logging.DEBUG, logger=WIRE_LOGGER_NAME)
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await gateway.send_request('req-1', CHAT)
    await gateway.collect_response('req-1')
    await finish(client, task)
    # the fake gateway's own (server-side) websockets logger is not the client's business
    client_records = [record for record in caplog.records if not record.name.startswith('websockets.server')]
    logged = '\n'.join(record.getMessage() for record in client_records)
    for secret in (TOKEN, SIGNING_SECRET, protocol.encode_body(json.dumps(CHAT).encode())[:SECRET_PROBE_CHARS],
                   STREAM_PARTS[0] + STREAM_PARTS[1]):
        assert secret not in logged
    assert any(record.name == WIRE_LOGGER_NAME for record in client_records)  # frames are still traced


async def test_authenticates_with_token_instance_and_protocol(gateway, running_client):
    assert gateway.auth_frames == [{
        'type': protocol.FRAME_AUTH, 'token': TOKEN, 'instance': INSTANCE,
        'protocol': protocol.PROTOCOL_VERSION, 'client_version': __version__,
        'proof': protocol.sign_auth(SIGNING_SECRET, NONCE, INSTANCE),
        'capabilities': list(protocol.CLIENT_CAPABILITIES),
    }]


async def test_refuses_a_gateway_that_sends_no_nonce(backend):
    async with FakeGateway(nonce=None) as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
        with pytest.raises(FatalGatewayError, match='nonce'):
            await asyncio.wait_for(client.run_forever(), STARTUP_TIMEOUT_SECONDS)
        assert gateway.auth_frames == []  # the token was never sent


async def test_reports_ollama_chat_models_with_a_conservative_context_length(gateway, running_client):
    report = await gateway.next_frame(protocol.FRAME_MODELS)
    # not loaded, no num_ctx: Ollama serves the default window, not the architecture's maximum
    assert report['backends'] == [{'name': 'ollama', 'kind': 'ollama', 'models': [
        {'name': MODEL, 'context_length': DEFAULT_OLLAMA_CONTEXT_LENGTH, 'supports_reasoning': False}]}]


async def _reported_context_length(gateway, backend_options=None, **fake_options) -> int:
    async with FakeBackend(**fake_options) as backend:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend, **(backend_options or {}))]))
        task = await start(client)
        report = await gateway.next_frame(protocol.FRAME_MODELS)
        await finish(client, task)
    return report['backends'][0]['models'][0]['context_length']


@pytest.mark.parametrize('fake_options, backend_options, expected', [
    # the window of the loaded model is what requests get
    ({'loaded_context_length': RUNTIME_CONTEXT_LENGTH, 'modelfile_num_ctx': MODELFILE_NUM_CTX}, None,
     RUNTIME_CONTEXT_LENGTH),
    ({'modelfile_num_ctx': MODELFILE_NUM_CTX}, None, MODELFILE_NUM_CTX),
    ({'process_list_status': HTTP_NOT_FOUND}, None, DEFAULT_OLLAMA_CONTEXT_LENGTH),
    # the configuration wins over everything Ollama says ...
    ({'loaded_context_length': RUNTIME_CONTEXT_LENGTH}, {'context_length': CONFIGURED_CONTEXT_LENGTH},
     CONFIGURED_CONTEXT_LENGTH),
    # ... but never exceeds what the model architecture supports
    ({}, {'context_length': CONTEXT_LENGTH * 2}, CONTEXT_LENGTH),
    ({'loaded_context_length': CONTEXT_LENGTH * 2}, None, CONTEXT_LENGTH),
])
async def test_ollama_context_length_sources(gateway, fake_options, backend_options, expected):
    assert await _reported_context_length(gateway, backend_options, **fake_options) == expected


async def test_reports_litellm_models_with_limits_and_prices(gateway, backend):
    client = TunnelClient(make_config(gateway.url, [litellm_backend(backend)]))
    task = await start(client)
    report = await gateway.next_frame(protocol.FRAME_MODELS)
    assert report['backends'][0]['models'] == [{
        'name': 'gpt-proxy', 'context_length': CONTEXT_LENGTH, 'max_output_tokens': 4096,
        'input_cost_per_token': 2e-06, 'output_cost_per_token': 8e-06, 'supports_reasoning': True}]
    info_request = next(r for r in backend.requests if r['path'] == '/v1/model/info')
    assert info_request['headers']['authorization'] == f'Bearer {BACKEND_API_KEY}'
    await finish(client, task)


async def test_a_configured_context_length_applies_to_other_backends_too(gateway, backend):
    client = TunnelClient(make_config(gateway.url, [litellm_backend(backend, context_length=MODELFILE_NUM_CTX)]))
    task = await start(client)
    report = await gateway.next_frame(protocol.FRAME_MODELS)
    assert report['backends'][0]['models'][0]['context_length'] == MODELFILE_NUM_CTX
    await finish(client, task)


async def test_litellm_key_without_model_info_access_falls_back_to_the_model_list(gateway):
    async with FakeBackend(model_info_status=HTTP_FORBIDDEN) as backend:
        client = TunnelClient(make_config(gateway.url, [litellm_backend(backend)]))
        task = await start(client)
        report = await gateway.next_frame(protocol.FRAME_MODELS)
        assert {model['name'] for model in report['backends'][0]['models']} >= {MODEL}
        await finish(client, task)


async def test_malformed_litellm_model_info_is_skipped_not_fatal(gateway):
    malformed = [{'model_name': 'listy', 'model_info': ['not', 'a', 'dict']},
                 {'model_name': 'stringy', 'model_info': 'chat'}]
    async with FakeBackend(extra_model_info_entries=malformed) as backend:
        client = TunnelClient(make_config(gateway.url, [litellm_backend(backend)]))
        task = await start(client)
        report = await gateway.next_frame(protocol.FRAME_MODELS)
        assert [model['name'] for model in report['backends'][0]['models']] == ['gpt-proxy']
        await finish(client, task)


async def test_a_failing_model_report_is_logged_and_the_tunnel_keeps_working(gateway, backend, monkeypatch,
                                                                            caplog):
    async def broken_report(*args, **kwargs):
        raise RuntimeError('discovery bug')

    monkeypatch.setattr(client_module, 'build_models_report', broken_report)
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await wait_until(lambda: 'discovery bug' in caplog.text)
    await gateway.send_request('req-1', CHAT)
    _, _, final = await gateway.collect_response('req-1')
    assert final['type'] == protocol.FRAME_RESPONSE_END
    assert client.connections_established == 1
    await finish(client, task)


async def test_a_crashed_keep_alive_drops_the_connection_instead_of_leaving_it_unwatched(gateway, backend,
                                                                                        monkeypatch, caplog):
    async def broken_keep_alive(self, socket, pong_received):
        raise RuntimeError('keep-alive bug')

    monkeypatch.setattr(TunnelClient, '_keep_alive', broken_keep_alive)
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: client.connections_established >= 2)
    assert 'keep-alive bug' in caplog.text
    await finish(client, task)


async def test_sharing_every_model_is_warned_about_at_start(gateway, backend, caplog):
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await finish(client, task)
    assert 'shares every model' in caplog.text


async def test_model_allow_list_limits_the_report(gateway, backend):
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend, models=['not-pulled'])]))
    task = await start(client)
    report = await gateway.next_frame(protocol.FRAME_MODELS)
    assert report['backends'][0]['models'] == []
    assert not [request for request in backend.requests if request['path'] == '/api/show']
    await finish(client, task)


async def test_litellm_backend_reports_only_the_listed_models(gateway):
    other_chat_model = {'model_name': 'other-chat', 'model_info': {'mode': 'chat'}}
    async with FakeBackend(extra_model_info_entries=[other_chat_model]) as backend:
        client = TunnelClient(make_config(gateway.url, [litellm_backend(backend, models=['gpt-proxy'])]))
        task = await start(client)
        report = await gateway.next_frame(protocol.FRAME_MODELS)
        assert [model['name'] for model in report['backends'][0]['models']] == ['gpt-proxy']
        await finish(client, task)


async def test_litellm_backend_without_model_info_reports_only_the_listed_models(gateway):
    async with FakeBackend(model_info_status=HTTP_FORBIDDEN) as backend:
        client = TunnelClient(make_config(gateway.url, [litellm_backend(backend, models=[EMBEDDING_MODEL])]))
        task = await start(client)
        report = await gateway.next_frame(protocol.FRAME_MODELS)
        assert [model['name'] for model in report['backends'][0]['models']] == [EMBEDDING_MODEL]
        await finish(client, task)


async def test_openai_backend_reports_only_the_listed_models(gateway, backend):
    openai_backend = {'name': 'vllm', 'kind': 'openai', 'base_url': backend.base_url, 'models': [MODEL]}
    client = TunnelClient(make_config(gateway.url, [openai_backend]))
    task = await start(client)
    report = await gateway.next_frame(protocol.FRAME_MODELS)
    assert report['backends'] == [{'name': 'vllm', 'kind': 'openai', 'models': [{'name': MODEL}]}]
    await finish(client, task)


async def test_the_report_sent_is_trimmed_to_the_gateway_limits(gateway, backend, monkeypatch, caplog):
    monkeypatch.setattr(protocol, 'MAX_MODELS_PER_BACKEND', 1)
    openai_backend = {'name': 'vllm', 'kind': 'openai', 'base_url': backend.base_url, 'models': SHARE_ALL_MODELS}
    client = TunnelClient(make_config(gateway.url, [openai_backend]))
    task = await start(client)
    report = await gateway.next_frame(protocol.FRAME_MODELS)
    assert report['backends'][0]['models'] == [{'name': MODEL}]
    assert EMBEDDING_MODEL in caplog.text
    await finish(client, task)


DEAD_BACKEND = {'name': 'ollama', 'kind': 'ollama', 'base_url': 'http://127.0.0.1:9/v1', 'models': SHARE_ALL_MODELS}
UNCHECKED_CAPABLE = [protocol.CAPABILITY_MODELS_UNCHECKED]
SHORT_DISCOVERY_TIMEOUT_SECONDS = 0.1


async def _first_report(gateway, backends: list) -> dict:
    client = TunnelClient(make_config(gateway.url, backends))
    task = await start(client)
    report = await gateway.next_frame(protocol.FRAME_MODELS)
    await finish(client, task)
    return report


async def test_a_backend_that_refuses_connections_is_reported_down_without_models(backend):
    """Known to be down: the gateway deactivates its models and keeps them for the retention as always."""
    async with FakeGateway(capabilities=UNCHECKED_CAPABLE) as gateway:
        report = await _first_report(gateway, [DEAD_BACKEND, litellm_backend(backend)])
    assert report['backends'][0] == {'name': 'ollama', 'kind': 'ollama', 'models': []}


async def test_a_backend_that_answers_too_late_is_reported_as_unchecked(backend, monkeypatch):
    """The owner's case: Ollama busy loading a model does not answer discovery in time."""
    monkeypatch.setattr(backends_module, 'DISCOVERY_TIMEOUT_SECONDS', SHORT_DISCOVERY_TIMEOUT_SECONDS)
    backend.hold_discovery.clear()
    async with FakeBackend() as working, FakeGateway(capabilities=UNCHECKED_CAPABLE) as gateway:
        report = await _first_report(gateway, [ollama_backend(backend), litellm_backend(working)])
    assert report['backends'][0] == {'name': 'ollama', 'kind': 'ollama', 'checked': False, 'models': []}
    # a backend that was checked carries no marker: the field is only ever false
    assert 'checked' not in report['backends'][1]
    assert report['backends'][1]['models']


async def test_a_gateway_without_models_unchecked_gets_the_backend_without_models(gateway, backend, monkeypatch):
    """No marker it does not know, and the reports of the other backends keep coming."""
    monkeypatch.setattr(backends_module, 'DISCOVERY_TIMEOUT_SECONDS', SHORT_DISCOVERY_TIMEOUT_SECONDS)
    backend.hold_discovery.clear()
    async with FakeBackend() as working:
        report = await _first_report(gateway, [ollama_backend(backend), litellm_backend(working)])
    assert report['backends'][0] == {'name': 'ollama', 'kind': 'ollama', 'models': []}
    assert report['backends'][1]['models']


# ----- relaying -----

async def test_streams_a_chat_completion(gateway, backend, running_client):
    await gateway.send_request('req-1', CHAT)
    start_frame, body, final = await gateway.collect_response('req-1')
    assert start_frame['status'] == HTTP_OK
    assert start_frame['headers'] == {'content-type': 'text/event-stream'}
    assert sse_text(body) == ''.join(STREAM_PARTS)
    assert body.endswith(b'data: [DONE]\n\n')
    assert final['type'] == protocol.FRAME_RESPONSE_END

    relayed = next(r for r in backend.requests if r['path'] == '/v1/chat/completions')
    assert relayed['json'] == CHAT
    assert 'authorization' not in relayed['headers']  # this backend has no api_key configured
    # raw bytes are relayed and only content-type travels back: a compressed body would be unreadable
    assert relayed['headers']['accept-encoding'] == 'identity'


async def test_chunks_arrive_while_the_backend_is_still_generating(gateway, backend, running_client):
    backend.hold_stream.clear()
    await gateway.send_request('req-1', CHAT)
    await gateway.next_frame(protocol.FRAME_RESPONSE_START)
    first = await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
    assert STREAM_PARTS[0] in protocol.decode_body(first['data']).decode()
    backend.hold_stream.set()
    assert (await gateway.next_frame(protocol.FRAME_RESPONSE_END))['id'] == 'req-1'


async def test_non_streaming_completion(gateway, running_client):
    await gateway.send_request('req-1', {**CHAT, 'stream': False})
    start_frame, body, final = await gateway.collect_response('req-1')
    assert json.loads(body)['choices'][0]['message']['content'] == ''.join(STREAM_PARTS)
    assert final['type'] == protocol.FRAME_RESPONSE_END


async def test_backend_error_status_is_passed_through(gateway, running_client):
    await gateway.send_request('req-1', {**CHAT, 'model': 'missing'})
    start_frame, body, final = await gateway.collect_response('req-1')
    assert start_frame['status'] == HTTP_NOT_FOUND
    assert b'model not found' in body
    assert final['type'] == protocol.FRAME_RESPONSE_END


async def test_backend_credentials_are_added_by_the_client(gateway, backend):
    client = TunnelClient(make_config(gateway.url, [litellm_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await gateway.send_request('req-1', CHAT, backend='litellm')
    await gateway.collect_response('req-1')
    relayed = next(r for r in backend.requests if r['path'] == '/v1/chat/completions')
    assert relayed['headers']['authorization'] == f'Bearer {BACKEND_API_KEY}'
    await finish(client, task)


async def test_concurrent_requests_are_kept_apart(gateway, running_client):
    await gateway.send_request('req-a', CHAT)
    await gateway.send_request('req-b', {**CHAT, 'stream': False})
    bodies = {'req-a': b'', 'req-b': b''}
    finished = set()
    while finished != set(bodies):
        frame = await gateway.next_frame()
        if frame['type'] == protocol.FRAME_RESPONSE_CHUNK:
            bodies[frame['id']] += protocol.decode_body(frame['data'])
        elif frame['type'] == protocol.FRAME_RESPONSE_END:
            finished.add(frame['id'])
    assert sse_text(bodies['req-a']) == ''.join(STREAM_PARTS)
    assert json.loads(bodies['req-b'])['choices']


# ----- what the client refuses -----

def _relayed_chats(fake_backend) -> list:
    return [request for request in fake_backend.requests if request['path'] == '/v1/chat/completions']


async def _refused(gateway, fake_backend, expected_code, payload=CHAT, **request):
    """Send a request the client must refuse: it answers with an error and never calls the backend."""
    relayed_before = len(_relayed_chats(fake_backend))
    await gateway.send_request('req-x', payload, **request)
    frame = await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    assert (frame['id'], frame['code']) == ('req-x', expected_code)
    assert len(_relayed_chats(fake_backend)) == relayed_before


async def test_refuses_a_request_signed_with_another_secret(gateway, backend, running_client):
    await _refused(gateway, backend, protocol.ERROR_BAD_SIGNATURE, secret='not-the-secret')


async def test_refuses_an_unsigned_request(gateway, backend, running_client):
    await _refused(gateway, backend, protocol.ERROR_BAD_SIGNATURE, signature='')


async def test_refuses_a_request_whose_body_was_changed_after_signing(gateway, backend, running_client):
    tampered = protocol.encode_body(json.dumps({**CHAT, 'model': 'other'}).encode())
    await _refused(gateway, backend, protocol.ERROR_BAD_SIGNATURE, body=tampered)


async def test_refuses_a_backend_that_is_not_configured(gateway, backend, running_client):
    await _refused(gateway, backend, protocol.ERROR_UNKNOWN_BACKEND, backend='http://169.254.169.254')


@pytest.mark.parametrize('method, path', [
    ('GET', '/models'),
    ('POST', '/api/pull'),
    ('DELETE', '/api/delete'),
    ('POST', '/chat/completions/../../api/delete'),
    ('POST', '@evil.example/chat/completions'),
])
async def test_refuses_everything_but_chat_completions(gateway, backend, running_client, method, path):
    await _refused(gateway, backend, protocol.ERROR_PATH_NOT_ALLOWED, method=method, path=path)


async def test_refuses_a_model_outside_the_backends_allow_list(gateway, backend):
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend, models=[MODEL])]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await _refused(gateway, backend, protocol.ERROR_MODEL_NOT_ALLOWED, payload={**CHAT, 'model': 'qwen3:14b'})
    await finish(client, task)


async def test_refuses_a_replayed_request_id(gateway, backend, running_client):
    await gateway.send_request('req-1', CHAT)
    await gateway.collect_response('req-1')
    await gateway.send_request('req-1', CHAT)
    frame = await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    assert frame['code'] == protocol.ERROR_REPLAYED_REQUEST
    assert len(_relayed_chats(backend)) == 1


async def test_refuses_a_frame_captured_on_an_earlier_connection(gateway, backend, running_client):
    captured = gateway.request_frame('req-old', CHAT)
    await gateway.drop_connection()
    await wait_until(lambda: len(gateway.connections) == 2)
    await gateway.send(captured)  # same id never seen on this connection: only the nonce binding catches it
    frame = await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    assert (frame['id'], frame['code']) == ('req-old', protocol.ERROR_BAD_SIGNATURE)
    assert _relayed_chats(backend) == []


async def test_reusing_the_id_of_a_request_in_flight_leaves_that_request_alone(gateway, backend, running_client):
    backend.hold_stream.clear()
    await gateway.send_request('req-1', CHAT)
    await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
    await gateway.send_request('req-1', CHAT)
    assert (await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR))['code'] == protocol.ERROR_REPLAYED_REQUEST
    # the first request is still tracked: cancelling it works
    await gateway.send({'type': protocol.FRAME_CANCEL, 'id': 'req-1'})
    await asyncio.sleep(QUIET_SECONDS)
    backend.hold_stream.set()
    await asyncio.sleep(QUIET_SECONDS)
    assert gateway.drain_frames_of('req-1') == []
    assert len(_relayed_chats(backend)) == 1


async def test_refused_requests_do_not_take_request_slots(gateway, backend):
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)], max_concurrent_requests=1))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    for number in range(REFUSED_BURST):
        await gateway.send_request(f'bad-{number}', CHAT, secret='not-the-secret')
    await gateway.send_request('good', CHAT)
    _, body, final = await gateway.collect_response('good')
    assert final['type'] == protocol.FRAME_RESPONSE_END
    assert sse_text(body) == ''.join(STREAM_PARTS)
    await finish(client, task)


async def test_requests_beyond_the_limit_are_refused_as_busy(gateway, backend):
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)], max_concurrent_requests=1))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    backend.hold_stream.clear()
    await gateway.send_request('first', CHAT)
    await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
    await gateway.send_request('second', CHAT)
    frame = await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    assert (frame['id'], frame['code']) == ('second', protocol.ERROR_CLIENT_BUSY)
    backend.hold_stream.set()
    await finish(client, task)


async def test_an_unexpected_error_is_answered_not_swallowed(gateway, backend, running_client, monkeypatch):
    async def broken_forward(*args, **kwargs):
        raise RuntimeError('bug in the relay')

    monkeypatch.setattr(Relay, '_forward', broken_forward)
    await gateway.send_request('req-1', CHAT)
    frame = await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    assert (frame['id'], frame['code']) == ('req-1', protocol.ERROR_CLIENT_INTERNAL)
    assert 'bug in the relay' not in frame['message']  # internals stay in the local log


async def test_a_request_above_the_size_limit_fails_alone(gateway, backend, monkeypatch):
    monkeypatch.setattr(protocol, 'MAX_REQUEST_FRAME_BYTES', SMALL_FRAME_LIMIT_BYTES)
    monkeypatch.setattr(client_module, 'MAX_GATEWAY_FRAME_BYTES', SMALL_FRAME_LIMIT_BYTES * 2)
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    oversized = {**CHAT, 'messages': [{'role': 'user', 'content': 'x' * SMALL_FRAME_LIMIT_BYTES}]}
    await gateway.send_request('too-big', oversized)
    frame = await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    assert (frame['id'], frame['code']) == ('too-big', protocol.ERROR_REQUEST_TOO_LARGE)
    await gateway.send_request('normal', CHAT)  # the tunnel is still up
    _, _, final = await gateway.collect_response('normal')
    assert final['type'] == protocol.FRAME_RESPONSE_END
    assert client.connections_established == 1
    await finish(client, task)


async def test_refuses_a_body_that_is_not_a_json_object(gateway, backend, running_client):
    await _refused(gateway, backend, protocol.ERROR_INVALID_REQUEST, payload=['not', 'an', 'object'])


async def test_unreachable_backend_is_reported_per_request(gateway):
    dead = {'name': 'ollama', 'kind': 'ollama', 'base_url': 'http://127.0.0.1:9/v1', 'models': SHARE_ALL_MODELS}
    client = TunnelClient(make_config(gateway.url, [dead]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await gateway.send_request('req-1', CHAT)
    frame = await gateway.next_frame(protocol.FRAME_RESPONSE_ERROR)
    assert frame['code'] == protocol.ERROR_BACKEND_UNREACHABLE
    await finish(client, task)


# ----- flow control -----

async def test_cancel_stops_the_upstream_call(gateway, backend, running_client):
    backend.hold_stream.clear()
    await gateway.send_request('req-1', CHAT)
    await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
    await gateway.send({'type': protocol.FRAME_CANCEL, 'id': 'req-1'})
    await asyncio.sleep(QUIET_SECONDS)
    backend.hold_stream.set()
    await asyncio.sleep(QUIET_SECONDS)
    # nothing more for the cancelled request: no further chunk, no end, no error
    assert gateway.drain_frames_of('req-1') == []


async def test_pause_holds_chunks_until_resume(gateway, backend, running_client):
    backend.hold_stream.clear()
    await gateway.send_request('req-1', CHAT)
    await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
    await gateway.send({'type': protocol.FRAME_PAUSE, 'id': 'req-1'})
    await asyncio.sleep(QUIET_SECONDS)
    backend.hold_stream.set()
    await asyncio.sleep(QUIET_SECONDS)
    assert gateway.drain_frames_of('req-1') == []  # the backend finished, the client holds the rest back

    await gateway.send({'type': protocol.FRAME_RESUME, 'id': 'req-1'})
    assert (await gateway.next_frame(protocol.FRAME_RESPONSE_END))['id'] == 'req-1'


# ----- keep-alive and reconnect -----

async def test_pings_the_gateway(backend, fast_keepalive):
    pings = []

    class CountingGateway(FakeGateway):
        async def _handle(self, socket):
            original_send = socket.send

            async def counting_send(message):
                if json.loads(message).get('type') == protocol.FRAME_PONG:
                    pings.append(message)
                await original_send(message)
            socket.send = counting_send
            await super()._handle(socket)

    async with CountingGateway() as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)], **fast_keepalive))
        task = await start(client)
        await wait_until(lambda: len(pings) >= 2)
        await finish(client, task)
    assert client.connections_established == 1


async def test_missing_pong_makes_the_client_reconnect(backend, fast_keepalive):
    async with FakeGateway(answer_pings=False) as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)], **fast_keepalive))
        task = await start(client)
        await wait_until(lambda: client.connections_established >= 2)
        await finish(client, task)


async def test_reconnects_after_the_gateway_drops_the_connection(gateway, backend, running_client):
    await gateway.next_frame(protocol.FRAME_MODELS)
    await gateway.drop_connection()
    await wait_until(lambda: len(gateway.connections) == 2)
    # the new connection is fully usable: models are reported again and requests are relayed
    await gateway.next_frame(protocol.FRAME_MODELS)
    await gateway.send_request('req-after', CHAT)
    _, body, final = await gateway.collect_response('req-after')
    assert sse_text(body) == ''.join(STREAM_PARTS)
    assert running_client.connections_established == 2


class ManualClock:
    def __init__(self):
        self.now = FROZEN_TIME

    def __call__(self) -> float:
        return self.now


async def test_a_reconnect_does_not_resend_a_report_the_gateway_just_acknowledged(gateway, backend):
    clock = ManualClock()
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]), clock=clock)
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await gateway.next_frame(protocol.FRAME_MODELS)
    await gateway.send({'type': protocol.FRAME_MODELS_ACK, 'created': [MODEL]})
    await asyncio.sleep(QUIET_SECONDS)

    await gateway.drop_connection()
    await wait_until(lambda: len(gateway.connections) == 2)
    await asyncio.sleep(QUIET_SECONDS)
    assert [frame for frame in gateway.drain_frames() if frame['type'] == protocol.FRAME_MODELS] == []

    clock.now += MODELS_REPORT_RESEND_SECONDS  # long enough: the next connection reports again
    await gateway.drop_connection()
    await wait_until(lambda: len(gateway.connections) == 3)
    await gateway.next_frame(protocol.FRAME_MODELS)
    await finish(client, task)


async def test_an_unacknowledged_report_is_sent_again_on_reconnect(gateway, backend):
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]), clock=ManualClock())
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await gateway.next_frame(protocol.FRAME_MODELS)  # no ack: e.g. the gateway could not store it
    await gateway.drop_connection()
    await wait_until(lambda: len(gateway.connections) == 2)
    await gateway.next_frame(protocol.FRAME_MODELS)
    await finish(client, task)


async def test_a_report_the_gateway_ignored_is_resent_once_its_rate_limit_passed(gateway, backend, monkeypatch):
    """After a restart the gateway may still hold the previous run's report time and ignore the first report."""
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', QUIET_SECONDS)
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await gateway.next_frame(protocol.FRAME_MODELS)
    await gateway.send({'type': protocol.FRAME_ERROR, 'message': 'Models report ignored: reports are accepted '
                                                                 'at most every 60 seconds'})
    resent = await gateway.next_frame(protocol.FRAME_MODELS)  # well before MODEL_REFRESH_INTERVAL_SECONDS
    assert resent['backends'][0]['models']
    assert client.connections_established == 1
    await finish(client, task)


async def test_an_ignored_first_report_is_resent_only_once_per_connection(gateway, backend, monkeypatch):
    """A gateway that never acknowledges must not get a resend every MODELS_REPORT_RESEND_SECONDS."""
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', QUIET_SECONDS)
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await gateway.next_frame(protocol.FRAME_MODELS)
    await gateway.next_frame(protocol.FRAME_MODELS)  # the one resend
    await asyncio.sleep(QUIET_SECONDS * RESEND_WINDOWS_WATCHED)
    assert [frame for frame in gateway.drain_frames() if frame['type'] == protocol.FRAME_MODELS] == []
    assert client.connections_established == 1
    await finish(client, task)


async def test_an_acknowledged_report_is_not_resent_early(gateway, backend, monkeypatch):
    monkeypatch.setattr(client_module, 'MODELS_REPORT_RESEND_SECONDS', QUIET_SECONDS)
    client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
    task = await start(client)
    await wait_until(lambda: gateway.connections)
    await gateway.next_frame(protocol.FRAME_MODELS)
    await gateway.send({'type': protocol.FRAME_MODELS_ACK, 'created': [MODEL]})
    await asyncio.sleep(QUIET_SECONDS * RESEND_WINDOWS_WATCHED)
    assert [frame for frame in gateway.drain_frames() if frame['type'] == protocol.FRAME_MODELS] == []
    await finish(client, task)


async def test_backs_off_exponentially_while_the_gateway_is_down(backend):
    delays = []

    async def recording_sleep(delay):
        delays.append(delay)
        await asyncio.sleep(0)

    async with FakeGateway() as gateway:
        url = gateway.url  # the port is free again once the fake is closed
    config = make_config(url, [ollama_backend(backend)], reconnect_min=RECONNECT_MIN, reconnect_max=RECONNECT_MAX)
    client = TunnelClient(config, sleep=recording_sleep, jitter=lambda: FULL_JITTER)
    task = await start(client)
    await wait_until(lambda: len(delays) >= len(EXPECTED_DELAYS))
    await finish(client, task)
    assert delays[:len(EXPECTED_DELAYS)] == EXPECTED_DELAYS
    assert client.connections_established == 0


async def test_backoff_starts_over_after_a_successful_connection(gateway, backend):
    delays = []

    async def recording_sleep(delay):
        delays.append(delay)
        await asyncio.sleep(0)

    config = make_config(gateway.url, [ollama_backend(backend)], reconnect_min=RECONNECT_MIN,
                         reconnect_max=RECONNECT_MAX)
    # every reading of the clock is STABLE_CONNECTION_SECONDS later: each connection counts as stable
    clock = itertools.count(step=STABLE_CONNECTION_SECONDS).__next__
    client = TunnelClient(config, sleep=recording_sleep, jitter=lambda: FULL_JITTER, clock=clock)
    task = await start(client)
    for expected_connections in (1, 2):
        await wait_until(lambda: len(gateway.connections) == expected_connections)
        await gateway.drop_connection()
    await wait_until(lambda: len(gateway.connections) == 3)
    await finish(client, task)
    # every drop followed a stable connection, so no delay ever doubled
    assert delays == [RECONNECT_MIN, RECONNECT_MIN]


async def test_connections_that_drop_right_away_keep_backing_off(gateway, backend):
    delays = []

    async def recording_sleep(delay):
        delays.append(delay)
        await asyncio.sleep(0)

    config = make_config(gateway.url, [ollama_backend(backend)], reconnect_min=RECONNECT_MIN,
                         reconnect_max=RECONNECT_MAX)
    client = TunnelClient(config, sleep=recording_sleep, jitter=lambda: FULL_JITTER, clock=lambda: FROZEN_TIME)
    task = await start(client)
    for expected_connections in (1, 2):
        await wait_until(lambda: len(gateway.connections) == expected_connections)
        await gateway.drop_connection()
    await wait_until(lambda: len(gateway.connections) == 3)
    await finish(client, task)
    # authenticated, but dropped at once each time: a gateway in trouble is not hammered
    assert delays == EXPECTED_DELAYS[:2]


async def test_stop_is_not_delayed_by_a_pending_reconnect(backend):
    async with FakeGateway(close_code_after_auth=protocol.CLOSE_REPLACED) as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))  # real sleep
        task = await start(client)
        await wait_until(lambda: gateway.auth_frames)
        await asyncio.sleep(QUIET_SECONDS)  # now waiting REPLACED_RETRY_SECONDS
        await finish(client, task)


async def test_unreachable_gateway_is_a_recoverable_loss(backend):
    client = TunnelClient(make_config('ws://127.0.0.1:9/ws/', [ollama_backend(backend)]))
    with pytest.raises(GatewayConnectionLost) as lost:
        await client.run_once()
    assert not lost.value.authenticated


@pytest.mark.parametrize('reason', ['invalid_token', 'missing_scope', 'unknown_instance', 'unsupported_protocol'])
async def test_rejected_credentials_stop_the_client(backend, reason):
    async with FakeGateway(auth_answer={'type': protocol.FRAME_AUTH_FAILED, 'reason': reason}) as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
        with pytest.raises(FatalGatewayError, match=reason):
            await asyncio.wait_for(client.run_forever(), STARTUP_TIMEOUT_SECONDS)
        assert len(gateway.auth_frames) == 1  # no retry storm with bad credentials


async def test_being_replaced_by_another_client_retries_after_a_long_pause(backend, caplog):
    delays = []

    async def recording_sleep(delay):
        delays.append(delay)
        await asyncio.sleep(0)

    async with FakeGateway(close_code_after_auth=protocol.CLOSE_REPLACED) as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]), sleep=recording_sleep)
        task = await start(client)
        await wait_until(lambda: len(gateway.auth_frames) >= 2)
        await finish(client, task)
    assert delays[0] == REPLACED_RETRY_SECONDS
    assert 'Another client connected' in caplog.text


async def test_revoked_credentials_stop_the_client(backend):
    async with FakeGateway(close_code_after_auth=protocol.CLOSE_REVOKED) as gateway:
        client = TunnelClient(make_config(gateway.url, [ollama_backend(backend)]))
        with pytest.raises(FatalGatewayError):
            await asyncio.wait_for(client.run_forever(), STARTUP_TIMEOUT_SECONDS)
        assert len(gateway.auth_frames) == 1


async def test_stop_ends_run_forever_and_cancels_requests_in_flight(gateway, backend, running_client):
    backend.hold_stream.clear()
    await gateway.send_request('req-1', CHAT)
    await gateway.next_frame(protocol.FRAME_RESPONSE_CHUNK)
    # the running_client fixture stops the client; it must not hang on the stalled request
