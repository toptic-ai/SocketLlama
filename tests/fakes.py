"""Fakes for the client tests: a gateway (WebSocket server) and an OpenAI-compatible backend (HTTP server)."""
import asyncio
import json

import websockets

from socketllama import protocol
from socketllama.setup_wizard import Discovered

LOCALHOST = '127.0.0.1'
EPHEMERAL_PORT = 0
TOKEN = 'tt_live_test-token'
INSTANCE = 'tester-home'
SIGNING_SECRET = 'test-signing-secret'
NONCE = 'gateway-nonce-1'
BACKEND_API_KEY = 'sk-local-backend-key'
MODEL = 'llama3.1:8b'
EMBEDDING_MODEL = 'nomic-embed-text'
CONTEXT_LENGTH = 131072
STREAM_PARTS = ['Hel', 'lo ', 'there']
HTTP_OK = 200
HTTP_SERVER_ERROR = 500
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
JSON_CONTENT_TYPE = 'application/json'
RECEIVE_TIMEOUT_SECONDS = 5.0
POLL_SECONDS = 0.01


class FakeGateway:
    """Speaks the gateway side of the tunnel protocol and records every frame of every connection."""

    def __init__(self, auth_answer=None, close_code_after_auth=None, answer_pings=True, nonce=NONCE,
                 capabilities=None):
        """``capabilities``: announced in auth_ok (None: none, like a gateway without capabilities)."""
        self.auth_answer = auth_answer or {'type': protocol.FRAME_AUTH_OK, 'instance': INSTANCE,
                                           'protocol': protocol.PROTOCOL_VERSION,
                                           **({'capabilities': capabilities} if capabilities is not None else {})}
        self.close_code_after_auth = close_code_after_auth
        self.answer_pings = answer_pings
        self.nonce = nonce
        # nonce sent on each accepted connection, in connection order (every connection gets its own)
        self.connection_nonces = []
        self._handshakes = 0
        self.connections = []
        self.frames = asyncio.Queue()
        self.auth_frames = []
        self._server = None
        self._sockets = []

    async def __aenter__(self):
        self._server = await websockets.serve(self._handle, LOCALHOST, EPHEMERAL_PORT)
        return self

    async def __aexit__(self, *exc_info):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f'ws://{LOCALHOST}:{port}/ws/socketllama/'

    def _nonce_for_next_connection(self):
        self._handshakes += 1
        if not self.nonce:
            return None
        return self.nonce if self._handshakes == 1 else f'{self.nonce}-{self._handshakes}'

    async def _handle(self, socket):
        self._sockets.append(socket)
        nonce = self._nonce_for_next_connection()
        await socket.send(json.dumps({'type': protocol.FRAME_AUTH_REQUIRED, 'protocol': protocol.PROTOCOL_VERSION,
                                      **({'nonce': nonce} if nonce else {})}))
        self.auth_frames.append(json.loads(await socket.recv()))
        await socket.send(json.dumps(self.auth_answer))
        if self.auth_answer['type'] != protocol.FRAME_AUTH_OK:
            await socket.close(code=protocol.CLOSE_FORBIDDEN)
            return
        if self.close_code_after_auth is not None:
            await socket.close(code=self.close_code_after_auth)
            return
        self.connections.append(socket)
        self.connection_nonces.append(nonce)
        try:
            async for raw in socket:
                frame = json.loads(raw)
                if frame.get('type') == protocol.FRAME_PING:
                    if self.answer_pings:
                        await socket.send(json.dumps({'type': protocol.FRAME_PONG}))
                    continue
                await self.frames.put(frame)
        except websockets.ConnectionClosed:
            pass

    async def send(self, frame: dict, connection: int = -1):
        await self.connections[connection].send(json.dumps(frame))

    def request_frame(self, request_id: str, payload, backend: str = 'ollama', method: str = 'POST',
                      path: str = '/chat/completions', secret: str = SIGNING_SECRET, connection: int = -1,
                      **overrides) -> dict:
        """A request frame signed for one connection; ``overrides`` replace frame fields after signing."""
        raw = json.dumps(payload).encode()
        nonce = self.connection_nonces[connection]
        frame = {'type': protocol.FRAME_REQUEST, 'id': request_id, 'backend': backend, 'method': method,
                 'path': path, 'body': protocol.encode_body(raw),
                 'signature': protocol.sign_request(secret, nonce, request_id, backend, method, path, raw)}
        frame.update(overrides)
        return frame

    async def send_request(self, request_id: str, payload, **options):
        """Send a correctly signed request frame on the latest connection (see request_frame)."""
        await self.send(self.request_frame(request_id, payload, **options))

    async def next_frame(self, frame_type: str | None = None) -> dict:
        """Next frame from the client, optionally skipping until one of ``frame_type`` arrives."""
        while True:
            frame = await asyncio.wait_for(self.frames.get(), RECEIVE_TIMEOUT_SECONDS)
            if frame_type is None or frame.get('type') == frame_type:
                return frame

    async def collect_response(self, request_id: str) -> tuple:
        """(start frame, body bytes, final frame) of one relayed request; progress frames are skipped."""
        start, body = None, b''
        while True:
            frame = await self.next_frame()
            if frame.get('id') != request_id or frame['type'] == protocol.FRAME_RESPONSE_PROGRESS:
                continue
            if frame['type'] == protocol.FRAME_RESPONSE_START:
                start = frame
            elif frame['type'] == protocol.FRAME_RESPONSE_CHUNK:
                body += protocol.decode_body(frame['data'])
            else:
                return start, body, frame

    def drain_frames(self) -> list:
        """Remove and return every frame queued so far."""
        frames = []
        while not self.frames.empty():
            frames.append(self.frames.get_nowait())
        return frames

    def drain_frames_of(self, request_id: str) -> list:
        """Remove and return the frames queued so far that belong to one request."""
        return [frame for frame in self.drain_frames() if frame.get('id') == request_id]

    async def drop_connection(self):
        await self.connections[-1].close()


class FakeBackend:
    """A minimal HTTP/1.1 server answering like Ollama's and LiteLLM's OpenAI-compatible endpoints."""

    def __init__(self, model_info_status=HTTP_OK, extra_model_info_entries=(), loaded_context_length=None,
                 modelfile_num_ctx=None, process_list_status=HTTP_OK, loaded_name=MODEL, more_models=(),
                 models_that_do_not_fit=(), loads_on_chat=False, usage=None, chat_failure=None):
        self.requests = []
        # (status, body, content type) every chat completion is answered with instead of an answer, the body
        # as the bytes to send, or as a list of the parts to send it in; None: it is answered
        self.chat_failure = chat_failure
        # cleared by tests that need such a body to stall after each of its parts but the last
        self.hold_failure_body = asyncio.Event()
        self.hold_failure_body.set()
        # how many parts of such bodies were sent so far
        self.failure_parts_sent = 0
        # cleared by tests that need Ollama not to tell what is in memory (/api/ps does not answer)
        self.hold_process_list = asyncio.Event()
        self.hold_process_list.set()
        # set by tests that need the backend to go away in the middle of a streamed answer, after its first event
        self.dies_mid_stream = False
        # Token counts a chat completion ends with ({"prompt_tokens": ..., "completion_tokens": ...}), as Ollama
        # gives them when asked to (stream_options.include_usage) and in every answer that is not streamed;
        # None: none, like a server that does not count
        self.usage = usage
        # Ollama: chat models besides MODEL that /v1/models lists
        self.more_models = list(more_models)
        # Ollama's memory, as its native endpoints change and show it: {model: the keep_alive of the
        # request that last used it (None: none given, Ollama's default applies)}. /api/generate
        # without a prompt loads a model (keep_alive 0 unloads it) unless it is one of
        # ``models_that_do_not_fit``; a chat completion loads its model only with ``loads_on_chat``
        # and then, giving no keep_alive, leaves it with the default
        self.in_memory = {}
        self.models_that_do_not_fit = list(models_that_do_not_fit)
        self.loads_on_chat = loads_on_chat
        # cleared by tests that need loading a model to take its time
        self.hold_generate = asyncio.Event()
        self.hold_generate.set()
        # Ollama: the name /api/ps lists MODEL under while it is loaded
        self.loaded_name = loaded_name
        self.model_info_status = model_info_status
        # Ollama: context window of MODEL while loaded (/api/ps; None = not loaded), num_ctx of its
        # Modelfile (/api/show parameters; None = not set), and the status of /api/ps (old versions: 404)
        self.loaded_context_length = loaded_context_length
        self.modelfile_num_ctx = modelfile_num_ctx
        self.process_list_status = process_list_status
        # appended to LiteLLM's /model/info answer, e.g. malformed entries
        self.extra_model_info_entries = list(extra_model_info_entries)
        # set by tests that need the stream to stall after its first event
        self.hold_stream = asyncio.Event()
        self.hold_stream.set()
        # cleared by tests that need the model list to answer late (a backend busy loading a model)
        self.hold_discovery = asyncio.Event()
        self.hold_discovery.set()
        # set by tests that need a chat completion to wait before its response headers (a model loading)
        self.hold_headers = asyncio.Event()
        self.hold_headers.set()
        self._server = None

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._handle, LOCALHOST, EPHEMERAL_PORT)
        return self

    async def __aexit__(self, *exc_info):
        self._server.close()
        self.hold_stream.set()
        self.hold_headers.set()
        self.hold_discovery.set()
        self.hold_generate.set()
        self.hold_failure_body.set()
        self.hold_process_list.set()
        await self._server.wait_closed()

    @property
    def root_url(self) -> str:
        port = self._server.sockets[0].getsockname()[1]
        return f'http://{LOCALHOST}:{port}'

    @property
    def base_url(self) -> str:
        return f'{self.root_url}/v1'

    async def _handle(self, reader, writer):
        try:
            request_line = (await reader.readline()).decode()
            if not request_line:
                return
            method, path, _ = request_line.split(' ', 2)
            headers = {}
            while (line := (await reader.readline()).decode().strip()):
                name, value = line.split(':', 1)
                headers[name.strip().lower()] = value.strip()
            body = await reader.readexactly(int(headers.get('content-length', 0)))
            payload = json.loads(body) if body else None
            self.requests.append({'method': method, 'path': path, 'headers': headers, 'json': payload})
            await self._route(method, path, payload, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # the client cancelled the request
        finally:
            writer.close()

    async def _route(self, method, path, payload, writer):
        if (method, path) == ('GET', '/v1/models'):
            await self.hold_discovery.wait()
            await self._json(writer, {'object': 'list', 'data': [{'id': name} for name in self.model_names]})
        elif (method, path) == ('POST', '/api/generate'):
            await self.hold_generate.wait()
            await self._load(writer, payload)
        elif (method, path) == ('POST', '/api/show'):
            is_chat = payload['model'] != EMBEDDING_MODEL
            parameters = 'stop                           "<|eot_id|>"'
            if self.modelfile_num_ctx is not None:
                parameters += f'\nnum_ctx                        {self.modelfile_num_ctx}'
            await self._json(writer, {
                'capabilities': ['completion', 'tools'] if is_chat else ['embedding'],
                'parameters': parameters,
                'model_info': {'general.architecture': 'llama', 'llama.context_length': CONTEXT_LENGTH}})
        elif (method, path) == ('GET', '/api/ps'):
            await self.hold_process_list.wait()
            if self.process_list_status != HTTP_OK:
                await self._json(writer, {'error': 'not found'}, status=self.process_list_status)
                return
            loaded = [] if self.loaded_context_length is None else [
                {'name': self.loaded_name, 'model': self.loaded_name, 'context_length': self.loaded_context_length}]
            loaded += [{'name': name, 'model': name} for name in self.in_memory]
            await self._json(writer, {'models': loaded})
        elif (method, path) == ('GET', '/v1/model/info'):
            if self.model_info_status != HTTP_OK:
                await self._json(writer, {'error': 'forbidden'}, status=self.model_info_status)
                return
            await self._json(writer, {'data': [
                {'model_name': 'gpt-proxy', 'litellm_params': {'model': 'openai/gpt-x'},
                 'model_info': {'mode': 'chat', 'max_input_tokens': CONTEXT_LENGTH, 'max_output_tokens': 4096,
                                'input_cost_per_token': 2e-06, 'output_cost_per_token': 8e-06,
                                'supports_reasoning': True}},
                {'model_name': 'embedder', 'litellm_params': {}, 'model_info': {'mode': 'embedding'}},
                *self.extra_model_info_entries,
            ]})
        elif (method, path) == ('POST', '/v1/chat/completions'):
            await self.hold_headers.wait()
            if self.chat_failure is not None:
                await self._fail(writer, *self.chat_failure)
                return
            if payload.get('model') == 'missing':
                # as Ollama's OpenAI-compatible endpoint answers it
                await self._json(writer, {'error': {'message': 'model not found', 'type': 'not_found_error'}},
                                 status=HTTP_NOT_FOUND)
                return
            if self.loads_on_chat:
                self.in_memory[payload.get('model')] = None
            if payload.get('stream'):
                await self._stream(writer)
            else:
                await self._json(writer, {'choices': [{'message': {'role': 'assistant',
                                                                   'content': ''.join(STREAM_PARTS)}}],
                                          **({'usage': self.usage} if self.usage is not None else {})})
        else:
            await self._json(writer, {'error': 'not found'}, status=HTTP_NOT_FOUND)

    @property
    def model_names(self) -> list:
        return [MODEL, EMBEDDING_MODEL, *self.more_models]

    async def _load(self, writer, payload):
        """Ollama's /api/generate without a prompt: load the model, or unload it with keep_alive 0."""
        model = payload.get('model')
        if model in self.models_that_do_not_fit:
            await self._json(writer, {'error': 'model requires more system memory (12.3 GiB) than is available '
                                               '(7.9 GiB)'}, status=HTTP_SERVER_ERROR)
        elif model not in self.model_names:
            await self._json(writer, {'error': f"model '{model}' not found"}, status=HTTP_NOT_FOUND)
        else:
            if payload.get('keep_alive') == 0:
                self.in_memory.pop(model, None)
            else:
                self.in_memory[model] = payload.get('keep_alive')
            await self._json(writer, {'model': model, 'response': '', 'done': True, 'done_reason': 'load'})

    async def _json(self, writer, payload, status=HTTP_OK):
        await self._raw(writer, status, json.dumps(payload).encode(), JSON_CONTENT_TYPE)

    @staticmethod
    async def _raw(writer, status: int, body: bytes, content_type: str, body_length: int | None = None):
        """``body_length``: the length announced, when ``body`` is only the first part of what follows."""
        writer.write(f'HTTP/1.1 {status} X\r\ncontent-type: {content_type}\r\n'
                     f'content-length: {len(body) if body_length is None else body_length}\r\n'
                     f'connection: close\r\n\r\n'.encode() + body)
        await writer.drain()

    async def _fail(self, writer, status: int, body, content_type: str):
        """Answer with an error whose body comes whole, or in the parts given (see hold_failure_body)."""
        first, *more = [body] if isinstance(body, bytes) else body
        await self._raw(writer, status, first, content_type, body_length=len(first) + sum(map(len, more)))
        self.failure_parts_sent += 1
        for part in more:
            await self.hold_failure_body.wait()
            writer.write(part)
            await writer.drain()
            self.failure_parts_sent += 1

    async def _stream(self, writer):
        writer.write(b'HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n'
                     b'transfer-encoding: chunked\r\nconnection: close\r\n\r\n')
        for part in STREAM_PARTS:
            event = f'data: {json.dumps({"choices": [{"delta": {"content": part}}]})}\n\n'.encode()
            writer.write(f'{len(event):x}\r\n'.encode() + event + b'\r\n')
            await writer.drain()
            await self.hold_stream.wait()
            if self.dies_mid_stream:
                return  # the connection is closed without the end of the body
        if self.usage is not None:
            event = f'data: {json.dumps({"choices": [], "usage": self.usage})}\n\n'.encode()
            writer.write(f'{len(event):x}\r\n'.encode() + event + b'\r\n')
        done = b'data: [DONE]\n\n'
        writer.write(f'{len(done):x}\r\n'.encode() + done + b'\r\n0\r\n\r\n')
        await writer.drain()


class ManualTimeouts:
    """Stands in for asyncio.timeout where the client bounds a wait (the ``timeout`` of TunnelClient, Relay and
    ModelMemory).

    No wait runs out by itself: a test says when one does (run_out), and reads which waits are bounded and
    by how many seconds (running, asked). So the seconds are the real ones of the configuration, and no
    test depends on how fast the machine is.
    """

    def __init__(self) -> None:
        # the seconds of every wait bounded so far, in the order they began
        self.asked: list[float] = []
        self._running: list[_ManualTimeout] = []

    def __call__(self, seconds: float) -> '_ManualTimeout':
        self.asked.append(seconds)
        return _ManualTimeout(seconds, self._running)

    def running(self) -> list[float]:
        """The seconds of the waits that are bounded right now."""
        return [timeout.seconds for timeout in self._running]

    async def run_out(self, seconds: float) -> None:
        """Let the wait bounded by ``seconds`` run out, once there is one (every one, if there are several)."""
        async with asyncio.timeout(RECEIVE_TIMEOUT_SECONDS):
            while seconds not in self.running():
                await asyncio.sleep(POLL_SECONDS)
        for timeout in [timeout for timeout in self._running if timeout.seconds == seconds]:
            timeout.run_out()


class _ManualTimeout:
    def __init__(self, seconds: float, running: list) -> None:
        self.seconds = seconds
        self._running = running
        # never runs out by itself
        self._timeout = asyncio.timeout(None)

    async def __aenter__(self) -> '_ManualTimeout':
        await self._timeout.__aenter__()
        self._running.append(self)
        return self

    async def __aexit__(self, *exc_info) -> bool | None:
        self._running.remove(self)
        return await self._timeout.__aexit__(*exc_info)

    def run_out(self) -> None:
        self._timeout.reschedule(asyncio.get_running_loop().time())


def sse_text(body: bytes) -> str:
    """Concatenated content deltas of an SSE chat-completion body."""
    text = ''
    for line in body.decode().splitlines():
        if line.startswith('data: ') and line != 'data: [DONE]':
            text += json.loads(line[len('data: '):])['choices'][0]['delta'].get('content', '')
    return text


class ScriptedConsole:
    """Stands in for the terminal of ``socketllama setup``: answers from a script, records what is shown.

    An answer that is an exception is raised instead (Ctrl-C: KeyboardInterrupt()); a script that
    runs out ends the input like Ctrl-D does (EOFError).
    """

    def __init__(self, answers):
        self.unused_answers = list(answers)
        # everything shown, prompts included
        self.shown = []
        # prompts of the answers read without echo
        self.secret_prompts = []
        # what a person would see: lines shown, and prompts followed by what was typed (unless not echoed)
        self.transcript = []

    def say(self, text: str = ''):
        self.shown.append(text)
        self.transcript.append(text)

    def ask(self, prompt: str) -> str:
        answer = self._next_answer(prompt)
        self.transcript.append(f'{prompt}{answer}')
        return answer

    def ask_secret(self, prompt: str) -> str:
        self.secret_prompts.append(prompt)
        answer = self._next_answer(prompt)
        self.transcript.append(prompt)
        return answer

    def _next_answer(self, prompt: str) -> str:
        self.shown.append(prompt)
        if not self.unused_answers:
            raise EOFError
        answer = self.unused_answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    @property
    def output(self) -> str:
        return '\n'.join(self.shown)


class ScriptedDiscovery:
    """Stands in for asking backends for their models: per address, the answers of successive attempts.

    An answer is a list of model names, a Discovered (names with notes) or an exception to raise;
    the last one repeats. An address without answers fails the test: setup must only ask the
    addresses it was given.
    """

    def __init__(self, answers_by_address: dict):
        self._answers = {address: list(answers) for address, answers in answers_by_address.items()}
        self.asked = []

    def __call__(self, backend) -> list:
        self.asked.append(backend)
        answers = self._answers[backend.base_url]
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer if isinstance(answer, Discovered) else Discovered(names=list(answer))
