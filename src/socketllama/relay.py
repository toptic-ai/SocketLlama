"""Executes the gateway's request frames against the configured backends.

Security boundary of the client: a frame is only executed when it is signed with the
instance's secret, names a configured backend, uses an allow-listed method and path and
(if the backend restricts models) asks for an allowed model. The gateway never supplies a
host, a header or a credential.
"""
import asyncio
import contextlib
import itertools
import json
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

import httpx
from websockets.exceptions import ConnectionClosed

from . import events, protocol
from .answer_watch import AnswerFacts, AnswerWatcher
from .backends import auth_headers, ollama_loaded_models, ollama_model_reference
from .config import KIND_OLLAMA, Backend, Config
from .models_memory import ModelsMemoryReport
from .ollama_memory import (
    OLLAMA_LOADING_ANSWER_MAX_BYTES, OLLAMA_LOADING_ANSWER_MAX_SECONDS, ModelMemory, says_model_is_loading,
)

logger = logging.getLogger('socketllama')

# What the gateway may ask a backend for
ALLOWED_REQUESTS = frozenset({('POST', '/chat/completions')})
# Response headers passed back (everything else stays local; the gateway accepts no others)
FORWARDED_RESPONSE_HEADERS = ('content-type',)
JSON_CONTENT_TYPE = 'application/json'
# The body is relayed as raw bytes without content-encoding, so the backend must not compress it
UNCOMPRESSED_ENCODING = 'identity'
# Request ids remembered to refuse replays within a connection
REPLAY_WINDOW_SIZE = 4096
# How long a request waits to learn from Ollama which models are in memory, before it goes to the backend, all
# in all: a local question Ollama answers at once. One that does not is not waited for: the progress then claims
# nothing.
MEMORY_CHECK_TIMEOUT_SECONDS = 5.0
# What the gateway and the owner are told when relaying failed for a reason the client did not expect (a bug)
CLIENT_FAILED_MESSAGE = 'The client failed to relay the request'


class RelayError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class _InFlight:
    """A request being relayed: its number in the owner's terminal (see events), when it started
    (the relay's clock), the event the gateway's pause clears and its resume sets, and how often it
    was paused. Then how it ended, for its end line: unless the relay says otherwise, it was
    cancelled (also before it ever ran), and why."""
    number: int
    started_at: float
    resume: asyncio.Event = field(default_factory=asyncio.Event)
    pauses: int = 0
    task: asyncio.Task | None = None
    outcome: str = events.OUTCOME_CANCELLED
    detail: str = events.STOPPED_BY_GATEWAY
    answer: AnswerFacts | None = None
    # What failed in a way the client did not expect (a bug), shown with its traceback
    failure: BaseException | None = None
    # Ollama did not list the request's model among those in memory before the request went to it (see
    # _report_progress)
    model_not_in_memory: bool = False


class Relay:
    """One per gateway connection. ``send`` is an async callable taking a frame dict.

    ``nonce`` is the one the gateway sent when this connection started; request frames are
    signed over it. ``reports_progress``: the gateway announced the response.progress capability.
    ``model_loading_timeout``: for how many seconds the gateway waits for the start of an answer
    while a model loads (None: it did not say); the client then waits at least as long (see
    _begin). ``timeout`` bounds each wait of relaying: called with seconds, it gives the context
    that raises TimeoutError when what runs in it takes longer (asyncio.timeout).
    ``memory`` is told which model answered, once its request is over, and which model Ollama was
    still loading for a request that was cancelled (see ollama_memory).
    ``memory_report`` (gateway with the models.memory capability only) is asked to check what is
    in memory once an Ollama backend's answer is over. ``numbers`` gives each request relayed its
    number in the owner's terminal (shared by the relays of one run); ``clock`` times the answers.
    """

    def __init__(self, config: Config, http: httpx.AsyncClient, send, nonce: str, reports_progress: bool = False,
                 memory: ModelMemory | None = None, memory_report: ModelsMemoryReport | None = None,
                 numbers: Iterator[int] | None = None, clock: Callable[[], float] = time.monotonic,
                 model_loading_timeout: float | None = None,
                 timeout: Callable[[float], AbstractAsyncContextManager] = asyncio.timeout):
        self._config = config
        self._memory = memory
        self._memory_report = memory_report
        self._http = http
        self._send = send
        self._nonce = nonce
        self._reports_progress = reports_progress
        self._model_loading_timeout = model_loading_timeout
        self._timeout = timeout
        self._numbers = itertools.count(events.FIRST_REQUEST_NUMBER) if numbers is None else numbers
        self._clock = clock
        self._in_flight: dict[str, _InFlight] = {}
        # A request closed before Ollama began its answer may have the client finish a load (not once it stops)
        self._finishes_loads = True
        self._seen_ids: OrderedDict = OrderedDict()

    # ----- frames from the gateway -----

    async def handle_request(self, frame: dict, frame_size: int):
        """Check the frame, then start relaying it; a refused frame is answered and takes no slot."""
        request_id = frame.get('id')
        if not isinstance(request_id, str) or not request_id:
            logger.warning('Ignoring request frame without id')
            return
        try:
            if frame_size > protocol.MAX_REQUEST_FRAME_BYTES:
                raise RelayError(protocol.ERROR_REQUEST_TOO_LARGE,
                                 f'Request frame of {frame_size} bytes exceeds {protocol.MAX_REQUEST_FRAME_BYTES}')
            backend, method, path, body, payload = self._validate(request_id, frame)
        except RelayError as ex:
            logger.warning(events.request_refused(ex.code, str(ex)))
            await self._answer_error(request_id, ex)
            return
        model = payload.get('model')
        relayed = _InFlight(number=next(self._numbers), started_at=self._clock())
        relayed.resume.set()
        logger.info(events.prompt_received(relayed.number, events.requester(frame.get(protocol.REQUEST_USER_FIELD)),
                                           backend.name, model, events.prompt_size(payload)))
        self._in_flight[request_id] = relayed
        relayed.task = asyncio.create_task(self._run(request_id, relayed, backend, method, path, body, model))
        relayed.task.add_done_callback(lambda _task: self._finished(request_id, relayed))

    def handle_cancel(self, frame: dict):
        relayed = self._in_flight.get(frame.get('id'))
        if relayed is not None:
            relayed.task.cancel()

    def handle_pause(self, frame: dict):
        relayed = self._in_flight.get(frame.get('id'))
        if relayed is not None and relayed.resume.is_set():
            relayed.resume.clear()
            relayed.pauses += 1
            # Backpressure may pause a long answer many times: the first pause and resume say it
            logger.log(logging.INFO if relayed.pauses == 1 else logging.DEBUG, events.paused(relayed.number))

    def handle_resume(self, frame: dict):
        relayed = self._in_flight.get(frame.get('id'))
        if relayed is not None and not relayed.resume.is_set():
            relayed.resume.set()
            logger.log(logging.INFO if relayed.pauses == 1 else logging.DEBUG, events.resumed(relayed.number))

    async def cancel_all(self, client_stopping: bool = False):
        """The connection ended: every request still being relayed is cancelled. ``client_stopping``:
        because the client stops, which then loads nothing on for them."""
        if client_stopping:
            self._finishes_loads = False
        in_flight = list(self._in_flight.values())
        for relayed in in_flight:
            relayed.detail = events.CONNECTION_ENDED
            relayed.task.cancel()
        await asyncio.gather(*(relayed.task for relayed in in_flight), return_exceptions=True)

    def in_flight(self) -> int:
        return len(self._in_flight)

    def _finished(self, request_id: str, relayed: _InFlight):
        """The request is over, however it ended: its slot is free, and its end line is shown."""
        self._in_flight.pop(request_id, None)
        line = events.answer_ended(relayed.number, relayed.outcome, self._clock() - relayed.started_at,
                                   relayed.answer.usage if relayed.answer is not None else None, relayed.detail)
        if relayed.failure is not None:
            logger.error(line, exc_info=relayed.failure)
        elif relayed.outcome == events.OUTCOME_ERROR:
            logger.warning(line)
            if relayed.answer is not None and relayed.answer.error:
                logger.debug(events.backend_said(relayed.number, relayed.answer.error))
        else:
            logger.info(line)

    # ----- execution -----

    async def _run(self, request_id: str, relayed: _InFlight, backend: Backend, method: str, path: str, body: bytes,
                   model):
        """Relay the request and note how its answer ended (shown by _finished; cancelled unless noted)."""
        try:
            relayed.answer = await self._forward(request_id, relayed, backend, method, path, body, model)
        except ConnectionClosed:
            # The connection went while the answer was sent: nobody is left to tell
            relayed.detail = events.CONNECTION_ENDED
            return
        except RelayError as ex:
            relayed.outcome, relayed.detail = events.OUTCOME_ERROR, f'{ex.code}, {ex}'
            await self._answer_error(request_id, ex)
            return
        except Exception as ex:
            # A bug here must not end the request silently: the gateway would wait for it until its deadline
            relayed.outcome, relayed.detail, relayed.failure = events.OUTCOME_ERROR, CLIENT_FAILED_MESSAGE, ex
            await self._answer_error(request_id, RelayError(protocol.ERROR_CLIENT_INTERNAL, CLIENT_FAILED_MESSAGE))
            return
        if relayed.answer.succeeded:
            relayed.outcome, relayed.detail = events.OUTCOME_DONE, None
        else:
            relayed.outcome, relayed.detail = events.OUTCOME_ERROR, _failure_in_words(backend, relayed.answer)

    async def _answer_error(self, request_id: str, error: RelayError):
        """Tell the gateway; a connection that is gone has nobody to tell."""
        try:
            await self._send({'type': protocol.FRAME_RESPONSE_ERROR, 'id': request_id,
                              'code': error.code, 'message': str(error)})
        except ConnectionClosed:
            logger.debug('Request %s: the connection ended before its error could be told', request_id)

    def _validate(self, request_id: str, frame: dict):
        try:
            body = protocol.decode_body(frame.get('body') or '')
        except (ValueError, TypeError):
            raise RelayError(protocol.ERROR_INVALID_REQUEST, 'Request body is not valid base64')
        if not protocol.signature_is_valid(self._config.signing_secret, self._nonce, frame, body):
            raise RelayError(protocol.ERROR_BAD_SIGNATURE, 'Request signature does not match the signing secret')
        if request_id in self._seen_ids:
            raise RelayError(protocol.ERROR_REPLAYED_REQUEST, 'Request id was already used on this connection')
        self._seen_ids[request_id] = True
        if len(self._seen_ids) > REPLAY_WINDOW_SIZE:
            self._seen_ids.popitem(last=False)

        backend = self._config.backends.get(frame.get('backend'))
        if backend is None:
            raise RelayError(protocol.ERROR_UNKNOWN_BACKEND, f'No backend named {frame.get("backend")!r} is configured')
        method, path = frame.get('method'), frame.get('path')
        if (method, path) not in ALLOWED_REQUESTS:
            raise RelayError(protocol.ERROR_PATH_NOT_ALLOWED, f'{method} {path} is not allowed')
        try:
            payload = json.loads(body)
        except ValueError:
            raise RelayError(protocol.ERROR_INVALID_REQUEST, 'Request body is not JSON')
        if not isinstance(payload, dict):
            raise RelayError(protocol.ERROR_INVALID_REQUEST, 'Request body must be a JSON object')
        if not backend.allows_model(payload.get('model')):
            raise RelayError(protocol.ERROR_MODEL_NOT_ALLOWED,
                             f'Model {payload.get("model")!r} is not enabled for backend {backend.name}')
        if len(self._in_flight) >= self._config.max_concurrent_requests:
            raise RelayError(protocol.ERROR_CLIENT_BUSY, 'Too many requests in flight on this client')
        return backend, method, path, body, payload

    async def _forward(self, request_id: str, relayed: _InFlight, backend: Backend, method: str, path: str,
                       body: bytes, model) -> AnswerFacts:
        """Relay the request and its answer; returns what was learnt from the answer on the way."""
        headers = {'content-type': JSON_CONTENT_TYPE, 'accept-encoding': UNCOMPRESSED_ENCODING,
                   **auth_headers(backend)}
        # The HTTP library has one time limit for reading, the start of an answer and each of its parts alike.
        # The two have limits of their own here (see _begin and _next_chunk), so it gets none.
        timeout = httpx.Timeout(backend.read_timeout, connect=backend.connect_timeout, read=None)
        # All of it before the backend is asked: nothing of the progress may follow the answer, nor an error frame
        if self._reports_progress:
            await self._report_progress(request_id, relayed, backend, model)
        request = self._http.build_request(method, backend.base_url + path, content=body, headers=headers,
                                           timeout=timeout)
        # Whether the backend began an answer of the model to this request: the model is loaded then
        answered = False
        try:
            async with contextlib.aclosing(await self._begin(request, relayed, backend, model)) as response:
                answered = response.is_success
                watcher = AnswerWatcher(response.status_code, response.headers.get('content-type', ''))
                async with contextlib.aclosing(_BodyReader(response.aiter_raw())) as answer:
                    # What had to be read of the answer before any of it is passed on
                    read_ahead: list[bytes] = []
                    if backend.kind == KIND_OLLAMA and response.is_server_error:
                        read_ahead, is_whole = await self._read_ahead(answer, OLLAMA_LOADING_ANSWER_MAX_BYTES,
                                                                      OLLAMA_LOADING_ANSWER_MAX_SECONDS)
                        if is_whole and says_model_is_loading(b''.join(read_ahead)):
                            # Not an answer to pass on: the gateway is told what it means
                            logger.info(events.backend_loading_model(relayed.number, backend.name))
                            raise RelayError(protocol.ERROR_BACKEND_MODEL_LOADING,
                                             f'Backend {backend.name} is still loading a model')
                    await self._send({
                        'type': protocol.FRAME_RESPONSE_START, 'id': request_id, 'status': response.status_code,
                        'headers': {name: response.headers[name] for name in FORWARDED_RESPONSE_HEADERS
                                    if name in response.headers},
                    })
                    for chunk in read_ahead:
                        await self._relay_chunk(request_id, relayed, watcher, chunk)
                    while (chunk := await self._next_chunk(answer, backend)) is not None:
                        await self._relay_chunk(request_id, relayed, watcher, chunk)
        except httpx.TimeoutException as ex:
            raise RelayError(protocol.ERROR_BACKEND_TIMEOUT, f'Backend {backend.name} timed out: {ex!r}')
        except httpx.HTTPError as ex:
            raise RelayError(protocol.ERROR_BACKEND_UNREACHABLE, f'Backend {backend.name} is unreachable: {ex!r}')
        finally:
            # The request to the backend is over, however it ended (also cancelled or broken off)
            if answered and self._memory is not None and isinstance(model, str):
                self._memory.model_answered(backend, model)
            # The model that answered is in memory now
            if answered and self._memory_report is not None and backend.kind == KIND_OLLAMA:
                self._memory_report.check()
        await self._send({'type': protocol.FRAME_RESPONSE_END, 'id': request_id})
        return watcher.result()

    async def _begin(self, request: httpx.Request, relayed: _InFlight, backend: Backend, model) -> httpx.Response:
        """Send the request and wait until the backend begins to answer: its status and headers.

        For as long as the backend's read_timeout says. For a model found not in memory, for as
        long as the gateway waits while a model loads, if that is longer
        (Backend.first_answer_timeout): giving up sooner would end a load the gateway still waits
        for. Only then: the gateway waits that long on being told model_not_in_memory and no
        longer than usual otherwise, so a longer wait of the client would buy nothing and only go
        against the time the owner gave the backend. When the wait is given up, here or because
        the request was cancelled, the request to the backend is closed, and Ollama gives up
        loading a model for a request that is gone; so the client then asks for the load itself
        (see _load_given_up).
        """
        waited = backend.first_answer_timeout(self._model_loading_timeout if relayed.model_not_in_memory else None)
        try:
            async with self._timeout(waited):
                return await self._http.send(request, stream=True)
        except TimeoutError:
            self._load_given_up(relayed, backend, model)
            raise RelayError(protocol.ERROR_BACKEND_TIMEOUT,
                             f'Backend {backend.name} did not begin to answer within {waited:g} s') from None
        except asyncio.CancelledError:
            # By the gateway, or with the connection to it
            self._load_given_up(relayed, backend, model)
            raise

    def _load_given_up(self, relayed: _InFlight, backend: Backend, model) -> None:
        """The request to the backend was closed before the backend began to answer. If its model was
        not in memory when the request came, Ollama was most likely still loading it and has given
        that up with the request: the memory component has it loaded all the same. The request is
        closed by now, so the load is asked for right after it, not before.

        Not always was Ollama still loading: it may have had the model in memory by then and been
        busy with a long prompt or an answer that is not streamed. The load asked for is then
        answered at once and does no harm. A client that stops loads nothing."""
        if relayed.model_not_in_memory and self._finishes_loads and self._memory is not None:
            self._memory.finish_loading(backend, model, relayed.number)

    async def _relay_chunk(self, request_id: str, relayed: _InFlight, watcher: AnswerWatcher, chunk: bytes):
        await relayed.resume.wait()  # backpressure: the gateway paused this request
        await self._send({'type': protocol.FRAME_RESPONSE_CHUNK, 'id': request_id,
                          'data': protocol.encode_body(chunk)})
        watcher.feed(chunk)

    async def _next_chunk(self, answer: '_BodyReader', backend: Backend) -> bytes | None:
        """The next chunk of an answer that has begun (None at its end), for which the backend has its
        read_timeout. Only for the read: what the gateway's pause holds up is not the backend's silence."""
        try:
            async with self._timeout(backend.read_timeout):
                return await answer.read()
        except TimeoutError:
            raise RelayError(protocol.ERROR_BACKEND_TIMEOUT,
                             f'Backend {backend.name} sent nothing more for {backend.read_timeout:g} s') from None

    async def _read_ahead(self, answer: '_BodyReader', limit: int, seconds: float) -> tuple[list[bytes], bool]:
        """Read ``answer`` until it ends, holds more than ``limit`` bytes or ``seconds`` have passed: the
        chunks read, as they came, and whether they are the whole body. What follows them is left to read."""
        chunks: list[bytes] = []
        size = 0
        try:
            async with self._timeout(seconds):
                while size <= limit:
                    chunk = await answer.read()
                    if chunk is None:
                        return chunks, True
                    chunks.append(chunk)
                    size += len(chunk)
        except TimeoutError:
            pass
        return chunks, False

    async def _report_progress(self, request_id: str, relayed: _InFlight, backend: Backend, model):
        """What this machine did with the request before it goes to the backend, each stage once.

        Received; and for Ollama, checked once against /api/ps, that the model is not in memory
        (so it is loaded first; noted for the request, see _load_given_up). Checked before the
        backend is asked, not meanwhile: /api/ps has no loading state and lists a model from the
        moment Ollama starts to load it, so a check that ran beside the request could find the
        very model this request made Ollama load. An unknown list of loaded models claims
        nothing. The gateway keeps showing the last stage, also to whoever opens the topic later.
        """
        await self._send_progress(request_id, protocol.PROGRESS_STAGE_RECEIVED)
        if backend.kind == KIND_OLLAMA and isinstance(model, str):
            loaded = await self._loaded_models(backend)
            if loaded is not None and ollama_model_reference(model) not in loaded:
                relayed.model_not_in_memory = True
                logger.info(events.model_not_in_memory(relayed.number, backend.name, model))
                await self._send_progress(request_id, protocol.PROGRESS_STAGE_MODEL_NOT_IN_MEMORY)

    async def _loaded_models(self, backend: Backend) -> dict | None:
        """What Ollama holds in memory, for the progress of a request; None if unknown. The request waits
        for this, so not for long, and whatever goes wrong with it only loses progress: it is logged."""
        try:
            async with self._timeout(MEMORY_CHECK_TIMEOUT_SECONDS):
                return await ollama_loaded_models(backend, self._http)
        except TimeoutError:
            logger.debug('Backend %s: no list of loaded models within %g s', backend.name, MEMORY_CHECK_TIMEOUT_SECONDS)
            return None
        except Exception:
            logger.warning('Backend %s: could not check which models are in memory', backend.name, exc_info=True)
            return None

    async def _send_progress(self, request_id: str, stage: str):
        await self._send({'type': protocol.FRAME_RESPONSE_PROGRESS, 'id': request_id, 'stage': stage})


def _failure_in_words(backend: Backend, answer: AnswerFacts) -> str:
    """Why an answer the backend gave failed: its HTTP status unless that was a success, and the kind of its
    error. Not its message: that may quote the request (shown as a detail, events.backend_said)."""
    status = '' if httpx.codes.is_success(answer.status) else f'HTTP {answer.status} '
    kind = f' ({answer.error_kind})' if answer.error_kind else ''
    return f'{status}from {backend.name}{kind}'


class _BodyReader:
    """Reads the body of an answer chunk by chunk, each read in a task of its own.

    Whoever waits for a chunk may stop waiting (a time limit, see _read_ahead) without ending the
    body: cancelling a read of the HTTP library closes its response. The read goes on instead, and
    the next ``read`` takes it up. ``aclose`` ends a read still on its way.
    """

    def __init__(self, body: AsyncIterator[bytes]) -> None:
        self._body = body
        self._reading: asyncio.Task | None = None
        self._ended = False

    async def read(self) -> bytes | None:
        """The next chunk; None at the end of the body. Raises what reading it raised."""
        if self._ended:
            return None
        if self._reading is None:
            self._reading = asyncio.create_task(self._next())
        chunk = await asyncio.shield(self._reading)
        self._reading = None
        self._ended = chunk is None
        return chunk

    async def _next(self) -> bytes | None:
        return await anext(self._body, None)

    async def aclose(self) -> None:
        if self._reading is not None:
            self._reading.cancel()
            await asyncio.gather(self._reading, return_exceptions=True)
