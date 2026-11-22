"""How long Ollama keeps models in memory, as far as the client has a say in it.

Ollama unloads a model some time after its last request (five minutes unless the server is set
up otherwise). A request can say how long instead, but only on Ollama's own endpoints: the
OpenAI-compatible one the client relays to has no such field. So the client says it with a
request of its own, POST /api/generate with the model's name, ``keep_alive`` and no prompt.
Ollama answers that at once for a model in memory and takes the time from it; for a model that
is not in memory it loads the model first.

Two uses, both per backend and both off unless configured:

``keep_loaded``    After a model answered a relayed request, it is told to stay for the
                   configured time. Every request Ollama gets sets the time anew, a relayed one
                   to Ollama's default, so this follows relayed requests that got an answer,
                   once such a request is over: whatever else is still running for the model,
                   the last word is one of these. One at a time for a model of a backend: the
                   answers that end while it is on its way are followed by one more, not by one
                   each. Only for a model /api/ps lists as in memory: the request must never load
                   one (it may have been unloaded in the meantime to make room for another).
                   Ollama may still unload a model between an answer and this request, or
                   another model may push it out afterwards.
``load_at_start``  The models listed are loaded when the client starts, one after another.

One more, for every Ollama backend and without a setting:

A load is finished  Ollama ties the loading of a model to the request that made it necessary: when
                   that request goes away before the model is in memory, Ollama gives the load up
                   and frees what it had read, and the next prompt starts from nothing. So when a
                   relayed request is closed before Ollama began its answer (the gateway cancelled
                   it, the connection to the gateway ended, or the client's own wait for the answer
                   ran out) and the model was not in memory when the request came, the client asks
                   for that load itself, with the same request as above; the next prompt then finds
                   the model in memory. At most one such load for a model of a backend, and the
                   client's own loads (these and those at start) go to Ollama one after another. Not
                   with ``keep_loaded`` set to no time at all: the model would not stay.

                   What the client knows is only that the model was not in memory when the request
                   came and that no answer had begun. Ollama may have loaded it by then and been
                   busy with a long prompt, or with an answer that is not streamed (whose first byte
                   comes when it is complete): the request is then answered at once and changes
                   nothing but the time the model stays, and the lines shown claim no more than that.

The client's own loads wait for Ollama as long as a relayed request waits for a model that was not in
memory (Backend.first_answer_timeout): the backend's read_timeout, or what the gateway said it waits
while a model loads, if that is longer. That bounds the load as a whole, as the waits of relaying are
bounded (ModelMemory's ``timeout``), and so does KEEP_LOADED_TIMEOUT_SECONDS a request that only says
how long a model stays.

None of it is part of relaying: nothing here delays or fails an answer. What goes wrong is logged.

Also here, as it is about Ollama loading a model: how to tell the error answer Ollama gives a request
while a model is loading (says_model_is_loading; the relay decides what to do about it).
"""
import asyncio
import json
import logging
import time
from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager

import httpx

from . import events
from .backends import auth_headers, describe_error, ollama_loaded_models, ollama_model_reference, ollama_root
from .config import Backend, keep_loaded_seconds
from .text import printable
from .units import SECONDS_PER_MINUTE

logger = logging.getLogger('socketllama')

OLLAMA_GENERATE_PATH = '/api/generate'
OLLAMA_KEEP_ALIVE_FIELD = 'keep_alive'
OLLAMA_ERROR_FIELD = 'error'
# How long Ollama keeps a model after its last request unless OLLAMA_KEEP_ALIVE or the request says otherwise
OLLAMA_DEFAULT_KEEP_ALIVE_MINUTES = 5
OLLAMA_DEFAULT_KEEP_ALIVE_SECONDS = OLLAMA_DEFAULT_KEEP_ALIVE_MINUTES * SECONDS_PER_MINUTE
# Models Ollama keeps in memory at once unless OLLAMA_MAX_LOADED_MODELS says otherwise: this many
# for each GPU, or this many without one. The client cannot read the server's setting.
OLLAMA_DEFAULT_MAX_LOADED_MODELS = 3
# What can go wrong with a request to Ollama: no connection, no usable answer, an address that is none
REQUEST_ERRORS = (httpx.HTTPError, httpx.InvalidURL, ValueError)
# Ollama answers at once when told how long a model in memory stays. Not the backend's read_timeout,
# which allows for a whole answer: a slow Ollama must not hold such a small request for minutes.
KEEP_LOADED_TIMEOUT_SECONDS = 15.0
# What Ollama answers a request with while a model is loading, as a server error (HTTP 500). The whole
# message, as its OpenAI-compatible endpoint gives it: "unexpected server status: llm server loading model"
OLLAMA_LOADING_MODEL_TEXT = 'llm server loading model'
# The message of an error on the OpenAI-compatible endpoint ({"error": {"message": ...}}); Ollama's own
# endpoints give the text itself ({"error": ...})
OLLAMA_ERROR_MESSAGE_FIELD = 'message'
# An error answer is read up to this size to tell whether it says that; the one it is about is far shorter
OLLAMA_LOADING_ANSWER_MAX_BYTES = 4 * 1024
# and for at most this long: an error answer that comes more slowly is passed on as it comes
OLLAMA_LOADING_ANSWER_MAX_SECONDS = 2.0


def keep_loaded_in_words(keep_loaded: str | int | float | None) -> str:
    """How long models stay in memory, to complete "they stay ...", "kept in memory ..."."""
    if keep_loaded is None:
        return 'as long as Ollama itself keeps them'
    seconds = keep_loaded_seconds(keep_loaded)
    if seconds < 0:
        return 'until Ollama stops'
    if seconds == 0:
        return 'not at all (unloaded right after an answer)'
    if isinstance(keep_loaded, str):
        return f'for {keep_loaded}'
    # Every digit of a whole number of seconds, however long ("9223372036", not "9.22337e+09")
    whole = isinstance(keep_loaded, float) and keep_loaded.is_integer()
    return f'for {int(keep_loaded) if whole else keep_loaded} seconds'


async def load_model(backend: Backend, http: httpx.AsyncClient, model: str) -> None:
    """Have Ollama load ``model`` (a no-op for one in memory) and keep it for the backend's
    keep_loaded; without one, for as long as Ollama itself would. Raises one of REQUEST_ERRORS.

    How long Ollama may take to answer is the caller's to bound, as a whole (ModelMemory's ``timeout``):
    a bound per phase of the request (httpx's read timeout) would let a slow answer go on for longer."""
    request: dict[str, object] = {'model': model}
    if backend.keep_loaded is not None:
        request[OLLAMA_KEEP_ALIVE_FIELD] = backend.keep_loaded
    response = await http.post(ollama_root(backend) + OLLAMA_GENERATE_PATH, json=request,
                               headers=auth_headers(backend),
                               timeout=httpx.Timeout(backend.read_timeout, connect=backend.connect_timeout, read=None))
    response.raise_for_status()


def failure_in_words(error: Exception) -> str:
    """Why a request to Ollama failed: Ollama's own reason if it gave one, made printable."""
    if isinstance(error, httpx.HTTPStatusError):
        try:
            reason = error.response.json().get(OLLAMA_ERROR_FIELD)
        except (ValueError, AttributeError):
            reason = None
        if isinstance(reason, str) and reason:
            return printable(reason)
    return describe_error(error)


def says_model_is_loading(body: bytes) -> bool:
    """Whether ``body``, that of an error answer of Ollama, says that a model is still loading: a JSON
    object whose error holds OLLAMA_LOADING_MODEL_TEXT. Anything else is some other error."""
    try:
        answer = json.loads(body)
    except (ValueError, RecursionError):
        return False
    error = answer.get(OLLAMA_ERROR_FIELD) if isinstance(answer, dict) else None
    message = error.get(OLLAMA_ERROR_MESSAGE_FIELD) if isinstance(error, dict) else error
    return isinstance(message, str) and OLLAMA_LOADING_MODEL_TEXT in message


# A model of a backend: the backend's name and the model's name as Ollama resolves it
ModelKey = tuple[str, str]


class ModelMemory:
    """One per client: what it tells Ollama about models in memory. See the module's description."""

    def __init__(self, http: httpx.AsyncClient, clock: Callable[[], float] = time.monotonic,
                 timeout: Callable[[float], AbstractAsyncContextManager] = asyncio.timeout) -> None:
        """``timeout``: what bounds each request to Ollama as a whole (asyncio.timeout; a test gives one
        that runs out when it says so)."""
        self._http = http
        self._clock = clock
        self._timeout = timeout
        # What is on its way to Ollama after answers: at most one task for a model of a backend
        self._telling: dict[ModelKey, asyncio.Task] = {}
        # Models that answered again while being told: each is told once more when that is over
        self._tell_again: set[ModelKey] = set()
        # Names of the backends whose last attempt failed: a failure is logged when it starts, not each time
        self._failing: set[str] = set()
        # Loads finished for requests that ended while Ollama was loading: at most one task for a model of a backend
        self._finishing: dict[ModelKey, asyncio.Task] = {}
        # The client's own loads go to Ollama one after another, those at start and those it finishes alike
        self._one_load_at_a_time = asyncio.Lock()
        # For how many seconds the gateway last connected to waits while a model loads; None: it did not say
        self._model_loading_timeout: float | None = None

    def gateway_waits_for_loading(self, seconds: float | None) -> None:
        """The gateway now connected to waits this long for the start of an answer while a model loads
        (None: it did not say): a load of the client's own gets at least as long."""
        self._model_loading_timeout = seconds

    def model_answered(self, backend: Backend, model: str) -> None:
        """``model`` answered a relayed request that is over now. Returns at once; what follows from
        it happens on its own and is never the caller's problem."""
        if backend.keep_loaded is None:
            return
        key = (backend.name, ollama_model_reference(model))
        if key in self._telling:
            # Telling Ollama now could not follow this answer: what is on its way may have gone out before it ended
            self._tell_again.add(key)
            return
        self._telling[key] = asyncio.create_task(self._tell(backend, model, key))

    async def idle(self) -> None:
        """Wait until nothing started by model_answered is on its way any more."""
        while self._telling:
            await asyncio.gather(*self._telling.values(), return_exceptions=True)

    async def close(self) -> None:
        """Drop what is still on its way, a load the client finishes included."""
        pending = [*self._telling.values(), *self._finishing.values()]
        self._tell_again.clear()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # A task cancelled before it ever ran did not get to remove itself
        self._telling.clear()
        self._finishing.clear()

    def finish_loading(self, backend: Backend, model: str, number: int) -> None:
        """Request ``number`` (see events) to ``model`` of an Ollama backend was closed before Ollama
        began its answer, and the model was not in memory when the request came: have it loaded all
        the same (see the module's description). Returns at once; the owner is shown what happens."""
        if backend.keep_loaded is not None and keep_loaded_seconds(backend.keep_loaded) == 0:
            logger.debug('Backend %s: %s is not loaded after a request without an answer, as keep_loaded lets no '
                         'model stay', backend.name, printable(model))
            return
        key = (backend.name, ollama_model_reference(model))
        if key in self._finishing:
            logger.info(events.load_already_going_on(number, backend.name, model))
            return
        logger.info(events.load_goes_on(number, backend.name, model))
        self._finishing[key] = asyncio.create_task(self._finish(backend, model, key, number))

    async def _finish(self, backend: Backend, model: str, key: ModelKey, number: int) -> None:
        # What the gateway waits may be another value by the time the load ends (a reconnect): the load runs
        # with the one of now, and so does what is said about it
        waited = backend.first_answer_timeout(self._model_loading_timeout)
        try:
            seconds = await self._load(backend, model, waited)
        except TimeoutError:
            logger.warning(events.load_failed(
                number, backend.name, model, f'Ollama did not answer within {waited:g} s; a model that takes longer '
                                             f"to load needs a larger read_timeout of the backend"))
        except REQUEST_ERRORS as ex:
            logger.warning(events.load_failed(number, backend.name, model, failure_in_words(ex)))
        except Exception:
            # Nobody awaits this task: what is not expected is logged here, or nowhere
            logger.exception('Backend %s: loading %s after a request without an answer failed unexpectedly',
                             backend.name, printable(model))
        else:
            logger.info(events.load_finished(number, backend.name, model, seconds))
        finally:
            del self._finishing[key]

    async def _load(self, backend: Backend, model: str, waited: float) -> float:
        """Have Ollama load ``model`` once no other load of the client is on its way, and wait up to
        ``waited`` seconds for it; returns how many seconds the load took. Raises one of REQUEST_ERRORS, or
        TimeoutError once the wait is over. The wait begins when the load is sent, not while another one runs."""
        async with self._one_load_at_a_time:
            started = self._clock()
            async with self._timeout(waited):
                await load_model(backend, self._http, model)
            return self._clock() - started

    async def _tell(self, backend: Backend, model: str, key: ModelKey) -> None:
        """Tell Ollama, and once more for as long as the model answered again meanwhile."""
        try:
            while True:
                await self._keep_loaded(backend, model)
                if key not in self._tell_again:
                    return
                self._tell_again.discard(key)
        except Exception:
            # Nobody awaits this task: what _keep_loaded does not expect is logged here, or nowhere
            logger.exception('Backend %s: telling Ollama to keep %s in memory failed unexpectedly. The answer was '
                             'not affected; the model stays as long as Ollama itself keeps it.', backend.name,
                             printable(model))
        finally:
            del self._telling[key]
            # Owed to an answer of this telling's time; the next telling follows answers of its own
            self._tell_again.discard(key)

    async def _keep_loaded(self, backend: Backend, model: str) -> None:
        how_long = keep_loaded_in_words(backend.keep_loaded)
        loaded = await ollama_loaded_models(backend, self._http)
        if loaded is None:
            self._failed(backend, 'Backend %s: keep_loaded has no effect for now, because Ollama does not tell what '
                                  'it has in memory (its /api/ps could not be read) and the client loads nothing '
                                  'unasked. Its models stay in memory as long as Ollama itself keeps them. This is '
                                  'logged once, until it works again.', backend.name)
            return
        if ollama_model_reference(model) not in loaded:
            logger.debug('Backend %s: %s is not in memory any more; not told to stay', backend.name, printable(model))
            return
        try:
            async with self._timeout(KEEP_LOADED_TIMEOUT_SECONDS):
                await load_model(backend, self._http, model)
        except (*REQUEST_ERRORS, TimeoutError) as ex:
            reason = f'Ollama did not answer within {KEEP_LOADED_TIMEOUT_SECONDS:g} s' \
                if isinstance(ex, TimeoutError) else failure_in_words(ex)
            self._failed(backend, 'Backend %s: could not tell Ollama to keep %s in memory %s (%s). The answer was '
                                  'not affected; the model stays as long as Ollama itself keeps it. This is logged '
                                  'once, until it works again.', backend.name, printable(model), how_long, reason)
            return
        if backend.name in self._failing:
            self._failing.discard(backend.name)
            logger.info('Backend %s: telling Ollama how long to keep models in memory works again', backend.name)
        logger.debug('Backend %s: %s told to stay in memory %s', backend.name, printable(model), how_long)

    def _failed(self, backend: Backend, message: str, *arguments: object) -> None:
        if backend.name not in self._failing:
            self._failing.add(backend.name)
            logger.warning(message, *arguments)

    async def load_at_start(self, backends: Iterable[Backend]) -> None:
        """Load the models of every backend that asks for it, one after another."""
        for backend in backends:
            if backend.load_at_start:
                await self._load_models(backend)

    async def _load_models(self, backend: Backend) -> None:
        how_long = keep_loaded_in_words(backend.keep_loaded)
        after_request = ' after their last request' if backend.keep_loaded is not None \
            and keep_loaded_seconds(backend.keep_loaded) > 0 else ''
        logger.info('Backend %s: loading %d model(s) into memory, to stay %s%s', backend.name, len(backend.models),
                    how_long, after_request)
        if len(backend.models) > OLLAMA_DEFAULT_MAX_LOADED_MODELS:
            logger.info('Backend %s: unless OLLAMA_MAX_LOADED_MODELS is set on its server, Ollama keeps at most %d '
                        'models in memory for each GPU (%d without one); beyond that, loading one unloads another',
                        backend.name, OLLAMA_DEFAULT_MAX_LOADED_MODELS, OLLAMA_DEFAULT_MAX_LOADED_MODELS)
        for model in backend.models:
            waited = backend.first_answer_timeout(self._model_loading_timeout)
            try:
                seconds = await self._load(backend, model, waited)
            except TimeoutError:
                # As a request that failed: a load that takes this long holds up those after it as well
                logger.warning('Backend %s: Ollama did not load %s within %g s; a model that takes longer to load '
                               'needs a larger read_timeout of the backend. The client goes on; Ollama loads a model '
                               'when the first prompt to it arrives.', backend.name, printable(model), waited)
                return
            except httpx.HTTPStatusError as ex:
                # Ollama answered: this model is the problem (it does not fit, it was never pulled)
                logger.warning('Backend %s: %s could not be loaded: %s. The client goes on without it; a prompt to '
                               'it makes Ollama try again.', backend.name, printable(model), failure_in_words(ex))
                continue
            except REQUEST_ERRORS as ex:
                logger.warning('Backend %s: no model could be loaded: %s. The client goes on; Ollama loads a model '
                               'when the first prompt to it arrives.', backend.name, failure_in_words(ex))
                return
            logger.info('Backend %s: %s is in memory after %.1f s', backend.name, printable(model), seconds)
        loaded = await ollama_loaded_models(backend, self._http)
        if loaded is not None:
            in_memory = [model for model in backend.models if ollama_model_reference(model) in loaded]
            logger.info('Backend %s: in memory now: %s', backend.name, ', '.join(map(printable, in_memory)) or 'none')
