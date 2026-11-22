"""Connection to the gateway: authenticate, report models, relay requests, keep alive, reconnect."""
import asyncio
import itertools
import json
import logging
import random
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

import httpx
import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI
from websockets.frames import Frame

from . import __version__, events, protocol
from .backends import build_models_report, without_unchecked_marker
from .config import Config
from .config_settings import MODEL_TABLES, ConfigSettings, ModelTable
from .models_memory import ModelsMemoryReport
from .ollama_memory import ModelMemory
from .relay import Relay

logger = logging.getLogger('socketllama')
WIRE_LOGGER_NAME = 'socketllama.wire'


class RedactFrameContent(logging.Filter):
    """websockets logs every frame at DEBUG, the auth frame's API token and every prompt included.

    Frames are reduced to their type and size, so `-v` traces the traffic without leaking it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and any(isinstance(arg, Frame) for arg in record.args):
            record.args = tuple(f'{arg.opcode.name} frame, {len(arg.data)} bytes' if isinstance(arg, Frame) else arg
                                for arg in record.args)
        return True


# Logger handed to websockets for the gateway connection (instead of its own "websockets.client")
wire_logger = logging.getLogger(WIRE_LOGGER_NAME)
wire_logger.addFilter(RedactFrameContent())

# WebSocket frame limit. Above protocol.MAX_REQUEST_FRAME_BYTES (which the gateway never exceeds)
# so that a request slightly too large is refused on its own instead of closing the connection;
# anything beyond this still closes it, to bound the memory one frame may take.
OVERSIZED_FRAME_HEADROOM = 2
MAX_GATEWAY_FRAME_BYTES = protocol.MAX_REQUEST_FRAME_BYTES * OVERSIZED_FRAME_HEADROOM
HANDSHAKE_TIMEOUT_SECONDS = 30.0
# Models pulled or configured after start show up without a restart. The gateway assumes this
# interval for how long a backend reported as not checked keeps its models: CLIENT_MODEL_REFRESH_INTERVAL_SECONDS
# of toptic.ai's gateway (ai_helper/service/socketllama_limits.py in the toptic.ai repository); change both
# together.
MODEL_REFRESH_INTERVAL_SECONDS = 300.0
BACKOFF_FACTOR = 2.0
# Each delay is scaled by a random factor in [JITTER_FLOOR, 1] so many clients do not reconnect in step
JITTER_FLOOR = 0.5
# A connection that lasted this long counts as stable: the reconnect backoff starts over after it.
# Shorter ones (accepted, then dropped at once) keep backing off, so a gateway in trouble is not hammered.
STABLE_CONNECTION_SECONDS = 60.0
# The gateway stores at most one models report per instance in this many seconds
# (SOCKETLLAMA_MODELS_REPORT_MIN_INTERVAL_SECONDS there, also across connections), timed from when
# the last one it stored arrived. A reconnect within it does not resend a report the gateway already
# acknowledged; and a report it may ignore (one within that time of the last it stored, e.g. right
# after this client restarted), the first of a connection or one asked for before the next refresh,
# is sent again if this time passes without an acknowledgement.
MODELS_REPORT_RESEND_SECONDS = 60.0
# Another client took over the instance. Not fatal (the other one may be gone soon, e.g. a laptop
# going to sleep), but retried slowly so two running clients do not keep replacing each other.
REPLACED_RETRY_SECONDS = 300.0
# Close codes after which reconnecting cannot help
FATAL_CLOSE_CODES = (protocol.CLOSE_UNAUTHENTICATED, protocol.CLOSE_FORBIDDEN, protocol.CLOSE_REVOKED)


class FatalGatewayError(Exception):
    """The gateway refused this client for a reason a reconnect will not fix."""


class GatewayConnectionLost(Exception):
    """The connection failed or dropped.

    ``authenticated``: it had been established; ``connected_seconds``: for how long;
    ``replaced``: another client took over the instance.
    """

    def __init__(self, message: str, authenticated: bool, connected_seconds: float = 0.0, replaced: bool = False):
        super().__init__(message)
        self.authenticated = authenticated
        self.connected_seconds = connected_seconds
        self.replaced = replaced


def warn_about_shared_backends(config: Config):
    for backend in config.backends.values():
        if backend.shares_all_models:
            logger.warning('Backend %s shares every model it offers (models = []), including models pulled '
                           'later; list the models to share instead', backend.name)


def backoff_delay(attempt: int, minimum: float, maximum: float, jitter: float) -> float:
    """Exponential backoff with jitter: attempt 0 waits about ``minimum``, later ones up to ``maximum``."""
    delay = min(maximum, minimum * (BACKOFF_FACTOR ** attempt))
    return delay * (JITTER_FLOOR + (1.0 - JITTER_FLOOR) * jitter)


@dataclass(frozen=True)
class LoadingWaitSaid:
    """What a gateway's auth_ok says about how long it waits while a model loads (see protocol)."""
    # As it says it; None: it says nothing usable
    seconds: int | None = None
    # Why what it says is not taken; None: nothing was ignored
    ignored: str | None = None


# What a gateway that leaves the field out says
NOTHING_SAID = LoadingWaitSaid()


class TunnelClient:

    def __init__(self, config: Config, http: httpx.AsyncClient | None = None, sleep=asyncio.sleep,
                 jitter=random.random, clock=time.monotonic,
                 timeout: Callable[[float], AbstractAsyncContextManager] = asyncio.timeout):
        """``timeout``: what bounds the waits of relaying and of the client's own requests to Ollama (see
        Relay and ModelMemory)."""
        self._config = config
        self._http = http
        self._owns_http = http is None
        self._sleep = sleep
        self._jitter = jitter
        self._clock = clock
        self._timeout = timeout
        self._stopping = asyncio.Event()
        self._closing_tasks = set()
        # last models report sent, and the one the gateway acknowledged (with when), across connections
        self._sent_report = None
        self._acked_report = None
        self._acked_at = None
        # when the last models report went out (self._clock), on whichever connection
        self._sent_at: float | None = None
        # set when what the gateway was told is out of date before the next refresh (see _until_next_report)
        self._report_requested = asyncio.Event()
        # models_ack frames received, so a report can tell whether it was acknowledged
        self._acks_received = 0
        # What Ollama is told about models in memory; set as soon as there is an HTTP client for it
        self._memory: ModelMemory | None = None
        # Set when the first models report of this run went out (see _load_models_once_reported)
        self._first_report_sent = asyncio.Event()
        # What the gateway is told about models in memory, while connected to one that takes it
        self._memory_report: ModelsMemoryReport | None = None
        # When the last of that went out (self._clock), on whichever connection: its least interval spans them
        self._memory_report_sent_at: float | None = None
        # And what of it the owner was last shown, on whichever connection: a reconnect shows nothing again
        self._memory_shown: list | None = None
        # The last models report the owner was shown, and whether the acknowledgement of the last one sent is
        # to be shown too (it is when the report was): an unchanged refresh shows neither
        self._shown_report = None
        self._shows_next_ack = False
        # What the gateway said about how long it waits while a model loads, as the owner was last shown it (see
        # _loading_wait)
        self._shown_loading_wait = NOTHING_SAID
        # Numbers of the requests relayed in this run, across connections (see events)
        self._request_numbers = itertools.count(events.FIRST_REQUEST_NUMBER)
        # The prices and descriptions of config.toml, as loaded and as written since (capabilities models.prices
        # and models.descriptions)
        self._settings = ConfigSettings(config)
        # For each table of config.toml, (backend, model) of every model a models report of this run listed that
        # a gateway taking the table acknowledged: it has seen what config.toml holds of the table for these, so a
        # value of it that it sends for one of them may be written there (see _take_model_settings)
        self._acknowledged_models: dict[ModelTable, set[tuple[str, str]]] = {table: set() for table in MODEL_TABLES}
        # The tables of config.toml the owner was told the gateway does not take (once per run each)
        self._told_not_taken: set[ModelTable] = set()
        self.connections_established = 0

    def stop(self):
        self._stopping.set()

    async def run_forever(self):
        """Stay connected until stop() is called or the gateway refuses this client for good."""
        if self._owns_http:
            self._http = httpx.AsyncClient(follow_redirects=False)
        warn_about_shared_backends(self._config)
        if self._settings.prices:
            logger.info(events.prices_in_config(self._settings.prices))
        if self._settings.descriptions:
            logger.info(events.descriptions_in_config(self._settings.descriptions))
        loading = asyncio.create_task(self._load_models_once_reported())
        attempt = 0
        try:
            while not self._stopping.is_set():
                try:
                    await self.run_once()
                except GatewayConnectionLost as ex:
                    if ex.replaced:
                        logger.warning('Another client connected for instance %s and took over. '
                                       'Retrying in %.0f s; stop one of the two clients to end this.',
                                       self._config.instance, REPLACED_RETRY_SECONDS)
                        await self._pause(REPLACED_RETRY_SECONDS)
                        continue
                    if ex.authenticated and ex.connected_seconds >= STABLE_CONNECTION_SECONDS:
                        attempt = 0
                    delay = backoff_delay(attempt, self._config.reconnect_min, self._config.reconnect_max,
                                          self._jitter())
                    attempt += 1
                    logger.warning('%s; reconnecting in %.1f s', ex, delay)
                    await self._pause(delay)
        finally:
            loading.cancel()
            await asyncio.gather(loading, return_exceptions=True)
            if self._memory is not None:
                await self._memory.close()
            if self._owns_http:
                await self._http.aclose()

    async def _load_models_once_reported(self) -> None:
        """Load the models of the backends that ask for it (load_at_start), once per run.

        Not before the gateway accepted this client: a client that is turned away must not have
        filled the machine's memory first. And only when the first models report is on its way:
        the models are then on offer at once instead of after minutes of loading, and discovery
        does not compete with the loading for Ollama's attention (a report built meanwhile could
        come out as "not checked"). A prompt that arrives while its model is still loading waits
        for that same load. What the report could not know yet, the context window Ollama gives a
        model once it is loaded, a fresh report sent once loading is over tells. A reconnect loads
        nothing again.
        """
        if not any(backend.load_at_start for backend in self._config.backends.values()):
            return
        await self._first_report_sent.wait()
        try:
            await self._model_memory().load_at_start(self._config.backends.values())
        except Exception:
            # Loading is a convenience: its failure must neither go unseen nor end the client
            logger.exception('Loading models at start failed; Ollama loads each when a prompt for it arrives')
        # The context window Ollama gave what was loaded is reported now, not at the next refresh
        self._report_requested.set()
        # and what is in memory now, as soon as its least interval allows
        if self._memory_report is not None:
            self._memory_report.check()

    def _model_memory(self) -> ModelMemory:
        if self._memory is None:
            self._memory = ModelMemory(self._http, clock=self._clock, timeout=self._timeout)
        return self._memory

    async def _pause(self, seconds: float):
        """Wait before reconnecting, but end the wait as soon as stop() is called."""
        waits = [asyncio.create_task(self._sleep(seconds)), asyncio.create_task(self._stopping.wait())]
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for wait in waits:
                wait.cancel()
            await asyncio.gather(*waits, return_exceptions=True)

    async def run_once(self):
        """One connection from handshake to close. Raises GatewayConnectionLost or FatalGatewayError."""
        authenticated = False
        connected_at = None
        try:
            async with websockets.connect(self._config.gateway_url, max_size=MAX_GATEWAY_FRAME_BYTES,
                                          ping_interval=None, open_timeout=HANDSHAKE_TIMEOUT_SECONDS,
                                          logger=wire_logger) as socket:
                nonce, capabilities, loading_wait_said = await self._authenticate(socket)
                authenticated = True
                connected_at = self._clock()
                self.connections_established += 1
                logger.info(events.connected(self._config.gateway_url, self._config.instance))
                await self._serve(socket, nonce, capabilities, self._loading_wait(loading_wait_said))
        except ConnectionClosed as ex:
            code = ex.rcvd.code if ex.rcvd is not None else None
            if code in FATAL_CLOSE_CODES:
                reason = ex.rcvd.reason if ex.rcvd is not None and ex.rcvd.reason else 'no reason given'
                raise FatalGatewayError(f'Gateway closed the connection for good (code {code}: {reason})') from ex
            raise GatewayConnectionLost(f'Connection to the gateway closed (code {code})', authenticated,
                                        self._connected_seconds(connected_at),
                                        replaced=code == protocol.CLOSE_REPLACED) from ex
        except (OSError, InvalidHandshake, InvalidURI, asyncio.TimeoutError) as ex:
            raise GatewayConnectionLost(f'Cannot reach the gateway: {ex!r}', authenticated,
                                        self._connected_seconds(connected_at)) from ex
        if self._stopping.is_set():
            return
        raise GatewayConnectionLost('Connection to the gateway ended', authenticated,
                                    self._connected_seconds(connected_at))

    def _connected_seconds(self, connected_at) -> float:
        return 0.0 if connected_at is None else self._clock() - connected_at

    # ----- handshake -----

    async def _authenticate(self, socket) -> tuple:
        """Handshake; returns the connection's nonce (request frames are signed over it), the
        capabilities the gateway announced (a frozenset of names, empty if it announced none) and
        what it says about how long it waits while a model loads (a LoadingWaitSaid)."""
        hello = await self._receive_frame(socket, HANDSHAKE_TIMEOUT_SECONDS)
        if hello.get('type') != protocol.FRAME_AUTH_REQUIRED:
            raise FatalGatewayError(f'Unexpected first frame from the gateway: {hello.get("type")!r}')
        nonce = hello.get('nonce')
        if not isinstance(nonce, str) or not nonce:
            raise FatalGatewayError('Gateway sent no nonce to sign; it speaks another protocol version')
        await self._send(socket, {
            'type': protocol.FRAME_AUTH, 'token': self._config.token, 'instance': self._config.instance,
            'protocol': protocol.PROTOCOL_VERSION, 'client_version': __version__,
            'proof': protocol.sign_auth(self._config.signing_secret, nonce, self._config.instance),
            'capabilities': list(protocol.CLIENT_CAPABILITIES),
        })
        answer = await self._receive_frame(socket, HANDSHAKE_TIMEOUT_SECONDS)
        if answer.get('type') == protocol.FRAME_AUTH_FAILED:
            raise FatalGatewayError(events.auth_refused(answer.get('reason')))
        if answer.get('type') != protocol.FRAME_AUTH_OK:
            raise FatalGatewayError(f'Unexpected answer to auth: {answer.get("type")!r}')
        announced = answer.get('capabilities')
        capabilities = frozenset(name for name in announced if isinstance(name, str)) \
            if isinstance(announced, list) else frozenset()
        try:
            said = LoadingWaitSaid(seconds=protocol.model_loading_timeout(answer))
        except ValueError as ex:
            # Not worth the connection: the client then waits as it does for a gateway that says nothing
            said = LoadingWaitSaid(ignored=str(ex))
        return nonce, capabilities, said

    def _loading_wait(self, said: LoadingWaitSaid) -> int | None:
        """For how many seconds the client takes the gateway to wait while a model loads (None: as long as
        each backend's read_timeout), given what its auth_ok ``said``. The owner is shown what it makes of
        that when the gateway says something else than what was last shown: a reconnect to the same gateway
        shows nothing again, a gateway that says nothing after one that did is said to, and one whose word
        was ignored before is not said to wait as long as read_timeout again."""
        taken = None if said.seconds is None else min(said.seconds, protocol.MAX_MODEL_LOADING_TIMEOUT_SECONDS)
        before, self._shown_loading_wait = self._shown_loading_wait, said
        if said == before:
            return taken
        if said.ignored is not None:
            logger.warning(events.loading_wait_ignored(said.ignored))
        elif taken is None:
            if before.seconds is not None:
                logger.info(events.loading_wait_not_said())
        elif said.seconds != taken:
            logger.warning(events.loading_wait_shortened(said.seconds, taken))
        else:
            logger.info(events.gateway_waits_for_loading(taken))
        return taken

    # ----- established connection -----

    async def _serve(self, socket, nonce: str, capabilities: frozenset, model_loading_timeout: int | None = None):
        """``model_loading_timeout``: for how many seconds this gateway is taken to wait while a model loads
        (None: it did not say)."""
        self._model_memory().gateway_waits_for_loading(model_loading_timeout)
        memory_report = ModelsMemoryReport(self._config.backends, self._http, lambda frame: self._send(socket, frame),
                                           self._clock, sent_at=self._memory_report_sent_at,
                                           shown=self._memory_shown) \
            if protocol.CAPABILITY_MODELS_MEMORY in capabilities else None
        self._memory_report = memory_report
        relay = Relay(self._config, self._http, lambda frame: self._send(socket, frame), nonce,
                      reports_progress=protocol.CAPABILITY_RESPONSE_PROGRESS in capabilities,
                      memory=self._model_memory(), memory_report=memory_report, numbers=self._request_numbers,
                      clock=self._clock, model_loading_timeout=model_loading_timeout, timeout=self._timeout)
        # The tables of config.toml this gateway takes and sets (capabilities models.prices, models.descriptions)
        tables = tuple(table for table in MODEL_TABLES if table.capability in capabilities)
        for table in MODEL_TABLES:
            if table not in tables and self._settings.held(table) and table not in self._told_not_taken:
                logger.warning(events.settings_not_taken(table.plural))
                self._told_not_taken.add(table)
        pong_received = asyncio.Event()
        background = [
            asyncio.create_task(self._report_models_periodically(
                socket, keeps_unchecked=protocol.CAPABILITY_MODELS_UNCHECKED in capabilities,
                memory_report=memory_report, tables=tables)),
            asyncio.create_task(self._keep_alive(socket, pong_received)),
            asyncio.create_task(self._close_when_stopped(socket)),
        ]
        if memory_report is not None:
            background.append(asyncio.create_task(memory_report.run()))
        for task in background:
            task.add_done_callback(lambda done: self._on_background_task_done(done, socket))
        try:
            async for raw in socket:
                frame = self._parse(raw)
                if frame is not None:
                    await self._dispatch(frame, len(raw), relay, pong_received, tables)
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await relay.cancel_all(client_stopping=self._stopping.is_set())
            if memory_report is not None:
                self._memory_report = None
                self._memory_report_sent_at = memory_report.sent_at
                self._memory_shown = memory_report.shown

    async def _dispatch(self, frame: dict, frame_size: int, relay: Relay, pong_received: asyncio.Event,
                        tables: tuple[ModelTable, ...] = ()):
        """``tables``: those of config.toml the gateway of this connection takes (models.prices,
        models.descriptions)."""
        frame_type = frame.get('type')
        if frame_type == protocol.FRAME_REQUEST:
            await relay.handle_request(frame, frame_size)
        elif frame_type == protocol.FRAME_CANCEL:
            relay.handle_cancel(frame)
        elif frame_type == protocol.FRAME_PAUSE:
            relay.handle_pause(frame)
        elif frame_type == protocol.FRAME_RESUME:
            relay.handle_resume(frame)
        elif frame_type == protocol.FRAME_PONG:
            pong_received.set()
        elif frame_type == protocol.FRAME_MODELS_ACK:
            self._acked_report, self._acked_at = self._sent_report, self._clock()
            self._acks_received += 1
            # What it acknowledged was sent on this connection, with what config.toml holds of these tables
            listed = {(backend['name'], model['name']) for backend in self._sent_report or []
                      for model in backend['models']}
            for table in tables:
                self._acknowledged_models[table].update(listed)
            # Shown with the report it acknowledges, and whenever the gateway changed something
            line = events.models_registered(frame, self._sent_report or [])
            if self._shows_next_ack or events.ack_changes_something(frame):
                logger.info(line)
            else:
                logger.debug(line)
            self._shows_next_ack = False
        elif frame_type == protocol.FRAME_MODELS_SETTINGS:
            await self._take_model_settings(frame, tables)
        elif frame_type == protocol.FRAME_ERROR:
            logger.warning(events.gateway_reported(frame.get('message')))
            if self._memory_report is not None:
                self._memory_report.gateway_reported_error()
        else:
            logger.debug('Ignoring unknown frame type %r', frame_type)

    async def _take_model_settings(self, frame: dict, tables: tuple[ModelTable, ...]) -> None:
        """Write the prices and descriptions of a models.settings frame into config.toml, as far as the gateway
        takes them (``tables``), and report them soon.

        Only for a model an acknowledged report of this run listed: toptic.ai decides which value is in
        force by comparing it with what config.toml holds, and has to have seen that first. Otherwise a
        line changed in the file while the client was not running could be written over before it was
        reported. A value that comes earlier is ignored; the answer to the report brings it again.
        """
        if not tables:
            logger.debug('Ignoring a %s frame: the gateway announced neither %s', protocol.FRAME_MODELS_SETTINGS,
                         ' nor '.join(table.capability for table in MODEL_TABLES))
            return
        acknowledged = {table: frozenset(self._acknowledged_models[table]) for table in tables}
        if not any(acknowledged.values()):
            logger.debug('Ignoring a %s frame: toptic.ai has not yet acknowledged a models report of this run, so it '
                         'has not seen what config.toml holds', protocol.FRAME_MODELS_SETTINGS)
            return
        try:
            # The file is read and written: not on the event loop, which keeps relaying meanwhile
            written = await asyncio.to_thread(self._settings.take_settings, frame, acknowledged)
        except Exception:
            # Writing the file is a convenience: its failure must neither go unseen nor end the connection
            logger.exception('Writing what toptic.ai set into the configuration file failed')
            return
        if written:
            # What the gateway was told of config.toml is out of date: the next report confirms the new values
            self._report_requested.set()

    def _on_background_task_done(self, task: asyncio.Task, socket):
        """A helper task that crashed must not leave the connection running unwatched: log and reconnect."""
        if task.cancelled() or task.exception() is None:
            return
        logger.error('Connection helper %s failed; reconnecting', task.get_coro().__qualname__,
                     exc_info=task.exception())
        closing = asyncio.create_task(socket.close())
        # the event loop only keeps a weak reference to tasks
        self._closing_tasks.add(closing)
        closing.add_done_callback(self._closing_tasks.discard)

    async def _report_models_periodically(self, socket, keeps_unchecked: bool,
                                          memory_report: ModelsMemoryReport | None = None,
                                          tables: tuple[ModelTable, ...] = ()):
        """``keeps_unchecked``: the gateway understands backends reported as not checked; otherwise
        such a backend is reported without models.
        ``memory_report`` learns of each report the gateway was sent or already holds.
        ``tables``: those of config.toml the gateway takes (models.prices, models.descriptions); each
        model with a line in one carries it, as the client holds it when the report is made."""
        first_of_connection = True
        # The report about to be sent is one the gateway may ignore: it is sent once more unless acknowledged
        confirming = True
        while True:
            sent = False
            # this report is built from what is true now: asking for a fresh one before was answered
            self._report_requested.clear()
            try:
                report = await build_models_report(self._config.backends, self._http,
                                                   config_fields=self._settings.report_fields(tables))
            except Exception:
                # Logged, not fatal: requests keep being relayed and the next refresh tries again
                logger.exception('Model discovery failed; retrying in %.0f s', MODEL_REFRESH_INTERVAL_SECONDS)
            else:
                if not keeps_unchecked:
                    report = without_unchecked_marker(report)
                if first_of_connection and self._recently_acknowledged(report):
                    logger.debug('Models unchanged since the gateway acknowledged them; not reporting again yet')
                else:
                    self._sent_report = report
                    acks_before = self._acks_received
                    await self._send(socket, {'type': protocol.FRAME_MODELS, 'backends': report})
                    self._sent_at = self._clock()
                    self._show_report(report)
                    sent = True
                    self._first_report_sent.set()
                if memory_report is not None:
                    memory_report.models_reported(report)
            wait = MODEL_REFRESH_INTERVAL_SECONDS
            if confirming and sent:
                await asyncio.sleep(MODELS_REPORT_RESEND_SECONDS)
                if self._acks_received == acks_before:
                    logger.info('The gateway did not register the models report; sending it again')
                    first_of_connection = confirming = False
                    continue
                wait -= MODELS_REPORT_RESEND_SECONDS
            first_of_connection = False
            confirming = await self._until_next_report(wait)

    async def _until_next_report(self, seconds: float) -> bool:
        """Wait ``seconds`` for the next refresh, or less when a fresh report was asked for (True
        then): as soon as the gateway stores one again.

        The gateway times its limit from when the last report it stored arrived, which is before
        its acknowledgement left; so the time is counted from when that acknowledgement came in,
        or from the last report sent if that is later (its acknowledgement may still be on its
        way). What the client cannot know of, the report's being acknowledged confirms."""
        try:
            await asyncio.wait_for(self._report_requested.wait(), seconds)
        except TimeoutError:
            return False
        known: list[float] = [moment for moment in (self._sent_at, self._acked_at) if moment is not None]
        if known:
            await asyncio.sleep(max(0.0, MODELS_REPORT_RESEND_SECONDS - (self._clock() - max(known))))
        return True

    def _show_report(self, report: list) -> None:
        """A report that is not the one last shown is shown, with its acknowledgement; a refresh that changed
        nothing only in the details."""
        if report == self._shown_report:
            logger.debug(events.models_reported(report))
            return
        logger.info(events.models_reported(report))
        self._shown_report = report
        self._shows_next_ack = True

    def _recently_acknowledged(self, report: list) -> bool:
        """A reconnect loop must not turn into a loop of identical reports the gateway refuses anyway."""
        return report == self._acked_report and self._clock() - self._acked_at < MODELS_REPORT_RESEND_SECONDS

    async def _keep_alive(self, socket, pong_received: asyncio.Event):
        """Application-level ping: keeps proxies from idling the socket out and detects a dead gateway."""
        while True:
            await asyncio.sleep(self._config.ping_interval)
            pong_received.clear()
            await self._send(socket, {'type': protocol.FRAME_PING})
            try:
                await asyncio.wait_for(pong_received.wait(), self._config.pong_timeout)
            except asyncio.TimeoutError:
                logger.warning('No pong from the gateway within %.0f s; dropping the connection',
                               self._config.pong_timeout)
                await socket.close()
                return

    async def _close_when_stopped(self, socket):
        await self._stopping.wait()
        await socket.close()

    # ----- frame helpers -----

    @staticmethod
    async def _send(socket, frame: dict):
        await socket.send(json.dumps(frame))

    async def _receive_frame(self, socket, timeout: float) -> dict:
        raw = await asyncio.wait_for(socket.recv(), timeout)
        frame = self._parse(raw)
        if frame is None:
            raise FatalGatewayError('Gateway sent something that is not a JSON object during the handshake')
        return frame

    @staticmethod
    def _parse(raw):
        try:
            frame = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning('Ignoring a frame that is not JSON')
            return None
        return frame if isinstance(frame, dict) else None
