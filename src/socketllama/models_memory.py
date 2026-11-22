"""What the gateway is told about the models Ollama holds in memory (capability models.memory).

Only for a gateway that announced the capability, and only about Ollama backends: other kinds
cannot tell what is in memory. Each frame is a full snapshot of what /api/ps lists for every
Ollama backend, given by the names the models report uses for them; models the report does not
carry (not shared, no chat model) are not told. A backend whose /api/ps could not be read, or
that the last models report could not check, is left out: what it holds is unknown.

When: once a connection's first models report is out (the gateway maps the names onto the
models it reported), and after that whenever the snapshot is not the one last sent on the
connection. It is checked every MODELS_MEMORY_POLL_SECONDS, after each models report, after a
relayed answer of an Ollama backend ended (its model is in memory now) and once load_at_start
is over. At most one frame goes out in MODELS_MEMORY_MIN_INTERVAL_SECONDS; what changes before
that has passed is sent once it has, as one frame with the latest state.

Nothing here is part of relaying: what goes wrong is logged, and requests are relayed on.
"""
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping

import httpx

from . import events, protocol
from .backends import ollama_loaded_models, ollama_model_reference
from .config import KIND_OLLAMA, Backend

logger = logging.getLogger('socketllama')

# How often the client looks at what Ollama holds in memory without a reason to: models also come
# and go without a request through the tunnel (Ollama unloads them after their keep_alive, the
# ollama command loads them)
MODELS_MEMORY_POLL_SECONDS = 60.0
# The least time between two models.memory frames, this client's own bound: what changes sooner goes
# out once it has passed, as one frame with the latest state. Timed on this client's clock from the last
# frame sent, on whichever connection. The gateway stores every valid frame, however soon it comes, and
# does not limit them itself: it counts on clients to keep to such a bound. It passes changes on to the
# open pages at most once per SOCKETLLAMA_MODELS_MEMORY_MIN_INTERVAL_SECONDS of toptic.ai's gateway
# (ai_helper/settings.py in the toptic.ai repository), its own coalescing interval, which this value need not
# match.
MODELS_MEMORY_MIN_INTERVAL_SECONDS = 5.0

# One Ollama backend of a frame: {"backend": name, "models": [names]}
BackendMemory = dict[str, str | list[str]]


class ModelsMemoryReport:
    """One per connection to a gateway with the capability; ``run`` is a task of the connection.

    ``send`` is an async callable taking a frame dict. ``sent_at``: when the last frame went out
    (``clock``), on an earlier connection; read it back from the attribute when this one is over.
    ``shown``: the snapshot the owner was last shown (see events), on an earlier connection, read back
    the same way; a snapshot is shown when it is not that one, so a reconnect shows nothing again.
    """

    def __init__(self, backends: Mapping[str, Backend], http: httpx.AsyncClient,
                 send: Callable[[dict], Awaitable[None]], clock: Callable[[], float] = time.monotonic,
                 sent_at: float | None = None, shown: list[BackendMemory] | None = None) -> None:
        self._backends = backends
        self._http = http
        self._send = send
        self._clock = clock
        self.sent_at = sent_at
        self.shown = shown
        # Of each Ollama backend the last models report checked, the names of its models; None until a report is out
        self._reported: dict[str, list[str]] | None = None
        # What the gateway was last sent on this connection; None: nothing it is known to hold
        self._sent: list[BackendMemory] | None = None
        self._check_requested = asyncio.Event()
        # Backends whose /api/ps could not be read last time: a failure is logged when it starts, not each time
        self._failing: set[str] = set()

    def models_reported(self, report: list) -> None:
        """The gateway was sent, or already holds, this models report (as sent): names are taken
        from it, and what is in memory is checked."""
        self._reported = {
            entry['name']: [model['name'] for model in entry['models']] for entry in report
            if entry['kind'] == KIND_OLLAMA and entry.get(protocol.BACKEND_CHECKED_FIELD) is not False}
        self.check()

    def check(self) -> None:
        """Have what Ollama holds in memory checked soon. Returns at once."""
        self._check_requested.set()

    def gateway_reported_error(self) -> None:
        """The gateway answered with an error frame. It does not say which frame it was about, so it
        may have ignored the last one of these: the next check sends the state even if unchanged.
        Not at once, so a gateway that refuses each one is not sent one after another."""
        self._sent = None

    async def run(self) -> None:
        """Check and send until cancelled, with the connection."""
        while True:
            await self._next_check()
            snapshot = await self._snapshot()
            if snapshot is not None and snapshot != self._sent:
                await self._send_latest(snapshot)

    async def _next_check(self) -> None:
        try:
            await asyncio.wait_for(self._check_requested.wait(), MODELS_MEMORY_POLL_SECONDS)
        except TimeoutError:
            pass
        self._check_requested.clear()

    async def _send_latest(self, snapshot: list[BackendMemory]) -> None:
        """Send ``snapshot``, or once MODELS_MEMORY_MIN_INTERVAL_SECONDS has passed since the last frame, what is
        true then."""
        wait = 0.0 if self.sent_at is None else MODELS_MEMORY_MIN_INTERVAL_SECONDS - (self._clock() - self.sent_at)
        if wait > 0:
            await asyncio.sleep(wait)
            if self._check_requested.is_set():
                # What changed meanwhile goes out with this frame, not in one more
                self._check_requested.clear()
                snapshot = await self._snapshot()
                if snapshot == self._sent:
                    return
        await self._send({'type': protocol.FRAME_MODELS_MEMORY, 'backends': snapshot})
        self._sent, self.sent_at = snapshot, self._clock()
        logger.debug('Told the gateway which models are in memory: %s', snapshot)
        # A snapshot without a backend knows nothing to show (no Ollama backend, or none whose memory could be read)
        if snapshot and snapshot != self.shown:
            logger.info(events.models_in_memory(snapshot))
            self.shown = snapshot

    async def _snapshot(self) -> list[BackendMemory] | None:
        """What every Ollama backend of the last models report holds in memory; None before a report."""
        if self._reported is None:
            return None
        snapshot: list[BackendMemory] = []
        for name, models in self._reported.items():
            loaded = await ollama_loaded_models(self._backends[name], self._http)
            if loaded is None:
                if name not in self._failing:
                    self._failing.add(name)
                    logger.warning('Backend %s: toptic.ai is not told which of its models are in memory for now, '
                                   'because Ollama does not tell it (its /api/ps could not be read). This is logged '
                                   'once, until it works again.', name)
                continue
            if name in self._failing:
                self._failing.discard(name)
                logger.info('Backend %s: toptic.ai is told which of its models are in memory again', name)
            snapshot.append({'backend': name, 'models': [model for model in models
                                                         if ollama_model_reference(model) in loaded]})
        return snapshot
