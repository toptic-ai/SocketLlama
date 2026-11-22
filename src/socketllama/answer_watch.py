"""What the client learns from an answer while it relays it, for the line its owner sees when the answer ends.

The relay passes the backend's bytes on unchanged; an AnswerWatcher reads along: the token counts the
backend reports and, for an answer that failed, the backend's reason. Nothing of the answer's text is
kept.

The relay talks to the OpenAI-compatible endpoint (``/chat/completions``). There the counts are in
``usage``: ``prompt_tokens`` and ``completion_tokens``, which Ollama fills from its own
``prompt_eval_count`` and ``eval_count``. A streamed answer has them in its last event when the request
asked for them (``stream_options.include_usage``, which toptic.ai sends), an answer that is not streamed
always. A backend that does not count gives none, and the line says nothing about tokens.
"""
import json
from dataclasses import dataclass

import httpx

from .text import SHOWN_TEXT_MAX_CHARS, shortened

STREAMING_CONTENT_TYPE = 'text/event-stream'
SSE_DATA_PREFIX = b'data:'
SSE_DONE = b'[DONE]'
LINE_END = b'\n'
CARRIAGE_RETURN = b'\r'
USAGE_FIELD = 'usage'
PROMPT_TOKENS_FIELD = 'prompt_tokens'
COMPLETION_TOKENS_FIELD = 'completion_tokens'
ERROR_FIELD = 'error'
ERROR_MESSAGE_FIELD = 'message'
# What kind of error it is, an identifier: OpenAI's "type", else its "code"
ERROR_KIND_FIELDS = ('type', 'code')
# Only an event holding one of these is parsed: the others carry answer text and nothing to learn.
# Inside a JSON string a quote is escaped, so the text of an answer never matches them by itself.
WATCHED_MARKERS = (f'"{USAGE_FIELD}"'.encode(), f'"{ERROR_FIELD}"'.encode())
# An event longer than this is passed on unread: one with usage or an error is far shorter
MAX_WATCHED_LINE_BYTES = 64 * 1024
# An answer that is not streamed is read whole, up to this size; a longer one is passed on unread
MAX_WATCHED_BODY_BYTES = 1024 * 1024
# A backend's reason and the kind of its error are cut to this length
ERROR_SHOWN_CHARS = SHOWN_TEXT_MAX_CHARS


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class AnswerFacts:
    """``status``: the backend's HTTP status; ``usage``: its token counts, if it gave them; ``error``: its
    reason (printable, one short line) if it gave one, in an error answer or in an event of a stream, and
    ``error_kind`` the kind of that error. The reason may quote the request (a proxy saying what it refused
    in it), so it is only a detail; the kind is an identifier."""
    status: int
    usage: TokenUsage | None = None
    error: str | None = None
    error_kind: str | None = None

    @property
    def succeeded(self) -> bool:
        return httpx.codes.is_success(self.status) and self.error is None


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class AnswerWatcher:
    """Reads along the body of one answer: ``feed`` every chunk as relayed, then ask for the ``result``."""

    def __init__(self, status: int, content_type: str) -> None:
        self._status = status
        self._streamed = content_type.startswith(STREAMING_CONTENT_TYPE)
        # The line of a stream read so far, or the body of an answer that is not streamed
        self._pending = bytearray()
        # The line (or body) being read went beyond its limit: skipped (a line up to its end)
        self._skipping = False
        self._usage: TokenUsage | None = None
        self._error: str | None = None
        self._error_kind: str | None = None

    def feed(self, chunk: bytes) -> None:
        if self._streamed:
            self._read_lines(chunk)
        else:
            self._keep(chunk, MAX_WATCHED_BODY_BYTES)

    def result(self) -> AnswerFacts:
        if not self._skipping and self._pending:
            if self._streamed:
                self._read_event(bytes(self._pending))
            else:
                self._read_json(bytes(self._pending))
        self._pending.clear()
        return AnswerFacts(status=self._status, usage=self._usage, error=self._error, error_kind=self._error_kind)

    def _read_lines(self, chunk: bytes) -> None:
        start = 0
        while (end := chunk.find(LINE_END, start)) >= 0:
            self._keep(chunk[start:end], MAX_WATCHED_LINE_BYTES)
            if not self._skipping:
                self._read_event(bytes(self._pending))
            self._pending.clear()
            self._skipping = False
            start = end + len(LINE_END)
        self._keep(chunk[start:], MAX_WATCHED_LINE_BYTES)

    def _keep(self, part: bytes, limit: int) -> None:
        if self._skipping:
            return
        if len(self._pending) + len(part) > limit:
            self._skipping = True
            self._pending.clear()
            return
        self._pending += part

    def _read_event(self, line: bytes) -> None:
        line = line.rstrip(CARRIAGE_RETURN)
        if not line.startswith(SSE_DATA_PREFIX):
            return
        data = line[len(SSE_DATA_PREFIX):].strip()
        if data != SSE_DONE and any(marker in data for marker in WATCHED_MARKERS):
            self._read_json(data)

    def _read_json(self, data: bytes) -> None:
        try:
            parsed = json.loads(data)
        except (ValueError, RecursionError):
            return
        if not isinstance(parsed, dict):
            return
        usage = parsed.get(USAGE_FIELD)
        if isinstance(usage, dict) and _is_count(usage.get(PROMPT_TOKENS_FIELD)) \
                and _is_count(usage.get(COMPLETION_TOKENS_FIELD)):
            self._usage = TokenUsage(prompt_tokens=usage[PROMPT_TOKENS_FIELD],
                                     completion_tokens=usage[COMPLETION_TOKENS_FIELD])
        error = parsed.get(ERROR_FIELD)
        message = error.get(ERROR_MESSAGE_FIELD) if isinstance(error, dict) else error
        if isinstance(message, str) and message:
            self._error = shortened(message, ERROR_SHOWN_CHARS)
        if isinstance(error, dict):
            kind = next((error[field] for field in ERROR_KIND_FIELDS if isinstance(error.get(field), str)
                         and error[field]), None)
            if kind is not None:
                self._error_kind = shortened(kind, ERROR_SHOWN_CHARS)
