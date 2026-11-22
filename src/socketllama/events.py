"""The lines the running client shows its owner: one line per event that matters.

Each function here returns the text of one line; the caller logs it, at INFO (a problem at WARNING),
and the log format adds the local time (console.LogFormatter). Details stay at DEBUG. What came from
elsewhere (a name the gateway sends, a model name, a backend's reason) is made printable here.

Never part of a line: the text of a prompt or of an answer, a token or a secret. Of a prompt the owner
sees who sent it, for which model, and its size: what is known of it before the answer, without a
tokenizer (the number of messages, the characters of their text, the attachments).

Requests are numbered from 1 for the whole run of the client, so the lines of one request can be told
apart from those of others running at the same time.
"""
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from . import protocol
from .protocol import MIN_PRICE_PER_UNIT
from .answer_watch import TokenUsage
from .config import ModelPrice
from .text import SHOWN_TEXT_MAX_CHARS, printable, shortened

FIRST_REQUEST_NUMBER = 1
REQUEST_NUMBER_MARK = '#'
UNKNOWN_USER = 'unknown user'
NO_REASON_GIVEN = 'no reason given'

# How an answer ended: the word that follows the request's number
OUTCOME_DONE = 'done'
OUTCOME_CANCELLED = 'cancelled'
OUTCOME_ERROR = 'error'
# Why an answer was cancelled
STOPPED_BY_GATEWAY = 'stopped by toptic.ai'
CONNECTION_ENDED = 'the connection to toptic.ai ended'

# Of a chat completion's messages
MESSAGES_FIELD = 'messages'
CONTENT_FIELD = 'content'
PART_TYPE_FIELD = 'type'
TEXT_PART_TYPE = 'text'
TEXT_FIELD = 'text'

# Fields of a models_ack frame that list what changed, with the words for them
ACK_CHANGES = (('created', 'new'), ('updated', 'changed'), ('removed', 'removed'), ('skipped', 'skipped'))
ACK_UNCHECKED_FIELD = 'unchecked'
# Of a models.memory frame's backends
MEMORY_BACKEND_FIELD = 'backend'
MEMORY_MODELS_FIELD = 'models'

# What an auth_failed reason means, and what to do about it
AUTH_FAILURES = {
    protocol.AUTH_FAILED_TOKEN: 'the connection token is wrong, revoked or expired; create a new one on the '
                                "machine's page and copy its config block",
    protocol.AUTH_FAILED_SCOPE: "the token may not connect machines; create a connection token on the machine's page",
    protocol.AUTH_FAILED_INSTANCE: 'no machine of this name belongs to the owner of the token, the token is that of '
                                   "another machine, or the signing secret is wrong; copy the config block of the "
                                   "machine's page again",
    protocol.AUTH_FAILED_PROTOCOL: 'toptic.ai speaks another version of the protocol; update the client',
    protocol.AUTH_FAILED_SECRET_UNREADABLE: "toptic.ai cannot read the machine's signing secret; create a new signing "
                                            "secret on the machine's page and copy its config block",
}
AUTH_FAILURE_UNKNOWN = "the machine's page on toptic.ai says why"

# How a price is shown: in dollars, with cents (more places where the price has them)
CENTS_PLACES = 2
# A price of the least there is (MIN_PRICE_PER_UNIT) in and out
FREE = 'free'


def _counted(count: int, noun: str) -> str:
    """"1 message", "1,234 characters": the count with the noun, in the plural unless it is one."""
    return f'{count:,} {noun}' if count == 1 else f'{count:,} {noun}s'


def _number(number: int) -> str:
    return f'{REQUEST_NUMBER_MARK}{number}'


def _model(backend: str, model: str) -> str:
    return f'{printable(backend)}/{printable(model)}'


# ----- a prompt and its answer -----

@dataclass(frozen=True)
class PromptSize:
    messages: int
    characters: int
    attachments: int


def prompt_size(payload: Mapping) -> PromptSize:
    """The size of a chat completion request: its messages, the characters of their text, and the parts
    that are not text (an image, a file). What does not have the expected form is not counted."""
    messages = payload.get(MESSAGES_FIELD)
    if not isinstance(messages, list):
        return PromptSize(messages=0, characters=0, attachments=0)
    characters = attachments = 0
    for message in messages:
        content = message.get(CONTENT_FIELD) if isinstance(message, dict) else None
        if isinstance(content, str):
            characters += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get(PART_TYPE_FIELD) != TEXT_PART_TYPE:
                    attachments += 1
                elif isinstance(part.get(TEXT_FIELD), str):
                    characters += len(part[TEXT_FIELD])
    return PromptSize(messages=len(messages), characters=characters, attachments=attachments)


def requester(name: object) -> str:
    """Who sent a prompt, as the gateway names them (protocol.REQUEST_USER_FIELD); UNKNOWN_USER without a name."""
    if not isinstance(name, str) or not name.strip():
        return UNKNOWN_USER
    return shortened(name.strip(), protocol.USER_NAME_MAX_LENGTH)


def prompt_received(number: int, requester_name: str, backend: str, model: object, size: PromptSize) -> str:
    sizes = [_counted(size.messages, 'message'), _counted(size.characters, 'character')]
    if size.attachments:
        sizes.append(_counted(size.attachments, 'attachment'))
    return (f'{_number(number)} prompt from {requester_name} for {printable(backend)}/{printable(str(model))}: '
            f'{", ".join(sizes)}')


def answer_ended(number: int, outcome: str, seconds: float, usage: TokenUsage | None = None,
                 detail: str | None = None) -> str:
    """How the answer to a request ended (OUTCOME_*), after how long, with the backend's token counts if it
    gave them, and why if ``detail`` says so."""
    line = f'{_number(number)} {outcome} after {seconds:.1f} s'
    if usage is not None:
        line += (f', {_counted(usage.prompt_tokens, "prompt token")}, '
                 f'{_counted(usage.completion_tokens, "answer token")}')
    return f'{line}: {detail}' if detail else line


def backend_said(number: int, message: str) -> str:
    """The backend's own words about a failed answer: a detail (DEBUG), as they may quote the request."""
    return f'{_number(number)} the backend said: {message}'


def gateway_reported(message: object) -> str:
    """An error frame of the gateway, about something the client sent."""
    return f'toptic.ai reported: {shortened(str(message), SHOWN_TEXT_MAX_CHARS)}'


def paused(number: int) -> str:
    return f'{_number(number)} paused: toptic.ai takes the answer more slowly than it comes'


def resumed(number: int) -> str:
    return f'{_number(number)} resumed'


def model_not_in_memory(number: int, backend: str, model: str) -> str:
    return f'{_number(number)} {printable(backend)}/{printable(model)} is not in memory: Ollama loads it first'


def backend_loading_model(number: int, backend: str) -> str:
    """An Ollama backend refused a prompt because it is still loading a model: the gateway is told just that
    (protocol.ERROR_BACKEND_MODEL_LOADING) instead of being passed Ollama's error answer."""
    return (f'{_number(number)} {printable(backend)} is still loading a model and answers no prompt until it is in '
            f'memory: toptic.ai is told so instead of being passed Ollama\'s error')


def request_refused(code: str, message: str) -> str:
    """A request the client did not relay (a bad signature, a model it does not share, ...); it got no number."""
    return f'Refused a request: {code}, {shortened(message, SHOWN_TEXT_MAX_CHARS)}'


# ----- the connection -----

def connected(gateway_url: str, instance: str) -> str:
    return f'Connected to {gateway_url} as machine {instance}'


def gateway_waits_for_loading(seconds: int) -> str:
    """What the gateway said at connect about how long it waits while a model loads (see protocol)."""
    return (f'toptic.ai waits up to {seconds:,} s for a model to load: this client waits at least as long for a '
            f'model that was not in memory when its prompt came, and for a load it asks Ollama for')


def loading_wait_shortened(seconds: int, longest: int) -> str:
    """The gateway says it waits longer while a model loads than the client goes along with."""
    return (f'toptic.ai says it waits {seconds:,} s for a model to load: this client takes that as {longest:,} s, '
            f'the longest it waits on the word of toptic.ai. A backend\'s read_timeout can be set longer')


def loading_wait_not_said() -> str:
    """Connected to a gateway that does not say how long it waits while a model loads, after one that did."""
    return ('This toptic.ai does not say how long it waits for a model to load: the client waits as long as each '
            'backend\'s read_timeout says again')


def loading_wait_ignored(reason: str) -> str:
    """The gateway said how long it waits while a model loads, in a way the client does not take."""
    return (f'Ignored how long toptic.ai says it waits for a model to load '
            f'({shortened(reason, SHOWN_TEXT_MAX_CHARS)}): the client waits as long as each backend\'s '
            f'read_timeout says')


def auth_refused(reason: object) -> str:
    """Why the gateway refused this client at connect, and what to do about it."""
    if not isinstance(reason, str) or not reason:
        return f'toptic.ai refused this machine ({NO_REASON_GIVEN}): {AUTH_FAILURE_UNKNOWN}'
    return (f'toptic.ai refused this machine ({shortened(reason, SHOWN_TEXT_MAX_CHARS)}): '
            f'{AUTH_FAILURES.get(reason, AUTH_FAILURE_UNKNOWN)}')


# ----- models -----

def _report_models(report: Iterable[Mapping]) -> int:
    return sum(len(backend['models']) for backend in report)


def models_reported(report: list) -> str:
    """A models report as sent: the number of models of each backend, or that it could not be checked."""
    parts = [f'{printable(backend["name"])} not checked' if backend.get(protocol.BACKEND_CHECKED_FIELD) is False
             else f'{printable(backend["name"])} {_counted(len(backend["models"]), "model")}' for backend in report]
    return f'Reported to toptic.ai: {", ".join(parts)}'


def ack_changes_something(ack: Mapping) -> bool:
    """Whether a models_ack says that the gateway changed or left out something."""
    return any(ack.get(field) for field, _ in ACK_CHANGES) or bool(ack.get(ACK_UNCHECKED_FIELD))


def models_registered(ack: Mapping, report: list) -> str:
    """The gateway's models_ack of ``report``: how many models it holds, and what changed."""
    changes = [f'{len(ack.get(field) or [])} {word}' for field, word in ACK_CHANGES if ack.get(field)]
    if ack.get(ACK_UNCHECKED_FIELD):
        changes.append(f'{_counted(len(ack[ACK_UNCHECKED_FIELD]), "backend")} kept as not checked')
    return (f'toptic.ai registered the {_counted(_report_models(report), "model")} reported: '
            f'{", ".join(changes) or "no changes"}')


def models_in_memory(snapshot: list) -> str:
    """A models.memory snapshot: the models each Ollama backend holds in memory now."""
    parts = [f'{printable(backend[MEMORY_BACKEND_FIELD])} '
             f'{", ".join(map(printable, backend[MEMORY_MODELS_FIELD])) or "none"}' for backend in snapshot]
    return f'In memory now: {"; ".join(parts)}'


# ----- a load the client finishes -----

# What these lines say is what the client knows: the model was not in memory when the prompt came, and the
# request ended (cancelled, or given up by the client) before Ollama began its answer. Mostly Ollama was
# still loading the model then, but not always: it may have been loaded by then and busy with a long prompt.

def load_goes_on(number: int, backend: str, model: str) -> str:
    """A request ended before Ollama began its answer, its model was not in memory when it came, and the
    client now has Ollama load it (see ollama_memory)."""
    return (f'{_number(number)} ended before {_model(backend, model)} began its answer, and the model was not in '
            f'memory when the prompt came: the client has Ollama load it, so that the next prompt finds it in memory')


def load_already_going_on(number: int, backend: str, model: str) -> str:
    """As load_goes_on, for a model the client has Ollama load already, after an earlier request."""
    return (f'{_number(number)} ended before {_model(backend, model)} began its answer, and the model was not in '
            f'memory when the prompt came: the client has Ollama load it already')


def load_finished(number: int, backend: str, model: str, seconds: float) -> str:
    """The load the client asked for after request ``number`` ended without an answer is over."""
    return (f'{_model(backend, model)} is in memory after {seconds:.1f} s: the client had Ollama load it after '
            f'{_number(number)} ended without an answer')


def load_failed(number: int, backend: str, model: str, reason: str) -> str:
    """The load the client asked for after request ``number`` ended without an answer failed, and why."""
    return (f'{_model(backend, model)} could not be loaded after {_number(number)} ended without an answer: '
            f'{shortened(reason, SHOWN_TEXT_MAX_CHARS)}. A prompt to it makes Ollama try again')


# ----- prices and descriptions -----

def _dollars(amount: float) -> str:
    """"$0.80": with cents, and with every further place a price has ("$0.0005")."""
    with_cents = f'{amount:.{CENTS_PLACES}f}'
    return f'${with_cents}' if float(with_cents) == amount else f'${amount!r}'


def price_in_words(price: ModelPrice) -> str:
    """"$0.80 in, $2.00 out per 1M tokens", or "free"."""
    if price.input == MIN_PRICE_PER_UNIT and price.output == MIN_PRICE_PER_UNIT:
        return FREE
    return f'{_dollars(price.input)} in, {_dollars(price.output)} out per 1M tokens'


def prices_in_config(prices: Mapping[tuple[str, str], ModelPrice]) -> str:
    """The prices config.toml holds, by (backend, model), as the client starts."""
    parts = [f'{_model(backend, model)} {price_in_words(price)}' for (backend, model), price in prices.items()]
    return f'Prices in config.toml: {"; ".join(parts)}'


def descriptions_in_config(described: Iterable[tuple[str, str]]) -> str:
    """The models config.toml holds a description for, by (backend, model), as the client starts."""
    return f'Descriptions in config.toml: {", ".join(_model(backend, model) for backend, model in described)}'


def description_in_words(description: str) -> str:
    """A description as a line shows it: quoted, made printable, and shortened if it is long."""
    return f'"{shortened(description, SHOWN_TEXT_MAX_CHARS)}"'


def settings_not_taken(plural: str) -> str:
    """config.toml holds ``plural`` ("prices"), and the gateway does not take them from the machine."""
    return (f'This toptic.ai does not take {plural} from your machine: the {plural} in config.toml are not '
            f'reported to it and change nothing there')


def setting_written(noun: str, backend: str, model: str, value: str, path: object) -> str:
    """toptic.ai set the ``noun`` ("price") of a model to ``value`` (in words), and the client wrote it."""
    return f'toptic.ai set the {noun} of {_model(backend, model)} to {value}: written to {path}'


def setting_shown_already(noun: str, backend: str, model: str, value: str, path: object) -> str:
    """toptic.ai set the value the model's line was changed to by hand since the client started."""
    return f'toptic.ai set the {noun} of {_model(backend, model)} to {value}, which {path} shows already'


def backup_kept(path: object, backup: object) -> str:
    return f'Before the first line toptic.ai set was written, {path} was copied to {backup}'


def setting_not_written(noun: str, table: str, backend: str, model: str, value: str, path: object, reason: str,
                        line: str) -> str:
    """A value toptic.ai set that could not be written (``reason``), with the ``line`` to put in by hand under
    the header [``table``]."""
    return (f'toptic.ai set the {noun} of {_model(backend, model)} to {value}, but it could not be written to '
            f'{path} ({shortened(reason, SHOWN_TEXT_MAX_CHARS)}). It stays in force on toptic.ai. To show it in the '
            f'file, put this line under [{table}] of backend {printable(backend)}: {printable(line)}')


def setting_changed_meanwhile(noun: str, backend: str, model: str, value: str, path: object) -> str:
    """toptic.ai set a value for a model whose line in config.toml was changed since the client read it."""
    return (f'toptic.ai set the {noun} of {_model(backend, model)} to {value}, but its line in {path} was changed '
            f'since the client started, so it is not written over. Restart the client to report the {noun} in the '
            f'file: the one that reaches toptic.ai last counts')


def setting_refused(noun: str, reason: str) -> str:
    """A value toptic.ai sent that this client does not take (not a price, too long a description, ...)."""
    return f'Ignored a {noun} toptic.ai sent: {shortened(reason, SHOWN_TEXT_MAX_CHARS)}'


def settings_entry_refused(reason: str) -> str:
    """An entry of the settings toptic.ai sent that this client does not take at all (no object, a model it does
    not share, ...)."""
    return f'Ignored what toptic.ai sent for a model: {shortened(reason, SHOWN_TEXT_MAX_CHARS)}'