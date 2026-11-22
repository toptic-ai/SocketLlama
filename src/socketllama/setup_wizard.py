"""The conversation of ``socketllama setup``: from nothing to a complete configuration.

Only the conversation lives here. What is shown and typed goes through a Console, backends are
asked for their models through a function handed in, and nothing is written: the result is a
SetupPlan for the caller to carry out. So the whole conversation runs in tests from scripted
answers, without a terminal and without a backend.
"""
import itertools
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Protocol
from urllib.parse import urlsplit

from . import protocol
from .backends import (
    DISCOVERY_TIMEOUT_SECONDS, DROPPED_NAMES_SHOWN, NOTE_BACKEND_NOT_REPORTED, NOTE_DETAILS_UNREADABLE, NOTE_DOWN,
    NOTE_ENTRY_IGNORED, NOTE_FAILED, NOTE_NAMES_REFUSED, NOTE_REPORT_TOO_LARGE, NOTE_TIMED_OUT, NOTE_TOO_MANY_MODELS,
    BackendUnavailable, DiscoveryNote,
)
from .config import (
    BACKEND_NAME_SEPARATOR, INSECURE_GATEWAY_SCHEME, KEY_KEEP_LOADED, KEY_LOAD_AT_START, KIND_LITELLM, KIND_OLLAMA,
    KIND_OPENAI, MAX_KEEP_LOADED_HOURS, MAX_PORT, MIN_PORT, OWNER_ONLY_PERMISSIONS, REASON_ADDRESS_BRACKETS,
    REASON_ADDRESS_PORT, REASON_BACKEND_NAME_CHARACTERS, REASON_BACKEND_NAME_TOO_LONG, REASON_BACKEND_URL_CREDENTIALS,
    REASON_BACKEND_URL_MALFORMED, REASON_GATEWAY_URL_MALFORMED, REASON_GATEWAY_URL_SCHEME, REASON_KEEP_LOADED,
    REASON_KEEP_LOADED_TOO_LONG, Backend, ConfigError, check_backend_name, check_gateway_url, check_keep_loaded,
    keep_loaded_seconds, normalized_backend_url, parse_config,
)
from .config_writer import (
    KEY_ALLOW_INSECURE, KEY_API_KEY, KEY_BASE_URL, KEY_INSTANCE, KEY_KIND, KEY_MODELS, KEY_NAME, KEY_SIGNING_SECRET,
    KEY_TOKEN, KEY_URL, SECTION_BACKENDS, SECTION_GATEWAY, ConfigTarget,
)
from .ollama_memory import (
    OLLAMA_DEFAULT_KEEP_ALIVE_MINUTES, OLLAMA_DEFAULT_KEEP_ALIVE_SECONDS, OLLAMA_DEFAULT_MAX_LOADED_MODELS,
    keep_loaded_in_words,
)
from .text import printable

# The addresses config.example.toml names: where a backend of a kind usually listens
DEFAULT_BACKEND_URLS = {
    KIND_OLLAMA: 'http://127.0.0.1:11434/v1',
    KIND_LITELLM: 'http://127.0.0.1:4000/v1',
}
KIND_LABELS = {
    KIND_OLLAMA: 'Ollama',
    KIND_LITELLM: 'LiteLLM proxy',
    KIND_OPENAI: 'OpenAI-compatible server',
}
FIRST_BACKEND_NAME = 'ollama'
# Examples in answers to an address that cannot be one: for a kind without a usual address (the
# port is the one vLLM listens on), and for the gateway
OPENAI_EXAMPLE_URL = 'http://127.0.0.1:8000/v1'
GATEWAY_EXAMPLE_URL = 'wss://toptic.example/ws/socketllama/'
# An address as a backend of each kind may have it: the example where an answer cannot be one
EXAMPLE_BACKEND_URLS = {**DEFAULT_BACKEND_URLS, KIND_OPENAI: OPENAI_EXAMPLE_URL}
IPV6_EXAMPLE = '[::1]'

# What the machine's page on toptic.ai shows in its config block instead of a value it does not have yet:
# CONFIG_TOKEN_PLACEHOLDER and CONFIG_SECRET_UNREADABLE_PLACEHOLDER of toptic.ai
# (ai_helper/service/provider_texts.py in the toptic.ai repository)
TOKEN_PLACEHOLDER = 'PASTE-YOUR-CONNECTION-TOKEN-HERE'
SIGNING_SECRET_PLACEHOLDER = 'CREATE-A-NEW-SIGNING-SECRET-BELOW'
PLACEHOLDERS = (TOKEN_PLACEHOLDER, SIGNING_SECRET_PLACEHOLDER)

# The texts of a pasted block's [gateway] section that setup takes
GATEWAY_TEXT_KEYS = (KEY_URL, KEY_INSTANCE, KEY_TOKEN, KEY_SIGNING_SECRET)
GATEWAY_LABELS = {
    KEY_URL: 'gateway address',
    KEY_INSTANCE: 'machine name',
    KEY_TOKEN: 'connection token',
    KEY_SIGNING_SECRET: 'signing secret',
}

# Ends a pasted block (which holds empty lines itself, so an empty line cannot)
END_OF_BLOCK = 'END'
BLOCK_LINE_PROMPT = ''

SELECT_ALL = 'all'
SELECT_FUTURE = 'future'
SELECT_NONE = 'none'
RANGE_SEPARATOR = '-'
SPACE_AROUND_RANGE_SEPARATOR = re.compile(rf'\s*{RANGE_SEPARATOR}\s*')
SELECTION_ITEM_SEPARATORS = re.compile(r'[,\s]+')
RANGE_ENDS = 2
FIRST_NUMBER = 1
# The number in the name suggested for a second backend of a kind ("ollama-2")
SECOND_NUMBER = 2
# range() leaves its end out
INCLUSIVE_END = 1
SELECTION_EXAMPLE = '1,3-4'
RANGE_EXAMPLE = '2-4'

YES_ANSWERS = ('y', 'yes')
NO_ANSWERS = ('n', 'no')
YES_BY_DEFAULT = 'Y/n'
NO_BY_DEFAULT = 'y/N'

# What a menu answer stands for
WAY_PASTE = 'paste'
WAY_QUESTIONS = 'questions'
ACTION_RETRY = 'retry'
ACTION_ADDRESS = 'address'
ACTION_API_KEY = 'api_key'
ACTION_SHARE_FUTURE = 'share_future'
ACTION_SKIP = 'skip'
ACTION_DONE = 'done'
# What to do about a backup that could not be made (SetupWizard.backup_failed)
BACKUP_RETRY = 'backup_retry'
BACKUP_SKIP = 'backup_skip'
BACKUP_STOP = 'backup_stop'

# How long models stay in memory, to pick from: (what the configuration gets, what the menu says)
KEEP_LOADED_CHOICES = (('30m', '30 minutes'), ('2h', '2 hours'), (-1, 'Until Ollama stops'))
ACTION_OTHER_TIME = 'other_time'
KEEP_LOADED_EXAMPLE = '45m'
# A time typed as a number alone: seconds; without a point, a whole number of them. Digits are 0-9
# alone (re.ASCII), as Ollama reads a number: int() and float() would also take others ("３６００")
SECONDS_TYPED = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', re.ASCII)
DECIMAL_POINT = '.'
# What Ollama does when nobody says otherwise, as the questions name it
OLLAMA_DEFAULT_KEEP = f'{OLLAMA_DEFAULT_KEEP_ALIVE_MINUTES} minutes'
# Said when a time typed after "longer?" is not longer
NOT_LONGER_THAN_DEFAULT = (f"That is not longer than Ollama's {OLLAMA_DEFAULT_KEEP}. Choose a longer time, or -1 for "
                           f"until Ollama stops. A shorter time can still be set by hand afterwards: keep_loaded in "
                           f"the configuration file, as the README describes.")

# Of an answer that is repeated in the reason it cannot be used, at most this many characters
ANSWER_SHOWN_LENGTH = 20
SHORTENED = '...'
# HTTP statuses that mean: the key is missing or wrong / may not do this
KEY_REFUSED_STATUSES = (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)

# A rule of the configuration (ConfigError.reason) in the words of the conversation. Each is
# completed with an example of what is asked for (see rule_in_words), whether it names one or not
RULES_IN_WORDS = {
    REASON_GATEWAY_URL_MALFORMED: 'That is not a gateway address. It looks like {example}; copy it as the page '
                                  'shows it.',
    REASON_GATEWAY_URL_SCHEME: 'A gateway address starts with wss://. Copy it as the page shows it.',
    REASON_BACKEND_NAME_CHARACTERS: f'A name cannot contain "{BACKEND_NAME_SEPARATOR}". Choose another, like '
                                    f'{{example}}.',
    REASON_BACKEND_NAME_TOO_LONG: f'A name can have at most {protocol.BACKEND_NAME_MAX_LENGTH} characters. Choose a '
                                  f'shorter one, like {{example}}.',
    REASON_BACKEND_URL_MALFORMED: 'That is not an address a backend can have. It starts with http:// or https://, '
                                  'like {example}.',
    REASON_BACKEND_URL_CREDENTIALS: 'Leave user name and password out of the address. A key the backend wants is '
                                    'asked for on its own.',
    REASON_ADDRESS_PORT: f'The port of an address is a whole number from {MIN_PORT} to {MAX_PORT}: the number after '
                         f'the last colon, as in {{example}}.',
    REASON_KEEP_LOADED: 'That is not a time Ollama takes. Write it like {example} or 1h30m, or as a number of '
                        'seconds; -1 means until Ollama stops.',
    REASON_KEEP_LOADED_TOO_LONG: f'That is not a time Ollama takes: it counts at most {MAX_KEEP_LOADED_HOURS} hours. '
                                 f'-1 means until Ollama stops.',
    REASON_ADDRESS_BRACKETS: f'An IPv6 address in an address stands in brackets, like {IPV6_EXAMPLE}. Here a bracket '
                             f'is missing, or the brackets hold something else.',
}

STEPS = ('the connection to toptic.ai', 'the models to share', 'saving the configuration')
STEP_GATEWAY, STEP_BACKENDS, STEP_SAVING = STEPS
INDENT = '  '
NOT_ECHOED = 'not shown while you type'
SHARES_EVERY_MODEL = 'every model it offers, including models you pull or add later'
FUTURE_MODELS_WARNING = ('This shares every model the backend has now and any you pull or add later, without '
                         'asking again.')


class AnswerError(ValueError):
    """An answer cannot be used; the message says why and what to enter instead."""


class SetupDeclined(Exception):
    """The person decided at the end not to write the file."""


class Console(Protocol):
    """What the conversation needs of a terminal."""

    def say(self, text: str = '') -> None: ...

    def ask(self, prompt: str) -> str: ...

    def ask_secret(self, prompt: str) -> str:
        """Like ask, but what is typed is not shown."""


@dataclass(frozen=True)
class Discovered:
    """What asking a backend gave."""
    # Names of the models it could share
    names: list
    # What discovery had to tell besides (DiscoveryNote), for the conversation to say in its words
    notes: Iterable = ()


# Asks a backend for the models it could share; raises BackendUnavailable
Discover = Callable[[Backend], Discovered]


@dataclass(frozen=True)
class Selection:
    """Which of the models listed are shared."""
    # Zero-based positions in the list, in its order
    positions: tuple = ()
    # Everything the backend offers, including models that are not there yet (models = [])
    includes_future: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.positions and not self.includes_future


@dataclass(frozen=True)
class SetupPlan:
    """What the conversation ended with."""
    # The configuration, shaped like a parsed configuration file
    data: dict
    # Whether a file that is in the way is kept under another name
    keep_backup: bool


def parse_selection(answer: str, count: int) -> Selection:
    """The answer to "which of these ``count`` models?": numbers and ranges like ``1,3-4``, or a keyword."""
    answer = SPACE_AROUND_RANGE_SEPARATOR.sub(RANGE_SEPARATOR, answer.strip().lower())
    if answer == SELECT_ALL:
        return Selection(positions=tuple(range(count)))
    if answer == SELECT_FUTURE:
        return Selection(includes_future=True)
    if answer == SELECT_NONE:
        return Selection()
    items = [item for item in SELECTION_ITEM_SEPARATORS.split(answer) if item]
    if not items:
        raise AnswerError(f'Enter the numbers of the models to share, like {SELECTION_EXAMPLE}, or {SELECT_ALL}, '
                          f'{SELECT_FUTURE} or {SELECT_NONE}.')
    chosen = set()
    for item in items:
        first, last = (_number_in_list(end, count) for end in _selection_range(item))
        for number, end in ((first, item.split(RANGE_SEPARATOR)[0]), (last, item.split(RANGE_SEPARATOR)[-1])):
            if number is None:
                raise AnswerError(f'{_shortened(end)} is not in the list: the numbers go from {FIRST_NUMBER} to '
                                  f'{count}.')
        if first > last:
            raise AnswerError(f'{item} runs backwards: write the lower number first, {last}{RANGE_SEPARATOR}{first}.')
        chosen.update(range(first - FIRST_NUMBER, last - FIRST_NUMBER + INCLUSIVE_END))
    return Selection(positions=tuple(sorted(chosen)))


def _selection_range(item: str) -> tuple:
    """The digits of (first, last) of one item of a selection: a number, or two joined by the range separator."""
    ends = item.split(RANGE_SEPARATOR)
    if len(ends) > RANGE_ENDS or not all(_is_number(end) for end in ends):
        raise AnswerError(f'"{_shortened(item)}" is not a number from the list, a range like {RANGE_EXAMPLE}, or one '
                          f'of {SELECT_ALL}, {SELECT_FUTURE}, {SELECT_NONE}.')
    return ends[0], ends[-1]


def _is_number(text: str) -> bool:
    return text.isascii() and text.isdigit()


def _number_in_list(digits: str, count: int) -> int | None:
    """The number ``digits`` write, if it is one of a list of ``count`` entries; None otherwise.

    More digits than ``count`` has cannot be one: they are not converted (Python refuses to
    convert more than a few thousand, and nothing is gained by trying).
    """
    significant = digits.lstrip('0')
    if _is_number(digits) and significant and len(significant) <= len(str(count)) and int(significant) <= count:
        return int(significant)
    return None


def _shortened(answer: str) -> str:
    """An answer as it is repeated in the reason it cannot be used: printable, and not at full length."""
    shown = printable(answer)
    return shown if len(shown) <= ANSWER_SHOWN_LENGTH else f'{shown[:ANSWER_SHOWN_LENGTH]}{SHORTENED}'


def parse_gateway_block(text: str) -> dict:
    """The usable [gateway] values of the config block the machine's page shows.

    Nothing else of the block is taken. A value that is missing, empty, of another type or one
    of the page's placeholders is left out: the conversation asks for it.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as ex:
        raise AnswerError(f'That cannot be read as a config block ({ex}). Copy the whole block from the page, '
                          f'from the line [{SECTION_GATEWAY}] to its last line.')
    gateway = data.get(SECTION_GATEWAY)
    if not isinstance(gateway, dict):
        raise AnswerError(f'That has no [{SECTION_GATEWAY}] section. Copy the whole block from the page, starting '
                          f'with the line [{SECTION_GATEWAY}].')
    found = {}
    for key in GATEWAY_TEXT_KEYS:
        value = gateway.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in PLACEHOLDERS:
            found[key] = value.strip()
    if gateway.get(KEY_ALLOW_INSECURE) is True:
        found[KEY_ALLOW_INSECURE] = True
    return found


def summary_lines(data: dict) -> list:
    """What will be written, for a person to check; of a secret only its length."""
    gateway = data[SECTION_GATEWAY]
    lines = [f'Gateway address:   {gateway[KEY_URL]}']
    if gateway.get(KEY_ALLOW_INSECURE):
        lines.append(f'{INDENT}not encrypted (ws://): for a development server only')
    lines += [f'Machine name:      {gateway[KEY_INSTANCE]}',
              f'Connection token:  {_masked(gateway[KEY_TOKEN])}',
              f'Signing secret:    {_masked(gateway[KEY_SIGNING_SECRET])}']
    for backend in data[SECTION_BACKENDS]:
        lines.append(f'Backend {backend[KEY_NAME]} ({KIND_LABELS[backend[KEY_KIND]]}) at {backend[KEY_BASE_URL]}')
        if backend.get(KEY_API_KEY):
            lines.append(f'{INDENT}API key: {_masked(backend[KEY_API_KEY])}')
        lines.append(f'{INDENT}shares: {", ".join(map(printable, backend[KEY_MODELS])) or SHARES_EVERY_MODEL}')
        if backend[KEY_KIND] == KIND_OLLAMA:
            lines.append(f'{INDENT}stays in memory: {_kept_in_words(backend.get(KEY_KEEP_LOADED))}')
            lines.append(f'{INDENT}loaded when the client starts: {"yes" if backend.get(KEY_LOAD_AT_START) else "no"}')
    return lines


def _kept_in_words(keep_loaded: str | int | float | None) -> str:
    """How long the models of a backend stay in memory, and counted from when."""
    counted = keep_loaded is not None and keep_loaded_seconds(keep_loaded) > 0
    return f'{keep_loaded_in_words(keep_loaded)}{" after a request through toptic.ai" if counted else ""}'


def _typed_keep_loaded(answer: str) -> str | int | float:
    """A time as it was typed, the way the configuration holds it: a number alone is a number of
    seconds, one without a point the whole number it is (never read through a float)."""
    if not SECONDS_TYPED.fullmatch(answer):
        return answer
    if DECIMAL_POINT in answer:
        return float(answer)
    try:
        return int(answer)
    except ValueError:
        # More digits than Python reads as a number (sys.get_int_max_str_digits): as a float it is
        # infinite, which the check refuses as longer than Ollama can count
        return float(answer)


def _masked(secret: str) -> str:
    return f'hidden ({len(secret)} characters)'


def _required_text(answer: str) -> str:
    if not answer:
        raise AnswerError('An answer is needed here.')
    if not answer.isprintable():
        raise AnswerError('That holds characters that cannot be shown; something may have gone wrong while pasting. '
                          'Enter it again.')
    return answer


def _secret_text(answer: str) -> str:
    if answer in PLACEHOLDERS:
        raise AnswerError("That is the page's placeholder, not a value. Create the value on the machine's page first.")
    return _required_text(answer)


def _gateway_address(answer: str) -> str:
    _required_text(answer)
    try:
        # plain ws:// passes here and is asked about on its own
        check_gateway_url(answer, allow_insecure=True)
    except ConfigError as ex:
        raise AnswerError(rule_in_words(ex, example=GATEWAY_EXAMPLE_URL))
    return answer


def rule_in_words(error: ConfigError, example: str) -> str:
    """Why a single answer breaks a rule of the configuration, said to somebody who never saw the file.

    ``example``: an answer that would do. A rule without words here (one of a later version) is
    said as the configuration says it.
    """
    words = RULES_IN_WORDS.get(error.reason)
    return words.format(example=example) if words else f'{error}.'


def _http_status(status: int) -> str:
    """A status as it is named to a person: with the standard phrase for it, never with the server's own."""
    try:
        return f'HTTP {status} {HTTPStatus(status).phrase}'
    except ValueError:
        return f'HTTP {status}'


def _unavailable_in_words(subject: str, problem: BackendUnavailable) -> list[str]:
    """What went wrong when ``subject`` (an address, or "backend <name>") was asked for its models."""
    detail = printable(str(problem))
    if problem.is_down:
        return [f'Nothing answers at {subject}.']
    if problem.timed_out:
        return [f'{subject} did not answer within {DISCOVERY_TIMEOUT_SECONDS:g} s.',
                'It may be busy, for example loading a model, or the address is not the right one.']
    if problem.status is None and problem.answered:
        return [f'{subject} answered, but not with its list of models ({detail}).',
                'Check the address: it is the one that serves /chat/completions and usually ends in /v1.']
    if problem.status is None:
        return [f'{subject} could not be reached ({detail}).',
                'Check that the computer it runs on is on, and the address: http or https, the name, the port.']
    status = _http_status(problem.status)
    if problem.status == HTTPStatus.UNAUTHORIZED:
        return [f'{subject} refused the request ({status}): the API key is missing or wrong.']
    if problem.status == HTTPStatus.FORBIDDEN:
        return [f'{subject} refused the request ({status}): the API key is missing, wrong, or may not list the '
                f'models.']
    if problem.status == HTTPStatus.NOT_FOUND:
        return [f'{subject} does not have what was asked for ({status}): the address is probably not the one that',
                'serves /chat/completions. It usually ends in /v1.']
    if problem.status == HTTPStatus.TOO_MANY_REQUESTS:
        return [f'{subject} is limiting requests ({status}): try again in a moment.']
    if HTTPStatus.MULTIPLE_CHOICES <= problem.status < HTTPStatus.BAD_REQUEST:
        return [f'{subject} sends requests on to another address ({status}). The client does not follow it there:',
                'enter the address the backend really has.']
    if problem.status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return [f'{subject} reported a problem of its own ({status}). It may be starting or busy: try again in a '
                f'moment.']
    return [f'{subject} refused the request ({status}). Check the address and the API key.']


def _names_shown(names: Iterable[str]) -> str:
    """Model names for a sentence: printable, and of a long list the first ones and how many more."""
    names = [printable(str(name)) for name in names]
    more = f' and {len(names) - DROPPED_NAMES_SHOWN} more' if len(names) > DROPPED_NAMES_SHOWN else ''
    return f'{", ".join(names[:DROPPED_NAMES_SHOWN])}{more}'


def note_in_words(note: DiscoveryNote) -> list[str]:
    """What discovery noted (and would log for the running client), as sentences of the conversation.

    A kind without words here (one of a later version) is said as it was logged: never lost.
    """
    backend = printable(note.backend)
    if note.kind == NOTE_DOWN:
        return [f'Backend {backend} does not answer: nothing listens at its address.']
    if note.kind in (NOTE_TIMED_OUT, NOTE_FAILED):
        lines = _unavailable_in_words(f'backend {backend}', note.problem)
        return [lines[0][0].upper() + lines[0][1:], *lines[1:]]
    if note.kind == NOTE_NAMES_REFUSED:
        return [f'{len(note.names)} model(s) of {backend} are left out: toptic.ai does not accept their names (longer '
                f'than {protocol.MODEL_NAME_MAX_LENGTH} characters, or holding characters that cannot be shown):',
                f'{INDENT}{_names_shown(note.names)}']
    if note.kind == NOTE_TOO_MANY_MODELS:
        return [f'{backend} has more models than the {protocol.MAX_MODELS_PER_BACKEND} toptic.ai accepts from one '
                f'backend. Left out:', f'{INDENT}{_names_shown(note.names)}']
    if note.kind == NOTE_REPORT_TOO_LARGE:
        return [f'The list of models is larger than toptic.ai accepts in one message. Left out of {backend}:',
                f'{INDENT}{_names_shown(note.names)}']
    if note.kind == NOTE_BACKEND_NOT_REPORTED:
        return [f'{backend} is left out: toptic.ai accepts {protocol.MAX_BACKENDS_PER_REPORT} backends from one '
                f'machine.']
    if note.kind == NOTE_DETAILS_UNREADABLE:
        status = _http_status(note.status)
        return [f'The API key may not read the model details of {backend} ({status}), so toptic.ai'
                if note.status in KEY_REFUSED_STATUSES
                else f'The model details of {backend} could not be read ({status}), so toptic.ai',
                'will not know the context sizes and prices of its models. They can be used all the same.']
    if note.kind == NOTE_ENTRY_IGNORED:
        return [f'{backend} lists a model in a form that cannot be used; it is left out:',
                f'{INDENT}{_names_shown(note.names)}']
    return [printable(note.text)]


def _optional_api_key(answer: str) -> str:
    return _required_text(answer) if answer else answer


class SetupWizard:

    def __init__(self, console: Console, discover: Discover, backend_urls: Mapping = DEFAULT_BACKEND_URLS):
        """``backend_urls``: per backend kind, the address to look at first or to suggest."""
        self._console = console
        self._discover = discover
        self._backend_urls = backend_urls
        # Notes already said: discovery makes the same one whenever it asks the same backend
        self._notes_said = set()

    def run(self, target: ConfigTarget) -> SetupPlan:
        """Ask for everything a configuration needs. Raises SetupDeclined; interruptions pass through."""
        say = self._console.say
        say('This sets up socketllama. It makes models that run on your side usable in toptic.ai:')
        say('the client connects out to toptic.ai and passes requests on to Ollama, a LiteLLM proxy or')
        say('another OpenAI-compatible server.')
        say(f'It takes {len(STEPS)} steps: {", ".join(STEPS)}.')
        say()
        say('Nothing is written until you confirm at the end. Press Ctrl-C at any time to stop.')
        if target.exists:
            say(f'{target.named} already exists. It stays as it is until you confirm replacing it at the end.')
        if target.readable_by_others:
            say()
            say('Other users of this computer can read that file. If it holds a connection token or a signing')
            say('secret, they may have been read: renew both on toptic.ai before you go on, and enter the new')
            say('ones here. The token: revoke it under Profile → API tokens, then create a new one on the')
            say("machine's page. The signing secret: create a new signing secret on the machine's page.")
        gateway = self._gateway_step()
        backends = self._backends_step(gateway[KEY_INSTANCE])
        data = {SECTION_GATEWAY: gateway, SECTION_BACKENDS: backends}
        parse_config(data, environ={})  # every answer was checked on its own; this checks them together
        return SetupPlan(data=data, keep_backup=self._confirm_step(data, target))

    def _announce_step(self, step: str) -> None:
        self._console.say()
        self._console.say(f'Step {STEPS.index(step) + FIRST_NUMBER} of {len(STEPS)}: {step}')
        self._console.say()

    # ----- step 1 -----

    def _gateway_step(self) -> dict:
        say = self._console.say
        self._announce_step(STEP_GATEWAY)
        say(f"The client needs {len(GATEWAY_TEXT_KEYS)} values from your machine's page on toptic.ai. To get there,")
        say('open My Machines in the account menu and add this computer as a machine (or open it, if you')
        say('added it before). The page shows them in a config block:')
        say(f'{INDENT}- the gateway address (url): where the client connects')
        say(f'{INDENT}- the machine name (instance): which of your machines this is')
        say(f'{INDENT}- the connection token (token): lets the client connect as you. Click "Create connection')
        say(f'{INDENT}  token" on the page first; the token is shown only once, inside the block')
        say(f'{INDENT}- the signing secret (signing_secret): proves that requests really come from toptic.ai')
        found = {}
        way = self._choose('How do you want to enter them?', [
            (WAY_PASTE, 'Paste the config block from the page'),
            (WAY_QUESTIONS, 'Answer one question at a time')], default=WAY_PASTE)
        if way == WAY_PASTE:
            found = self._read_gateway_block()
        return self._complete_gateway(found)

    def _read_gateway_block(self) -> dict:
        """The values of a pasted block; nothing if it cannot be read and the person prefers the questions."""
        say = self._console.say
        while True:
            say()
            say(f'Paste the whole block here. Then type {END_OF_BLOCK} on a line of its own and press Enter')
            say('(or press Ctrl-D on an empty line).')
            say('The block holds your token and your secret, and it stays visible in this window.')
            try:
                found = parse_gateway_block(self._read_block())
            except AnswerError as ex:
                say()
                say(str(ex))
                if self._confirm('Paste it again? Otherwise the values are asked one at a time.', default=True):
                    continue
                return {}
            taken = [GATEWAY_LABELS[key] for key in GATEWAY_TEXT_KEYS if key in found]
            say()
            say(f'Read from the block: {", ".join(taken) or "nothing that can be used"}.')
            return found

    def _read_block(self) -> str:
        """The lines typed or pasted up to the end marker, or to the end of input (Ctrl-D): that is the
        habit for "this was all", and a terminal goes on answering the questions that follow."""
        lines = []
        while True:
            try:
                line = self._console.ask(BLOCK_LINE_PROMPT).rstrip()
            except EOFError:
                return '\n'.join(lines)
            if line.upper() == END_OF_BLOCK:
                return '\n'.join(lines)
            if line.endswith(END_OF_BLOCK):
                # typed right behind a paste that did not end with a line break
                return '\n'.join([*lines, line.removesuffix(END_OF_BLOCK)])
            lines.append(line)

    def _complete_gateway(self, found: dict) -> dict:
        """The gateway values: those of the block that keep the rules, the others asked for."""
        allow_insecure = found.get(KEY_ALLOW_INSECURE, False)
        url = found.get(KEY_URL)
        while True:
            url = self._settled(url, KEY_URL, _gateway_address, [
                'The gateway address is the line url = "..." of the block. It starts with wss://'])
            if allow_insecure or urlsplit(url).scheme != INSECURE_GATEWAY_SCHEME:
                break
            self._console.say()
            self._console.say('This address is not encrypted (ws://). That is only right for a toptic.ai development '
                              'server on this computer.')
            if self._confirm('Use it anyway?', default=False):
                allow_insecure = True
                break
            url = None
        gateway = {KEY_URL: url}
        gateway[KEY_INSTANCE] = self._settled(found.get(KEY_INSTANCE), KEY_INSTANCE, _required_text, [
            'The machine name is the line instance = "..." of the block.'])
        gateway[KEY_TOKEN] = self._settled(found.get(KEY_TOKEN), KEY_TOKEN, _secret_text, [
            'The connection token is the line token = "..." of the block. If the line still reads',
            f'{TOKEN_PLACEHOLDER}, click "Create connection token" on the page: the token then',
            'appears in the block, only this once. It is ' + NOT_ECHOED + '.'], secret=True)
        gateway[KEY_SIGNING_SECRET] = self._settled(found.get(KEY_SIGNING_SECRET), KEY_SIGNING_SECRET, _secret_text, [
            'The signing secret is the line signing_secret = "..." of the block. If the line reads',
            f'{SIGNING_SECRET_PLACEHOLDER}, create a new signing secret on the page first.',
            'It is ' + NOT_ECHOED + '.'], secret=True)
        if allow_insecure:
            gateway[KEY_ALLOW_INSECURE] = True
        return gateway

    def _settled(self, given: str | None, key: str, check: Callable, where_to_find: list, secret: bool = False) -> str:
        """A gateway value that keeps the rules: ``given`` (from the block) if it does, else asked for."""
        label = GATEWAY_LABELS[key]
        if given is not None:
            try:
                return check(given)
            except AnswerError as ex:
                self._console.say()
                self._console.say(f'The {label} of the block cannot be used. {ex}')
        self._console.say()
        for line in where_to_find:
            self._console.say(line)
        return self._ask(label.capitalize(), check, secret=secret)

    # ----- step 2 -----

    def _backends_step(self, instance: str) -> list:
        say = self._console.say
        self._announce_step(STEP_BACKENDS)
        say('A backend is the program that runs your models: Ollama, a LiteLLM proxy or another')
        say('OpenAI-compatible server. Only the models you choose here can be used through toptic.ai. They')
        say(f'appear in its model picker as {instance}/<backend>/<model>, for you and for everybody you')
        say("share the machine with on its page. toptic.ai can reach nothing else on your side.")
        backends = []
        first = self._configure_backend(KIND_OLLAMA, FIRST_BACKEND_NAME, self._backend_urls[KIND_OLLAMA], api_key='',
                                        asks_api_key=False)
        if first:
            backends.append(first)
        while len(backends) < protocol.MAX_BACKENDS_PER_REPORT:
            kind = self._choose_backend_to_add(has_backends=bool(backends))
            if kind == ACTION_DONE:
                return backends
            added = self._add_backend(kind, backends)
            if added:
                backends.append(added)
        say()
        say(f'That makes {len(backends)} backends, the most toptic.ai accepts from one machine.')
        return backends

    def _choose_backend_to_add(self, has_backends: bool) -> str:
        kinds = [(KIND_LITELLM, 'Add a LiteLLM proxy'),
                 (KIND_OPENAI, 'Add another OpenAI-compatible server (vLLM, llama.cpp server, LM Studio, ...)'),
                 (KIND_OLLAMA, 'Add an Ollama at another address')]
        if has_backends:
            return self._choose('Do you want to add another backend?',
                                [*kinds, (ACTION_DONE, 'No, continue')], default=ACTION_DONE)
        self._console.say()
        self._console.say('No backend is set up yet, and at least one that shares models is needed to finish.')
        return self._choose('Which one do you want to add? (Ctrl-C stops setup without writing anything.)', kinds)

    def _add_backend(self, kind: str, backends: list) -> dict | None:
        say = self._console.say
        taken_names = [backend[KEY_NAME] for backend in backends]
        taken_urls = [backend[KEY_BASE_URL] for backend in backends]
        default_name = self._free_default_name(kind, taken_names)

        def free_name(answer: str) -> str:
            _required_text(answer)
            try:
                check_backend_name(answer)
            except ConfigError as ex:
                raise AnswerError(rule_in_words(ex, example=default_name))
            if answer in taken_names:
                raise AnswerError(f'There is already a backend called {answer}. Choose another name.')
            return answer

        say()
        say('The name tells your backends apart; it is part of how the models appear in toptic.ai. Any name of')
        say(f'up to {protocol.BACKEND_NAME_MAX_LENGTH} characters without "{BACKEND_NAME_SEPARATOR}" will do.')
        name = self._ask('Name', free_name, default=default_name)
        say()
        say('The address is the one that serves /chat/completions; it usually ends in /v1.')
        default_url = self._backend_urls.get(kind)
        base_url = self._ask_backend_address(kind, name, None if default_url in taken_urls else default_url)
        say()
        say('The API key is sent to this backend only, never to toptic.ai.')
        return self._configure_backend(kind, name, base_url, self._ask_api_key(), asks_api_key=True)

    @staticmethod
    def _free_default_name(kind: str, taken_names: list) -> str:
        numbered = (f'{kind}-{number}' for number in itertools.count(SECOND_NUMBER))
        return next(name for name in itertools.chain([kind], numbered) if name not in taken_names)

    def _ask_backend_address(self, kind: str, name: str, default: str | None) -> str:
        def address(answer: str) -> str:
            _required_text(answer)
            try:
                return normalized_backend_url(name, answer)
            except ConfigError as ex:
                raise AnswerError(rule_in_words(ex, example=EXAMPLE_BACKEND_URLS[kind]))
        return self._ask('Address', address, default=default)

    def _ask_api_key(self) -> str:
        return self._ask(f'API key ({NOT_ECHOED}; press Enter if the backend needs none)', _optional_api_key,
                         secret=True)

    def _configure_backend(self, kind: str, name: str, base_url: str, api_key: str, asks_api_key: bool) -> dict | None:
        """Ask the backend for its models and the person which to share; None if they skip it.

        ``asks_api_key``: whether a key was asked for before. One can always be entered once
        the backend refuses a request for the lack of one.
        """
        say = self._console.say
        label = KIND_LABELS[kind]
        while True:
            say()
            say(f'Asking {label} at {base_url} for its models ...')
            try:
                found = self._discover(Backend(name=name, kind=kind, base_url=base_url, api_key=api_key))
            except BackendUnavailable as ex:
                self._explain_unavailable(kind, base_url, ex)
                enter_key = (ACTION_API_KEY, 'Enter the API key again' if api_key else 'Enter an API key')
                actions = [(ACTION_RETRY, 'Try again'), (ACTION_ADDRESS, 'Enter another address'),
                           *([enter_key] if asks_api_key or ex.status in KEY_REFUSED_STATUSES else []),
                           (ACTION_SKIP, 'Skip this backend')]
            else:
                names = found.names
                self.say_notes(found.notes)
                if names:
                    models = self._choose_models(kind, names)
                    return None if models is None else self._settled_backend(kind, name, base_url, api_key, models)
                say(f'{label} answers, but it {"has no chat model" if kind == KIND_OLLAMA else "lists no model"} yet.')
                if kind == KIND_OLLAMA:
                    say('Pull one in another terminal, for example: ollama pull llama3.1:8b')
                actions = [(ACTION_RETRY, 'Look again'),
                           (ACTION_SHARE_FUTURE, 'Share every model it offers later, without asking again'),
                           (ACTION_SKIP, 'Skip this backend')]
            action = self._choose('What do you want to do?', actions)
            if action == ACTION_SKIP:
                return None
            if action == ACTION_ADDRESS:
                base_url = self._ask_backend_address(kind, name, default=None)
            elif action == ACTION_API_KEY:
                api_key = self._ask_api_key()
            elif action == ACTION_SHARE_FUTURE and self._confirm_future_models():
                return self._settled_backend(kind, name, base_url, api_key, models=[])

    def _explain_unavailable(self, kind: str, base_url: str, problem: BackendUnavailable) -> None:
        """Say what went wrong at ``base_url``, and what this kind of backend needs done about it."""
        say = self._console.say
        for line in _unavailable_in_words(base_url, problem):
            say(line)
        if problem.is_down and kind == KIND_OLLAMA:
            say('If Ollama is installed, start it: open the Ollama app, or run "ollama serve" in another')
            say('terminal. "ollama list" then shows its models. If it runs elsewhere, enter that address.')
        elif problem.is_down:
            say('Check that the server is running and that the address is right.')

    def say_notes(self, notes: Iterable[DiscoveryNote]) -> None:
        """Say what discovery noted while a backend was asked (see note_in_words), each note once."""
        for note in notes:
            if note not in self._notes_said:
                self._notes_said.add(note)
                for line in note_in_words(note):
                    self._console.say(line)

    def _choose_models(self, kind: str, names: list) -> list | None:
        """The models to share: names from the list, [] for everything including later ones, None for none."""
        say = self._console.say
        # Ollama's discovery leaves out what cannot chat; the other kinds list what the server names
        found = f'has {len(names)} chat model(s)' if kind == KIND_OLLAMA else f'lists {len(names)} model(s)'
        say(f'{KIND_LABELS[kind]} answers. It {found}:')
        for number, name in enumerate(names, start=FIRST_NUMBER):
            say(f'{INDENT}{number}) {printable(name)}')
        say()
        say('Which of them do you want to share?')
        say(f'{INDENT}numbers  the models with these numbers, for example {SELECTION_EXAMPLE}')
        say(f'{INDENT}{SELECT_ALL}      every model listed above. A model you pull later is not shared')
        say(f'{INDENT}         until you add it')
        say(f'{INDENT}{SELECT_FUTURE}   every model listed above and every model you pull later, without asking again')
        say(f'{INDENT}{SELECT_NONE}     nothing: skip this backend')
        while True:
            try:
                selection = parse_selection(self._console.ask('Models to share: '), len(names))
            except AnswerError as ex:
                say(str(ex))
                continue
            if selection.is_empty:
                return None
            if not selection.includes_future:
                return [names[position] for position in selection.positions]
            if self._confirm_future_models():
                return []

    def _confirm_future_models(self) -> bool:
        self._console.say(FUTURE_MODELS_WARNING)
        return self._confirm('Share them all?', default=False)

    def _settled_backend(self, kind: str, name: str, base_url: str, api_key: str, models: list[str]) -> dict:
        """The backend as the configuration holds it; of Ollama, with what is decided about its memory."""
        entry = {KEY_NAME: name, KEY_KIND: kind, KEY_BASE_URL: base_url,
                 **({KEY_API_KEY: api_key} if api_key else {}), KEY_MODELS: models}
        return {**entry, **self._memory_settings(name, models)} if kind == KIND_OLLAMA else entry

    def _memory_settings(self, name: str, models: list[str]) -> dict:
        """Ask how long Ollama keeps this backend's models in memory and whether they are loaded at
        start; only what departs from Ollama's own ways is returned (and written)."""
        say = self._console.say
        settings = {}
        say()
        say(f'Ollama takes a model out of memory {OLLAMA_DEFAULT_KEEP} after its last request, unless it is set up')
        say('otherwise. The next prompt then waits until the model is loaded again. The client can tell Ollama')
        say('to keep the models used through toptic.ai for longer. A model in memory occupies RAM (or VRAM, on')
        say('a graphics card) for all that time. This is applied after requests through toptic.ai, but Ollama')
        say('has one copy of each model: what you choose here also decides when a model you use yourself,')
        say('outside toptic.ai, leaves memory, and how much memory stays occupied.')
        if self._confirm(f"Keep models in memory longer than Ollama's {OLLAMA_DEFAULT_KEEP}?", default=False):
            settings[KEY_KEEP_LOADED] = self._ask_keep_loaded(name)
        say()
        if not models:
            say('Loading models when the client starts is not offered for this backend: it shares whatever')
            say('models it has, and loading all of them could take more memory than there is.')
            return settings
        say('The client can also load the chosen models into memory when it starts, so that the first prompt')
        say('does not wait for that. They occupy their memory right away. Loading runs in the background')
        say('once the client is connected; a prompt that arrives meanwhile waits for its model.')
        if KEY_KEEP_LOADED not in settings:
            say(f'Without a longer time, Ollama takes them out again after its {OLLAMA_DEFAULT_KEEP} without a')
            say('request.')
        if len(models) > OLLAMA_DEFAULT_MAX_LOADED_MODELS:
            say(f'You chose {len(models)} models. Unless OLLAMA_MAX_LOADED_MODELS is set on its server, Ollama keeps')
            say(f'at most {OLLAMA_DEFAULT_MAX_LOADED_MODELS} in memory for each graphics card '
                f'({OLLAMA_DEFAULT_MAX_LOADED_MODELS} without one): loading more takes out the ones loaded first.')
        if self._confirm('Load the chosen models when the client starts?', default=False):
            settings[KEY_LOAD_AT_START] = True
        return settings

    def _ask_keep_loaded(self, name: str) -> str | int | float:
        def time(answer: str) -> str | int | float:
            _required_text(answer)
            keep_loaded = _typed_keep_loaded(answer)
            try:
                check_keep_loaded(name, keep_loaded)
            except ConfigError as ex:
                raise AnswerError(rule_in_words(ex, example=KEEP_LOADED_EXAMPLE))
            # Asked after a yes to "longer?": a shorter time (zero unloads at once) is no answer to it
            if 0 <= keep_loaded_seconds(keep_loaded) <= OLLAMA_DEFAULT_KEEP_ALIVE_SECONDS:
                raise AnswerError(NOT_LONGER_THAN_DEFAULT)
            return keep_loaded

        choice = self._choose('How long should a model stay in memory after a request?',
                              [*KEEP_LOADED_CHOICES, (ACTION_OTHER_TIME, 'Another time')])
        if choice != ACTION_OTHER_TIME:
            return choice
        return self._ask(f'Time (for example {KEEP_LOADED_EXAMPLE}, 8h or 1h30m; a number alone is seconds)', time)

    # ----- step 3 -----

    def _confirm_step(self, data: dict, target: ConfigTarget) -> bool:
        """Show what will be written and ask whether to; returns whether a file in the way is kept."""
        say = self._console.say
        self._announce_step(STEP_SAVING)
        say(f'This will be written to {target.named}:')
        for line in summary_lines(data):
            say(f'{INDENT}{line}')
        say()
        if target.is_link:
            say(self._link_in_words(target))
        say('The file holds your token and your secret, so it is written for you alone: other users of this')
        say(f'computer cannot read it (permissions {OWNER_ONLY_PERMISSIONS:o}).')
        if not target.exists:
            if not self._confirm('Write it?', default=True):
                raise SetupDeclined
            return True
        say()
        say(self._link_in_words(target) if target.is_link else f'{target.named} already exists.')
        if not self._confirm('Replace it?', default=False):
            raise SetupDeclined
        return self._confirm('Keep the old file as a backup next to it? Only you can read the backup.', default=True)

    @staticmethod
    def _link_in_words(target: ConfigTarget) -> str:
        return (f'{target.named} is a link to {target.file}; {target.file} will be '
                f'{"replaced" if target.exists else "written"}.')

    def backup_failed(self, target: ConfigTarget, reason: str) -> str:
        """The backup could not be made and nothing was changed: ask what to do (one of BACKUP_*)."""
        say = self._console.say
        say()
        say(f'The backup of {target.file} could not be made ({printable(reason)}).')
        say('Nothing was changed: the old file is in place, and your answers are kept.')
        return self._choose('What do you want to do?', [
            (BACKUP_RETRY, 'Try the backup again'),
            (BACKUP_SKIP, 'Write the new file without a backup: the old file is then gone'),
            (BACKUP_STOP, 'Stop without writing anything')])

    def offer_start(self, command: str) -> bool:
        """After the file is written and checked: name the command that starts the client, offer to run it."""
        say = self._console.say
        say()
        say('To start the client, run:')
        say(f'{INDENT}{command}')
        say("It stays connected until you stop it with Ctrl-C. While it runs, the machine's page on toptic.ai")
        say('shows "Connected" and your models are in the model picker.')
        return self._confirm('Start it now?', default=True)

    # ----- questions -----

    def _ask(self, label: str, check: Callable, default: str | None = None, secret: bool = False) -> str:
        """Ask until ``check`` takes the answer; it returns the value to use or raises AnswerError."""
        prompt = f'{label} [{default}]: ' if default else f'{label}: '
        read = self._console.ask_secret if secret else self._console.ask
        while True:
            answer = read(prompt).strip()
            try:
                return check(answer or default or '')
            except AnswerError as ex:
                self._console.say(str(ex))

    def _confirm(self, question: str, default: bool) -> bool:
        prompt = f'{question} [{YES_BY_DEFAULT if default else NO_BY_DEFAULT}]: '
        while True:
            answer = self._console.ask(prompt).strip().lower()
            if not answer:
                return default
            if answer in YES_ANSWERS + NO_ANSWERS:
                return answer in YES_ANSWERS
            self._console.say('Answer y or n.')

    def _choose(self, question: str, choices: list, default: str | None = None) -> str:
        """Show a numbered menu of (value, text) and return the value picked."""
        say = self._console.say
        say()
        say(question)
        for number, (_, text) in enumerate(choices, start=FIRST_NUMBER):
            say(f'{INDENT}{number}) {text}')
        values = [value for value, _ in choices]
        default_number = values.index(default) + FIRST_NUMBER if default is not None else None
        prompt = f'Your choice [{default_number}]: ' if default_number else 'Your choice: '
        while True:
            answer = self._console.ask(prompt).strip()
            if not answer and default is not None:
                return default
            number = _number_in_list(answer, len(choices))
            if number is not None:
                return values[number - FIRST_NUMBER]
            say(f'Enter a number from {FIRST_NUMBER} to {len(choices)}.')
