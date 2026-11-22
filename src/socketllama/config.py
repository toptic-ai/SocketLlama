"""Configuration file of socketllama (TOML).

The backends listed here are the complete allow-list: the gateway names a backend, it can
never supply a URL. Secrets may be given inline or through environment variables
(``token_env`` / ``signing_secret_env`` / ``api_key_env``).
"""
import dataclasses
import math
import os
import re
import stat
import tomllib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

from .protocol import ARABIC_LETTER_MARK, BACKEND_NAME_MAX_LENGTH, FIRST_BIDI_EMBEDDING_CONTROL, \
    FIRST_BIDI_ISOLATE_CONTROL, LAST_BIDI_EMBEDDING_CONTROL, LAST_BIDI_ISOLATE_CONTROL, LEFT_TO_RIGHT_MARK, \
    MAX_MODEL_DESCRIPTION_CHARS, MAX_PRICE_PER_UNIT, MIN_PRICE_PER_UNIT, RIGHT_TO_LEFT_MARK, refused_in_description
from .toml_writer import KEY_VALUE_SEPARATOR, toml_key
from .units import MICROSECONDS_PER_SECOND, MILLISECONDS_PER_SECOND, NANOSECONDS_PER_SECOND, SECONDS_PER_HOUR, \
    SECONDS_PER_MINUTE

KIND_OLLAMA = 'ollama'
KIND_LITELLM = 'litellm'
KIND_OPENAI = 'openai'
BACKEND_KINDS = (KIND_OLLAMA, KIND_LITELLM, KIND_OPENAI)

SECURE_GATEWAY_SCHEME = 'wss'
INSECURE_GATEWAY_SCHEME = 'ws'
BACKEND_SCHEMES = ('http', 'https')
BACKEND_NAME_SEPARATOR = '/'

DEFAULT_PING_INTERVAL_SECONDS = 30.0
DEFAULT_PONG_TIMEOUT_SECONDS = 20.0
DEFAULT_RECONNECT_MIN_SECONDS = 1.0
DEFAULT_RECONNECT_MAX_SECONDS = 60.0
DEFAULT_MAX_CONCURRENT_REQUESTS = 4
MIN_CONCURRENT_REQUESTS = 1
DEFAULT_BACKEND_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_BACKEND_READ_TIMEOUT_SECONDS = 600.0


# Which rule a single value breaks (ConfigError.reason), for a caller that says it in its own
# words: setup asks for one value at a time and does not speak of keys of a file
REASON_GATEWAY_URL_MALFORMED = 'gateway_url_malformed'
REASON_GATEWAY_URL_SCHEME = 'gateway_url_scheme'
REASON_BACKEND_NAME_CHARACTERS = 'backend_name_characters'
REASON_BACKEND_NAME_TOO_LONG = 'backend_name_too_long'
REASON_BACKEND_URL_MALFORMED = 'backend_url_malformed'
REASON_BACKEND_URL_CREDENTIALS = 'backend_url_credentials'
# Of the gateway's address and a backend's alike
REASON_ADDRESS_PORT = 'address_port'
REASON_ADDRESS_BRACKETS = 'address_brackets'

REASON_KEEP_LOADED = 'keep_loaded'
REASON_KEEP_LOADED_TOO_LONG = 'keep_loaded_too_long'

# Keys of an Ollama backend that decide about its memory
KEY_KEEP_LOADED = 'keep_loaded'
KEY_LOAD_AT_START = 'load_at_start'
# The table of a backend with the price of its models: one line per model, its key the model's name
KEY_PRICES = 'prices_per_1m_tokens'
# The two numbers of a price, USD per 1M tokens of the prompt and of the answer
KEY_PRICE_INPUT = 'input'
KEY_PRICE_OUTPUT = 'output'
# The header of a table of a backend is [backends.<key>]: [backends.prices_per_1m_tokens]
BACKEND_TABLE_PREFIX = 'backends.'
PRICE_FORM = f'{{ {KEY_PRICE_INPUT} = <number>, {KEY_PRICE_OUTPUT} = <number> }} in USD per 1M tokens'
# The table of a backend with a description of its models: one line per model, its key the model's name
KEY_DESCRIPTIONS = 'descriptions'
DESCRIPTION_FORM = f'"<one line of text, at most {MAX_MODEL_DESCRIPTION_CHARS} characters>"'
# How a message names a character without showing it: U+202E
CODE_POINT_FORM = 'U+{:04X}'
# The rule for a description, as the refusal of one states it (protocol.refused_in_description)
DESCRIPTION_RULE = (
    f'one line of 1 to {MAX_MODEL_DESCRIPTION_CHARS} characters, without the blanks around it, and without '
    f'control characters (a tab, a line break), separators of lines or paragraphs, or the controls of writing '
    f'direction ({", ".join(map(CODE_POINT_FORM.format, (LEFT_TO_RIGHT_MARK, RIGHT_TO_LEFT_MARK, ARABIC_LETTER_MARK)))}'
    f', {CODE_POINT_FORM.format(FIRST_BIDI_EMBEDDING_CONTROL)} to {CODE_POINT_FORM.format(LAST_BIDI_EMBEDDING_CONTROL)}'
    f', {CODE_POINT_FORM.format(FIRST_BIDI_ISOLATE_CONTROL)} to {CODE_POINT_FORM.format(LAST_BIDI_ISOLATE_CONTROL)})')
# What a message says of a character that has no name in Unicode: what it is, by its category
NAMELESS_CHARACTERS = {'Cc': 'a control character', 'Cs': 'half of a UTF-16 pair'}
NAMELESS_CHARACTER = 'without a name'
# The units of a time as Ollama takes it (Go's duration syntax: "30m", "1h30m", "1.5h"), in seconds.
# Exact (Decimal), so that a time is judged as written, not as a float rounds it
DURATION_UNITS = {'ns': Decimal(1) / NANOSECONDS_PER_SECOND, 'us': Decimal(1) / MICROSECONDS_PER_SECOND,
                  'µs': Decimal(1) / MICROSECONDS_PER_SECOND, 'μs': Decimal(1) / MICROSECONDS_PER_SECOND,
                  'ms': Decimal(1) / MILLISECONDS_PER_SECOND, 's': Decimal(1), 'm': Decimal(SECONDS_PER_MINUTE),
                  'h': Decimal(SECONDS_PER_HOUR)}
# A number and its unit. Written so that a text is matched in one pass, however long it is: digits
# belong to one place only (before the point, or after it). Digits are 0-9 alone (re.ASCII), as Go
# reads them: \d would also take others ("５m"), which Ollama refuses
DURATION_PART = re.compile(rf'(\d+(?:\.\d*)?|\.\d+)({"|".join(DURATION_UNITS)})', re.ASCII)
# The longest time Ollama can count. Go keeps a duration as a signed 64-bit number of nanoseconds;
# a number of seconds Ollama multiplies by a second (beyond this it would overflow unnoticed), a text
# Go refuses beyond it. In whole seconds, the same limit for both; how long either way, for negative
# times too (any negative one means until Ollama stops: -1 says that)
GO_MAX_DURATION_NANOSECONDS = 2 ** 63 - 1
MAX_KEEP_LOADED_SECONDS = GO_MAX_DURATION_NANOSECONDS // NANOSECONDS_PER_SECOND
MAX_KEEP_LOADED_HOURS = MAX_KEEP_LOADED_SECONDS // SECONDS_PER_HOUR
MINUS = '-'
PLUS = '+'
# The one time that needs no unit
ZERO_DURATION = '0'
KEEP_LOADED_FORMS = 'a time like "30m" or "2h", a number of seconds, or -1 for as long as Ollama runs'

# The port numbers there are
MIN_PORT = 0
MAX_PORT = 65535
# An IPv6 address in a URL stands between these
IPV6_BRACKETS = '[]'


class ConfigError(ValueError):
    """The configuration file is missing something or contradicts itself.

    The message is for somebody who edits the file; ``reason`` (one of REASON_*, or None)
    names the rule a single value breaks.
    """

    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ModelPrice:
    """What a model costs, in USD per 1M tokens: of the prompt (input) and of the answer (output)."""
    input: float
    output: float

    def same_as(self, other: 'ModelPrice | None') -> bool:
        """Whether ``other`` is this price. Compared as toptic.ai compares prices (math.isclose), so that a
        price both sides hold is never taken for a different one because of how a float was rounded."""
        return other is not None and math.isclose(self.input, other.input) and math.isclose(self.output,
                                                                                            other.output)

    def as_table(self) -> dict:
        """The price as config.toml holds it (and as the wire carries it, see protocol.PRICE_*_FIELD)."""
        return {KEY_PRICE_INPUT: self.input, KEY_PRICE_OUTPUT: self.output}


@dataclass(frozen=True)
class Backend:
    name: str
    kind: str
    base_url: str
    api_key: str = field(default='', repr=False)
    # Only these models are reported and relayed; an empty list (written out in the
    # configuration, never implied) shares every model the backend lists
    models: tuple = ()
    connect_timeout: float = DEFAULT_BACKEND_CONNECT_TIMEOUT_SECONDS
    read_timeout: float = DEFAULT_BACKEND_READ_TIMEOUT_SECONDS
    # Context window to report for every model of this backend (None: ask the backend)
    context_length: int | None = None
    # Ollama only. How long a model stays in memory after a request through the tunnel, as Ollama
    # takes it (its keep_alive: "30m", seconds, negative for as long as it runs); None: the
    # client says nothing and Ollama's own setting applies
    keep_loaded: str | int | float | None = None
    # Ollama only. Whether the client loads the models listed when it starts
    load_at_start: bool = False
    # {model name: ModelPrice} of the backend's price table, as the file was loaded
    prices: dict = field(default_factory=dict, hash=False)
    # {model name: text} of the backend's table of descriptions, as the file was loaded
    descriptions: dict = field(default_factory=dict, hash=False)

    @property
    def shares_all_models(self) -> bool:
        return not self.models

    def allows_model(self, model_name) -> bool:
        return self.shares_all_models or model_name in self.models

    def first_answer_timeout(self, model_loading_timeout: float | None) -> float:
        """For how many seconds the client waits for this backend to begin an answer while a model
        loads: its read_timeout, or what the gateway said it waits then (``model_loading_timeout``;
        None: it said nothing, or no model is loading) if that is longer. Giving up sooner than
        the gateway would end a load the gateway still waits for."""
        return self.read_timeout if model_loading_timeout is None else max(self.read_timeout, model_loading_timeout)


# Permission bits that let anybody but the owner read or change a file
NOT_OWNER_PERMISSIONS = stat.S_IRWXG | stat.S_IRWXO
# Those of them that let somebody else read it
READ_BY_OTHERS_PERMISSIONS = stat.S_IRGRP | stat.S_IROTH
# Mode of a file only its owner can read and change (chmod 600)
OWNER_ONLY_PERMISSIONS = stat.S_IRUSR | stat.S_IWUSR


@dataclass(frozen=True)
class Config:
    gateway_url: str
    instance: str
    token: str = field(repr=False)
    signing_secret: str = field(repr=False)
    backends: dict = field(default_factory=dict)
    ping_interval: float = DEFAULT_PING_INTERVAL_SECONDS
    pong_timeout: float = DEFAULT_PONG_TIMEOUT_SECONDS
    reconnect_min: float = DEFAULT_RECONNECT_MIN_SECONDS
    reconnect_max: float = DEFAULT_RECONNECT_MAX_SECONDS
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    # The file the configuration was read from (load_config); None: it was not read from one
    path: Path | None = None


def load_config(path) -> Config:
    path = Path(path)
    try:
        with path.open('rb') as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f'Configuration file not found: {path}')
    except tomllib.TOMLDecodeError as ex:
        raise ConfigError(f'{path} is not valid TOML: {ex}')
    inline_secrets = []
    config = parse_config(data, inline_secrets=inline_secrets)
    if inline_secrets and path.stat().st_mode & NOT_OWNER_PERMISSIONS:
        raise ConfigError(f'{path} holds {", ".join(inline_secrets)} but other users may read it: '
                          f'run chmod 600 {path}, or use the *_env options instead')
    return dataclasses.replace(config, path=path)


def parse_config(data: dict, environ=None, inline_secrets: list | None = None) -> Config:
    """Validate a parsed configuration file. Secrets written into it are added to ``inline_secrets``."""
    environ = os.environ if environ is None else environ
    inline_secrets = [] if inline_secrets is None else inline_secrets
    gateway = _table(data, 'gateway')
    client = data.get('client') or {}
    if not isinstance(client, dict):
        raise ConfigError('[client] must be a table')

    gateway_url = _required_str(gateway, 'url', 'gateway')
    check_gateway_url(gateway_url, allow_insecure=bool(gateway.get('allow_insecure', False)))

    backends = {}
    for entry in data.get('backends') or []:
        backend = _parse_backend(entry, environ, inline_secrets)
        if backend.name in backends:
            raise ConfigError(f'Backend name {backend.name!r} is used twice')
        backends[backend.name] = backend
    if not backends:
        raise ConfigError('At least one [[backends]] entry is required')

    config = Config(
        gateway_url=gateway_url,
        instance=_required_str(gateway, 'instance', 'gateway'),
        token=_secret(gateway, 'token', 'gateway', environ, inline_secrets),
        signing_secret=_secret(gateway, 'signing_secret', 'gateway', environ, inline_secrets),
        backends=backends,
        ping_interval=_positive(client, 'ping_interval', DEFAULT_PING_INTERVAL_SECONDS),
        pong_timeout=_positive(client, 'pong_timeout', DEFAULT_PONG_TIMEOUT_SECONDS),
        reconnect_min=_positive(client, 'reconnect_min', DEFAULT_RECONNECT_MIN_SECONDS),
        reconnect_max=_positive(client, 'reconnect_max', DEFAULT_RECONNECT_MAX_SECONDS),
        max_concurrent_requests=_whole_number(client, 'max_concurrent_requests', DEFAULT_MAX_CONCURRENT_REQUESTS,
                                              MIN_CONCURRENT_REQUESTS, 'client'),
    )
    if config.reconnect_max < config.reconnect_min:
        raise ConfigError('client.reconnect_max must not be smaller than client.reconnect_min')
    return config


def _parse_backend(entry, environ, inline_secrets: list) -> Backend:
    if not isinstance(entry, dict):
        raise ConfigError('Every [[backends]] entry must be a table')
    name = _required_str(entry, 'name', 'backends')
    check_backend_name(name)
    kind = _required_str(entry, 'kind', f'backend {name}')
    if kind not in BACKEND_KINDS:
        raise ConfigError(f'Backend {name}: kind must be one of {", ".join(BACKEND_KINDS)}')
    base_url = normalized_backend_url(name, _required_str(entry, 'base_url', f'backend {name}'))
    if 'models' not in entry:
        # Sharing a machine is opt-in per model: pulling a new model must not publish it unasked
        raise ConfigError(f'Backend {name}: list the models to share, e.g. models = ["llama3.1:8b"] '
                          f'(models = [] shares every model the backend offers, including ones pulled later)')
    models = entry['models']
    if not isinstance(models, list) or not all(isinstance(model, str) and model for model in models):
        raise ConfigError(f'Backend {name}: models must be a list of model names')
    keep_loaded, load_at_start = _memory_settings(entry, name, kind, models)
    prices = _model_table(entry, name, models, KEY_PRICES, 'price', 'prices', PRICE_FORM, parse_price)
    descriptions = _model_table(entry, name, models, KEY_DESCRIPTIONS, 'description', 'describes', DESCRIPTION_FORM,
                                parse_description)
    return Backend(
        name=name, kind=kind, base_url=base_url,
        api_key=_secret(entry, 'api_key', f'backend {name}', environ, inline_secrets, required=False),
        models=tuple(models),
        connect_timeout=_positive(entry, 'connect_timeout', DEFAULT_BACKEND_CONNECT_TIMEOUT_SECONDS),
        read_timeout=_positive(entry, 'read_timeout', DEFAULT_BACKEND_READ_TIMEOUT_SECONDS),
        context_length=_optional_positive_int(entry, 'context_length', f'backend {name}'),
        keep_loaded=keep_loaded, load_at_start=load_at_start, prices=prices, descriptions=descriptions,
    )


def _model_table(entry: dict, name: str, models: list[str], key: str, noun: str, verb: str, form: str,
                 parse: Callable[[object, str], object]) -> dict:
    """{model name: value} of a backend's table with one line per model: its prices (KEY_PRICES) or its
    descriptions (KEY_DESCRIPTIONS); none without the table. Every kind of backend may have one. ``noun``
    and ``verb`` say what a line is and does ("price", "prices"), ``form`` how its value is written, and
    ``parse`` reads the value. A name the backend does not share is refused: it is a typo, its line would do
    nothing."""
    if key not in entry:
        return {}
    table = entry[key]
    if not isinstance(table, dict):
        raise ConfigError(f'Backend {name}: {key} must be a table, [{BACKEND_TABLE_PREFIX}{key}] below '
                          f'the backend with one line per model: "<model>" = {form}')
    values = {}
    for model, value in table.items():
        if models and model not in models:
            # Also what is left when the owner takes a model out of models after the client wrote its line
            raise ConfigError(f'Backend {name}: the line {toml_key(model)}{KEY_VALUE_SEPARATOR}... under '
                              f'[{BACKEND_TABLE_PREFIX}{key}] {verb} a model that is not among its models. If '
                              f'you no longer share that model, simply delete the line; if you do, check the name '
                              f'or add it to models')
        values[model] = parse(value, f'Backend {name}: the {noun} of {model!r}')
    return values


def parse_description(value: object, where: str) -> str:
    """``value`` as the description of a model, without the blanks around it; a ConfigError naming ``where``
    unless it keeps to the rule the gateway keeps it to too (DESCRIPTION_RULE, see protocol)."""
    if not isinstance(value, str):
        raise ConfigError(f'{where} must be {DESCRIPTION_FORM}, not {value!r}')
    description = value.strip()
    if not description:
        raise ConfigError(f'{where} must be {DESCRIPTION_FORM}, not empty; to have none, leave the line out')
    if len(description) > MAX_MODEL_DESCRIPTION_CHARS:
        raise ConfigError(f'{where} has {len(description)} characters; it can have at most '
                          f'{MAX_MODEL_DESCRIPTION_CHARS}')
    refused = refused_in_description(description)
    if refused is not None:
        raise ConfigError(f'{where} holds {_code_point(refused)}, which a description may not: it must be '
                          f'{DESCRIPTION_RULE}')
    return description


def _code_point(character: str) -> str:
    """"U+202E (RIGHT-TO-LEFT OVERRIDE)", "U+0009 (a control character)": a character as a message names it,
    without showing it."""
    name = unicodedata.name(character, '') or NAMELESS_CHARACTERS.get(unicodedata.category(character),
                                                                      NAMELESS_CHARACTER)
    return f'{CODE_POINT_FORM.format(ord(character))} ({name})'


def parse_price(value: object, where: str) -> ModelPrice:
    """``value`` as a price, {input = <number>, output = <number>}; a ConfigError naming ``where`` unless both
    numbers are there, nothing else is, and both are prices the gateway takes (protocol.*_PRICE_PER_UNIT)."""
    if not isinstance(value, dict) or set(value) != {KEY_PRICE_INPUT, KEY_PRICE_OUTPUT}:
        raise ConfigError(f'{where} must be {PRICE_FORM}')
    return ModelPrice(input=_price_number(value[KEY_PRICE_INPUT], f'{where}: {KEY_PRICE_INPUT}'),
                      output=_price_number(value[KEY_PRICE_OUTPUT], f'{where}: {KEY_PRICE_OUTPUT}'))


def _price_number(value: object, where: str) -> float:
    # A whole number is compared as it is: a huge one would not even become a float
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            (isinstance(value, float) and not math.isfinite(value)) or \
            not MIN_PRICE_PER_UNIT <= value <= MAX_PRICE_PER_UNIT:
        raise ConfigError(f'{where} must be a number from {MIN_PRICE_PER_UNIT:g} to {MAX_PRICE_PER_UNIT:g} (USD per '
                          f'1M tokens), not {value!r}')
    # -0.0 is taken as the 0 it is, not written or sent with its sign
    return abs(float(value))


def _memory_settings(entry: dict, name: str, kind: str, models: list[str]) -> tuple[str | int | float | None, bool]:
    """(keep_loaded, load_at_start) of a backend. Only Ollama loads and unloads models on its own:
    on another kind the settings would do nothing, so they are refused there instead of ignored."""
    for key in (KEY_KEEP_LOADED, KEY_LOAD_AT_START):
        if key in entry and kind != KIND_OLLAMA:
            raise ConfigError(f'Backend {name}: {key} is only for backends of kind {KIND_OLLAMA} (it is about how '
                              f'long Ollama keeps a model in memory)')
    keep_loaded = entry.get(KEY_KEEP_LOADED)
    if KEY_KEEP_LOADED in entry:
        check_keep_loaded(name, keep_loaded)
    load_at_start = entry.get(KEY_LOAD_AT_START, False)
    if not isinstance(load_at_start, bool):
        raise ConfigError(f'Backend {name}: {KEY_LOAD_AT_START} must be true or false')
    if load_at_start and not models:
        # models = [] shares whatever the server has: all of it at once may not fit into its memory
        raise ConfigError(f'Backend {name}: {KEY_LOAD_AT_START} needs the models listed (models = ["..."]): with '
                          f'models = [] the client would load every model the server has, which may not fit into '
                          f'its memory')
    if load_at_start and keep_loaded is not None and keep_loaded_seconds(keep_loaded) == 0:
        raise ConfigError(f'Backend {name}: {KEY_LOAD_AT_START} with {KEY_KEEP_LOADED} = {keep_loaded!r} would load '
                          f'every model and unload it at once; give a longer time or leave one of the two out')
    return keep_loaded, load_at_start


class KeepLoadedTooLong(ValueError):
    """A keep_loaded value of a form Ollama takes, but longer than it can count (MAX_KEEP_LOADED_SECONDS)."""


def check_keep_loaded(name: str, value: object) -> None:
    """``value`` as keep_loaded of the backend called ``name``: ConfigError unless Ollama would take it."""
    try:
        keep_loaded_seconds(value)
    except KeepLoadedTooLong:
        raise ConfigError(f'Backend {name}: {KEY_KEEP_LOADED} is longer than Ollama can count: it can be at most '
                          f'{MAX_KEEP_LOADED_HOURS}h ({MAX_KEEP_LOADED_SECONDS} seconds); -1 keeps models for as '
                          f'long as Ollama runs', reason=REASON_KEEP_LOADED_TOO_LONG)
    except ValueError:
        raise ConfigError(f'Backend {name}: {KEY_KEEP_LOADED} must be {KEEP_LOADED_FORMS}', reason=REASON_KEEP_LOADED)


def keep_loaded_seconds(value: object) -> int | float:
    """The seconds Ollama reads a keep_alive value as; negative means until it stops. ValueError if
    it would not take the value: a number of seconds, or a time in Go's syntax (see DURATION_UNITS);
    KeepLoadedTooLong if it is longer than Ollama can count. A number is returned as it is, a whole
    one as the int it is."""
    if isinstance(value, bool):
        raise ValueError('true and false are not a time')
    if isinstance(value, (int, float)):
        if math.isnan(value):
            raise ValueError('not a number')
        if abs(value) > MAX_KEEP_LOADED_SECONDS:
            raise KeepLoadedTooLong('longer than Ollama can count')
        return value
    if not isinstance(value, str):
        raise ValueError('neither a number nor a text')
    negative = value.startswith(MINUS)
    unsigned = value.removeprefix(MINUS) if negative else value.removeprefix(PLUS)
    seconds = Decimal(0) if unsigned == ZERO_DURATION else _duration_seconds(unsigned)
    return float(-seconds if negative else seconds)


def _duration_seconds(text: str) -> Decimal:
    """The seconds of a time in Go's syntax without its sign, exactly. Read in one pass, part after
    part, and no further than the limit: ValueError if it is no such time, KeepLoadedTooLong beyond it."""
    if not text:
        raise ValueError('not a time')
    seconds = Decimal(0)
    position = 0
    while position < len(text):
        part = DURATION_PART.match(text, position)
        if part is None:
            raise ValueError('not a time')
        number, unit = part.groups()
        seconds += Decimal(number) * DURATION_UNITS[unit]
        if seconds > MAX_KEEP_LOADED_SECONDS:
            raise KeepLoadedTooLong('longer than Ollama can count')
        position = part.end()
    return seconds


def check_backend_name(name: str) -> None:
    if BACKEND_NAME_SEPARATOR in name or not name.isprintable():
        raise ConfigError(f'Backend name {name!r} must not contain "{BACKEND_NAME_SEPARATOR}"',
                          reason=REASON_BACKEND_NAME_CHARACTERS)
    if len(name) > BACKEND_NAME_MAX_LENGTH:
        # the gateway refuses every models report naming such a backend
        raise ConfigError(f'Backend name {name!r} must not be longer than {BACKEND_NAME_MAX_LENGTH} characters',
                          reason=REASON_BACKEND_NAME_TOO_LONG)


def normalized_backend_url(name: str, base_url: str) -> str:
    """``base_url`` of the backend called ``name`` as the client uses it, or a ConfigError saying what is wrong."""
    base_url = base_url.rstrip('/')
    parts = _url_parts(base_url, f'Backend {name}: base_url')
    if parts is None or parts.scheme not in BACKEND_SCHEMES or not parts.hostname:
        raise ConfigError(f'Backend {name}: base_url must be an http(s) URL', reason=REASON_BACKEND_URL_MALFORMED)
    if parts.username or parts.password:
        raise ConfigError(f'Backend {name}: put credentials into api_key, not into base_url',
                          reason=REASON_BACKEND_URL_CREDENTIALS)
    return base_url


def check_gateway_url(url: str, allow_insecure: bool) -> None:
    parts = _url_parts(url, 'gateway.url')
    if parts is None or not parts.hostname:
        raise ConfigError('gateway.url must be a WebSocket URL, e.g. wss://toptic.example/ws/socketllama/',
                          reason=REASON_GATEWAY_URL_MALFORMED)
    if parts.scheme == SECURE_GATEWAY_SCHEME:
        return
    if parts.scheme == INSECURE_GATEWAY_SCHEME and allow_insecure:
        return
    raise ConfigError('gateway.url must use wss:// (set gateway.allow_insecure = true only for local testing)',
                      reason=REASON_GATEWAY_URL_SCHEME)


def _url_parts(url: str, where: str) -> SplitResult | None:
    """The parts of ``url``; None if it cannot be split into any.

    A ConfigError naming ``where`` the URL stands if its port is no port number or its brackets
    do not hold an IPv6 address: such a URL looks right at its start, so that is what is said.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        if any(bracket in url for bracket in IPV6_BRACKETS):
            raise ConfigError(f'{where}: the brackets must come in a pair and hold an IPv6 address',
                              reason=REASON_ADDRESS_BRACKETS)
        return None
    try:
        parts.port  # only checked when asked for
    except ValueError:
        raise ConfigError(f'{where}: the port must be a whole number from {MIN_PORT} to {MAX_PORT}',
                          reason=REASON_ADDRESS_PORT)
    return parts


def _table(data: dict, key: str) -> dict:
    table = data.get(key)
    if not isinstance(table, dict):
        raise ConfigError(f'[{key}] section is required')
    return table


def _required_str(table: dict, key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'{where}: {key} is required')
    return value.strip()


def _secret(table: dict, key: str, where: str, environ, inline_secrets: list, required: bool = True) -> str:
    """Value of ``key``, or of the environment variable named by ``<key>_env``."""
    env_name = table.get(f'{key}_env')
    if env_name:
        value = environ.get(env_name, '')
        if not value and required:
            raise ConfigError(f'{where}: environment variable {env_name} is not set')
        return value
    value = table.get(key, '')
    if not isinstance(value, str):
        raise ConfigError(f'{where}: {key} must be a string')
    if not value and required:
        raise ConfigError(f'{where}: {key} (or {key}_env) is required')
    if value:
        inline_secrets.append(f'{where}: {key}')
    return value


def _optional_positive_int(table: dict, key: str, where: str):
    value = table.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f'{where}: {key} must be a positive whole number')
    return value


def _whole_number(table: dict, key: str, default: int, minimum: int, where: str) -> int:
    """An integer of at least ``minimum``; 0.5 or 2.0 are refused rather than silently rounded."""
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f'{where}.{key} must be a whole number of at least {minimum}, not {value!r}')
    return value


def _positive(table: dict, key: str, default):
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f'{key} must be a positive number')
    return value
