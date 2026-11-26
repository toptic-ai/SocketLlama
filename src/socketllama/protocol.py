"""Wire protocol of the SocketLlama tunnel, client side.

Mirror of toptic.ai's gateway (ai_helper/service/socketllama_protocol.py in the toptic.ai
repository); both sides refuse a peer with another PROTOCOL_VERSION. All frames are JSON text,
HTTP bodies travel base64-encoded.

Additions that need no new version are capabilities: the gateway lists the ones it supports in
auth_ok ("capabilities"), and this client uses one only when it is listed. A gateway that lists
none gets none of them.

    response.progress   Before response.start of a request, response.progress frames tell
                        what the machine did so far, as one of PROGRESS_STAGES, each stage
                        once; the gateway keeps showing the last one until the answer starts.
    models.unchecked    A backend whose discovery timed out or got no usable answer is
                        reported with "checked": false and no models; the gateway keeps its
                        available models for a grace period instead of deactivating them. A
                        backend known to be down (connection refused, host name not found) is
                        reported without models. Without the capability the marker is left
                        out.
    models.memory       A models.memory frame tells which reported models Ollama holds in
                        memory: {"backends": [{"backend": <name>, "models": [<name>, ...]}]},
                        names as the models report has them. A full snapshot of every Ollama
                        backend, not a change; one with nothing loaded has an empty list, one
                        whose /api/ps could not be read (or that the last models report did not
                        check) is left out, since its state is unknown. Sent after the first
                        models report of a connection, then whenever it changes, at most one
                        per MODELS_MEMORY_MIN_INTERVAL_SECONDS (see models_memory).
    models.prices       Each model of a models report carries "config_price": {"input": <USD>,
                        "output": <USD>} per 1M tokens, the price config.toml holds for it (left
                        out for a model without one). The gateway answers with models.settings
                        frames: {"models": [{"backend": <name>, "model": <name>, "price":
                        {"input": <USD>, "output": <USD>}}, ...]}, the owner's price in force of
                        each model listed; the client writes it into config.toml (see
                        config_settings). Not before the gateway acknowledged a models report of
                        this run, which told it what config.toml holds: a frame that comes
                        earlier is ignored, and the answer to that report brings the prices
                        again. No answer frame: the next report is the answer. Without the
                        capability the report carries no prices and the frames are ignored.
    models.descriptions Each model of a models report carries "config_description": "<text>", the
                        description config.toml holds for it (no such key for a model without
                        one): one line of at most MAX_MODEL_DESCRIPTION_CHARS characters. An entry
                        of a models.settings frame (see models.prices) may carry "description":
                        "<text>", the description in force, besides or instead of "price"; the
                        client writes it into config.toml the same way and on the same condition
                        (see config_settings). Each half of an entry is taken or refused for
                        itself. Without the capability the report carries no descriptions, and a
                        description in a frame is ignored.

Besides capabilities, auth_ok may carry "model_loading_timeout_seconds": a positive whole number, how
long the gateway waits for the start of an answer once it was told that the model is not in memory
(the progress stage model_not_in_memory; longer than it waits otherwise). On the same condition the
client then waits at least that long for the backend to begin its answer, also where the backend's
read_timeout is shorter, and so do the loads it asks Ollama for itself (see relay and
ollama_memory). Every other wait is the backend's read_timeout: for a model that was in memory,
for another kind of backend, and between the parts of an answer that has begun. No capability: a
gateway may leave the field out, and a client that does not know it ignores it. A value that is no
positive whole number is ignored; one above MAX_MODEL_LOADING_TIMEOUT_SECONDS counts as that.

Without a capability as well, each model of an Ollama backend in a models report that has a
"context_length" carries "context_length_source", where that window comes from (other kinds of
backends send none; CONTEXT_LENGTH_SOURCES):

    configured          The backend's context_length in config.toml.
    loaded              The window of the model in memory, as Ollama's /api/ps lists it: what
                        the server gives it, also where OLLAMA_CONTEXT_LENGTH set it.
    modelfile           num_ctx of the model's Modelfile (/api/show).
    default             None of these named one: Ollama's default num_ctx
                        (backends.DEFAULT_OLLAMA_CONTEXT_LENGTH), a guess. The server's
                        OLLAMA_CONTEXT_LENGTH may give requests a larger one, which Ollama's API
                        shows only once the model is loaded. Also when /api/show could not be read
                        and nothing else named one.

The window is never more than the model's architecture supports; the source stays the one that
named it (a capped default is still "default"). What the gateway makes of it: while a model is not
in memory and reported with "default", it keeps the window last reported for it as "loaded", the
server's own setting; a report with any other source replaces that window (a configured or
Modelfile window that is gone was removed by the owner, so the default counts again then). So
that it learns a loaded window soon, the client sends a fresh models report when /api/ps lists a
window for a model of an Ollama backend without a configured context_length that the last report
did not carry for it as loaded: the model came into memory, or was loaded again with another
window (seen by the models.memory checks, so only with that capability; see models_memory),
within the gateway's limit of one report per MODELS_REPORT_RESEND_SECONDS (client). A gateway
that does not know the field ignores it.

The other way round, this client lists what it can take in auth ("capabilities"), and the gateway
uses one only when it is listed. A gateway that does not know the list ignores it.

    request.user        A request frame may carry "user": the name toptic.ai shows of the
                        person who sent the prompt (their display name, never an email
                        address). Only shown to this machine's owner, in the terminal; not
                        part of the signature, so nothing the client decides depends on it.
                        Without it the prompt is from an unknown user.
    models.prices       The gateway may send models.settings frames (see models.prices above).
    models.descriptions The entries of those frames may carry "description" (see above); the
                        gateway sends one only to a client that listed this.
"""
import base64
import hashlib
import hmac
import re
import unicodedata

from .units import SECONDS_PER_HOUR

PROTOCOL_VERSION = 2

# client -> gateway
FRAME_AUTH = 'auth'
FRAME_MODELS = 'models'
FRAME_RESPONSE_START = 'response.start'
FRAME_RESPONSE_CHUNK = 'response.chunk'
FRAME_RESPONSE_END = 'response.end'
FRAME_RESPONSE_ERROR = 'response.error'
FRAME_PING = 'ping'
FRAME_RESPONSE_PROGRESS = 'response.progress'
FRAME_MODELS_MEMORY = 'models.memory'

# Stages of a response.progress frame
# The request arrived on this machine and goes to its backend
PROGRESS_STAGE_RECEIVED = 'received'
# Ollama did not list the model among those in memory (/api/ps): it is loaded first
PROGRESS_STAGE_MODEL_NOT_IN_MEMORY = 'model_not_in_memory'

# Field of a backend in a models report; only ever sent as false (see models.unchecked)
BACKEND_CHECKED_FIELD = 'checked'
# Field of a model of an Ollama backend in a models report: where its context_length comes from, one of these
# (see the module docstring)
CONTEXT_LENGTH_SOURCE_FIELD = 'context_length_source'
CONTEXT_LENGTH_SOURCE_CONFIGURED = 'configured'
CONTEXT_LENGTH_SOURCE_LOADED = 'loaded'
CONTEXT_LENGTH_SOURCE_MODELFILE = 'modelfile'
CONTEXT_LENGTH_SOURCE_DEFAULT = 'default'
CONTEXT_LENGTH_SOURCES = (CONTEXT_LENGTH_SOURCE_CONFIGURED, CONTEXT_LENGTH_SOURCE_LOADED,
                          CONTEXT_LENGTH_SOURCE_MODELFILE, CONTEXT_LENGTH_SOURCE_DEFAULT)

# gateway -> client
FRAME_AUTH_REQUIRED = 'auth_required'
FRAME_AUTH_OK = 'auth_ok'
FRAME_AUTH_FAILED = 'auth_failed'
FRAME_REQUEST = 'request'
FRAME_CANCEL = 'cancel'
FRAME_PAUSE = 'pause'
FRAME_RESUME = 'resume'
FRAME_MODELS_ACK = 'models_ack'
FRAME_PONG = 'pong'
FRAME_ERROR = 'error'
FRAME_MODELS_SETTINGS = 'models.settings'

# Capabilities a gateway may announce in auth_ok (see the module docstring)
CAPABILITY_RESPONSE_PROGRESS = 'response.progress'
CAPABILITY_MODELS_UNCHECKED = 'models.unchecked'
CAPABILITY_MODELS_MEMORY = 'models.memory'
# Announced by both sides: the gateway in auth_ok, this client in auth
CAPABILITY_MODELS_PRICES = 'models.prices'
CAPABILITY_MODELS_DESCRIPTIONS = 'models.descriptions'

# Field of auth_ok: for how many seconds the gateway waits for the start of an answer while a model loads
# (see the module docstring)
MODEL_LOADING_TIMEOUT_FIELD = 'model_loading_timeout_seconds'
# The longest such wait the client takes from a gateway: an hour, and a longer one counts as an hour. The
# gateway is trusted to route requests, not to keep this machine's request slots and its backend waiting
# without end; and an hour is several times what the largest models take to load from a slow disk (17 GB
# took under seven minutes). The gateway checks its own setting against the same value (its mirror of this
# constant): change both together.
MAX_MODEL_LOADING_TIMEOUT_HOURS = 1
MAX_MODEL_LOADING_TIMEOUT_SECONDS = MAX_MODEL_LOADING_TIMEOUT_HOURS * SECONDS_PER_HOUR

# Capabilities this client announces in auth (see the module docstring)
CAPABILITY_REQUEST_USER = 'request.user'
CLIENT_CAPABILITIES = (CAPABILITY_REQUEST_USER, CAPABILITY_MODELS_PRICES, CAPABILITY_MODELS_DESCRIPTIONS)

# Field of a model in a models report: its price in config.toml (capability models.prices)
CONFIG_PRICE_FIELD = 'config_price'
# Fields of a price, in a report and in a models.settings frame: USD per 1M tokens of the prompt and of the answer
PRICE_INPUT_FIELD = 'input'
PRICE_OUTPUT_FIELD = 'output'
# Fields of a models.settings frame and of each of its entries
SETTINGS_MODELS_FIELD = 'models'
SETTINGS_BACKEND_FIELD = 'backend'
SETTINGS_MODEL_FIELD = 'model'
SETTINGS_PRICE_FIELD = 'price'
SETTINGS_DESCRIPTION_FIELD = 'description'
# Field of a model in a models report: its description in config.toml (capability models.descriptions)
CONFIG_DESCRIPTION_FIELD = 'config_description'
# The rule for the description of a model, the gateway's rule as well (its constants of the same names: change
# both together). config.toml is held to it, and so is a description the gateway sends. After the blanks around
# it are taken off (str.strip), a description has 1 to MAX_MODEL_DESCRIPTION_CHARS code points, none of them one
# of these:
MAX_MODEL_DESCRIPTION_CHARS = 500
# By Unicode category: control characters (a tab, a line feed), the separators of lines and of paragraphs, and
# halves of UTF-16 pairs (no characters at all)
DESCRIPTION_REFUSED_CATEGORIES = frozenset({'Cc', 'Zl', 'Zp', 'Cs'})
# And the bidirectional controls, which would let a description reorder what toptic.ai shows next to it (the
# name of a model): the marks, then the embeddings and overrides, then the isolates, each range with both ends
LEFT_TO_RIGHT_MARK = 0x200E
RIGHT_TO_LEFT_MARK = 0x200F
ARABIC_LETTER_MARK = 0x061C
FIRST_BIDI_EMBEDDING_CONTROL = 0x202A   # LEFT-TO-RIGHT EMBEDDING
LAST_BIDI_EMBEDDING_CONTROL = 0x202E    # RIGHT-TO-LEFT OVERRIDE
FIRST_BIDI_ISOLATE_CONTROL = 0x2066     # LEFT-TO-RIGHT ISOLATE
LAST_BIDI_ISOLATE_CONTROL = 0x2069      # POP DIRECTIONAL ISOLATE
DESCRIPTION_REFUSED_CODE_POINTS = frozenset({
    LEFT_TO_RIGHT_MARK, RIGHT_TO_LEFT_MARK, ARABIC_LETTER_MARK,
    *range(FIRST_BIDI_EMBEDDING_CONTROL, LAST_BIDI_EMBEDDING_CONTROL + 1),
    *range(FIRST_BIDI_ISOLATE_CONTROL, LAST_BIDI_ISOLATE_CONTROL + 1)})
# Everything else is allowed: every space (a no-break space too), joiners, a soft hyphen, private use

# Field of a request frame with the name of the person asking (capability request.user)
REQUEST_USER_FIELD = 'user'
# Longest such name (the gateway's Account.display_name holds at most this many characters)
USER_NAME_MAX_LENGTH = 128

# Reasons of an auth_failed frame: AUTH_FAILED_* of toptic.ai's gateway (mq_system/socketllama_consumer.py in the
# toptic.ai repository)
AUTH_FAILED_TOKEN = 'invalid_token'
AUTH_FAILED_SCOPE = 'missing_scope'
AUTH_FAILED_INSTANCE = 'unknown_instance'
AUTH_FAILED_PROTOCOL = 'unsupported_protocol'
AUTH_FAILED_SECRET_UNREADABLE = 'signing_secret_unreadable'

# WebSocket close codes the gateway uses
CLOSE_UNAUTHENTICATED = 4001
CLOSE_FORBIDDEN = 4003
CLOSE_REPLACED = 4009
# The token, the user, the instance or its signing secret was revoked
CLOSE_REVOKED = 4010

# Error codes of response.error frames produced by this client
ERROR_BAD_SIGNATURE = 'bad_signature'
ERROR_REPLAYED_REQUEST = 'replayed_request'
ERROR_UNKNOWN_BACKEND = 'unknown_backend'
ERROR_PATH_NOT_ALLOWED = 'path_not_allowed'
ERROR_MODEL_NOT_ALLOWED = 'model_not_allowed'
ERROR_INVALID_REQUEST = 'invalid_request'
ERROR_BACKEND_UNREACHABLE = 'backend_unreachable'
ERROR_BACKEND_TIMEOUT = 'backend_timeout'
ERROR_CLIENT_BUSY = 'client_busy'
ERROR_CLIENT_INTERNAL = 'client_error'
ERROR_REQUEST_TOO_LARGE = 'request_too_large'
# An Ollama backend refused the request because it is still loading a model (see ollama_memory): the client
# sends this instead of passing the backend's error answer on, so no response.start comes before it. No
# capability: a gateway that does not know the code shows it like any unknown code.
ERROR_BACKEND_MODEL_LOADING = 'backend_model_loading'

# Largest request frame the gateway sends: toptic.ai's gateway (ai_helper/service/socketllama_protocol.py in the
# toptic.ai repository) holds the same value as CLIENT_MAX_REQUEST_FRAME_BYTES and refuses a setting above it
MAX_REQUEST_FRAME_BYTES = 64 * 1024 * 1024

# What the gateway accepts in one models report: mirrors of MAX_BACKENDS_PER_REPORT and MAX_MODELS_PER_BACKEND of
# toptic.ai's gateway (ai_helper/service/socketllama_discovery.py in the toptic.ai repository), and of its
# BACKEND_NAME_MAX_LENGTH and UPSTREAM_MODEL_NAME_MAX_LENGTH (ai_helper/store/socketllama_models.py in the toptic.ai
# repository). It skips backends and models beyond the counts and refuses a report with a longer name; the client
# trims its report to all of them.
MAX_BACKENDS_PER_REPORT = 8
MAX_MODELS_PER_BACKEND = 200
BACKEND_NAME_MAX_LENGTH = 64
MODEL_NAME_MAX_LENGTH = 255
# The prices the gateway takes, in USD per 1M tokens: none negative, none above MAX_PRICE_PER_UNIT of toptic.ai's
# gateway (ai_helper/service/machine_prices.py in the toptic.ai repository). config.toml is held to the same
# bounds, and so is a price the gateway sends.
MIN_PRICE_PER_UNIT = 0.0
MAX_PRICE_PER_UNIT = 1000.0
# Most entries of one models.settings frame: SOCKETLLAMA_MAX_MODELS_PER_INSTANCE of toptic.ai's gateway
# (ai_helper/settings.py in the toptic.ai repository), the models one machine may have there; the client takes
# no more of a frame than that
MAX_SETTINGS_PER_FRAME = 100
# Largest frame the gateway accepts from a client (its MAX_FRAME_CHARS, the default
# websocket_max_message_size of Daphne as well); a bigger one is refused or drops the connection.
MAX_GATEWAY_INBOUND_FRAME_BYTES = 1024 * 1024

SIGNATURE_FIELD_SEPARATOR = '\n'
# Signatures are HMAC-SHA256 in lowercase hex
SIGNATURE_PATTERN = re.compile(f'[0-9a-f]{{{hashlib.sha256().digest_size * 2}}}')


def encode_body(body: bytes) -> str:
    return base64.b64encode(body).decode('ascii')


def decode_body(data: str) -> bytes:
    return base64.b64decode(data.encode('ascii'), validate=True)


def refused_in_description(text: str) -> str | None:
    """The first character of ``text`` that a description may not hold (see DESCRIPTION_REFUSED_CATEGORIES and
    DESCRIPTION_REFUSED_CODE_POINTS); None if it holds none."""
    return next((character for character in text
                 if unicodedata.category(character) in DESCRIPTION_REFUSED_CATEGORIES
                 or ord(character) in DESCRIPTION_REFUSED_CODE_POINTS), None)


def model_loading_timeout(auth_ok: dict) -> int | None:
    """What an auth_ok frame says under MODEL_LOADING_TIMEOUT_FIELD, in seconds, as it says it (see
    MAX_MODEL_LOADING_TIMEOUT_SECONDS for what the client makes of a long one); None without the field.
    Raises ValueError, saying why, for anything but a positive whole number."""
    if MODEL_LOADING_TIMEOUT_FIELD not in auth_ok:
        return None
    seconds = auth_ok[MODEL_LOADING_TIMEOUT_FIELD]
    if not isinstance(seconds, int) or isinstance(seconds, bool):
        raise ValueError(f'{seconds!r} is not a whole number of seconds')
    if seconds <= 0:
        raise ValueError(f'{seconds} is not a positive number of seconds')
    return seconds


def sign_request(secret: str, nonce: str, request_id: str, backend: str, method: str, path: str,
                 body: bytes) -> str:
    """HMAC-SHA256 over the connection's nonce and the fields that determine what this client will execute.

    The nonce is the one the gateway sent at the start of this connection, so a frame captured
    on one connection is refused on every later one.
    """
    body_digest = hashlib.sha256(body).hexdigest()
    message = SIGNATURE_FIELD_SEPARATOR.join((nonce, request_id, backend, method, path, body_digest))
    return hmac.new(secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()


def sign_auth(secret: str, nonce: str, instance: str) -> str:
    """Proof that this client knows the instance's signing secret, bound to the gateway's nonce.

    Without it a stolen API token would be enough to pose as the instance's machine.
    """
    message = SIGNATURE_FIELD_SEPARATOR.join((nonce, instance))
    return hmac.new(secret.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()


def signature_is_valid(secret: str, nonce: str, frame: dict, body: bytes) -> bool:
    signature = frame.get('signature')
    # Format first: compare_digest raises TypeError on non-ASCII text
    if not isinstance(signature, str) or not SIGNATURE_PATTERN.fullmatch(signature):
        return False
    expected = sign_request(secret, nonce, str(frame.get('id')), str(frame.get('backend')),
                            str(frame.get('method')), str(frame.get('path')), body)
    return hmac.compare_digest(expected, signature)
