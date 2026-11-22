"""Model discovery on the configured backends.

Every backend is OpenAI-compatible and its ``base_url`` is the API root that serves
``/chat/completions`` (Ollama: http://localhost:11434/v1, LiteLLM: http://localhost:4000/v1).
Discovery adds what each kind can tell about its models: Ollama the context window,
LiteLLM limits and prices.
"""
import json
import logging
import socket
from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from . import protocol
from .config import KIND_LITELLM, KIND_OLLAMA, Backend
from .text import printable

logger = logging.getLogger('socketllama')

# Name lookup answers that mean the name does not exist (EAI_NODATA: it exists without addresses;
# not every platform defines it)
NAME_NOT_FOUND_ERRORS = frozenset({socket.EAI_NONAME, *([socket.EAI_NODATA] if hasattr(socket, 'EAI_NODATA') else [])})

# Failures before any answer arrived: the connection, or an address the HTTP library does not take
NO_ANSWER_ERRORS = (httpx.TransportError, httpx.InvalidURL)

MODELS_PATH = '/models'
LITELLM_MODEL_INFO_PATH = '/model/info'
OLLAMA_OPENAI_SUFFIX = '/v1'
OLLAMA_SHOW_PATH = '/api/show'
OLLAMA_PROCESS_LIST_PATH = '/api/ps'
OLLAMA_NUM_CTX_PARAMETER = 'num_ctx'
OLLAMA_CONTEXT_LENGTH_SUFFIX = '.context_length'
OLLAMA_CAPABILITY_COMPLETION = 'completion'
OLLAMA_CAPABILITY_THINKING = 'thinking'
# A model reference without a tag means this tag (the tag follows the last ':' of the last path part)
OLLAMA_DEFAULT_TAG = 'latest'
OLLAMA_TAG_SEPARATOR = ':'
OLLAMA_PATH_SEPARATOR = '/'
LITELLM_CHAT_MODE = 'chat'
# Ollama's default num_ctx: the window a model gets when neither the Modelfile, the server
# (OLLAMA_CONTEXT_LENGTH) nor the backend configuration sets one. Deliberately low: overstating
# the window makes toptic send prompts that Ollama silently cuts from the front.
DEFAULT_OLLAMA_CONTEXT_LENGTH = 4096
DISCOVERY_TIMEOUT_SECONDS = 15.0
# A models report stays this far below the gateway's frame limit (protocol.MAX_GATEWAY_INBOUND_FRAME_BYTES)
MODELS_REPORT_HEADROOM_BYTES = 64 * 1024
MODELS_REPORT_MAX_BYTES = protocol.MAX_GATEWAY_INBOUND_FRAME_BYTES - MODELS_REPORT_HEADROOM_BYTES
# json.dumps' default item separator: what removing one of several list items saves besides the item
JSON_ITEM_SEPARATOR = ', '
# Model names a warning lists at most; the rest is counted
DROPPED_NAMES_SHOWN = 10


def describe_error(ex: Exception) -> str:
    """The exception's type and message; several httpx errors carry no message at all.

    The message may quote the backend (a redirect's Location header, for one): it is made printable.
    """
    return printable(f'{type(ex).__name__}: {ex}' if str(ex) else type(ex).__name__)


def auth_headers(backend: Backend) -> dict:
    return {'authorization': f'Bearer {backend.api_key}'} if backend.api_key else {}


class BackendUnavailable(Exception):
    """A backend's models could not be listed; the message says why.

    ``is_down``: nothing serves the backend (it refuses connections, or its host name does not
    exist). ``timed_out``: it did not answer within DISCOVERY_TIMEOUT_SECONDS. ``answered``: it
    gave an answer, one that cannot be used; ``status`` is its HTTP status if that is what was
    wrong with it. None of them: it could not be reached, which says nothing about whether it
    runs (no route, a failed TLS handshake, an address that is none).
    """

    def __init__(self, detail: str, is_down: bool = False, timed_out: bool = False, answered: bool = False,
                 status: int | None = None):
        super().__init__(detail)
        self.is_down = is_down
        self.timed_out = timed_out
        self.answered = answered
        self.status = status


# What discovery logs for a person it also attaches to the log record, under this attribute, as
# a DiscoveryNote: setup reads the notes and says them in its own words instead of showing log lines
NOTE_ATTRIBUTE = 'discovery_note'
# The backend is down / did not answer in time / could not be listed for another reason (problem)
NOTE_DOWN = 'down'
NOTE_TIMED_OUT = 'timed_out'
NOTE_FAILED = 'failed'
# Models left out of a report (names): the backend is beyond the number of backends a report holds,
# the gateway would refuse the names, the backend has more models than a report holds, the report
# would be larger than a frame
NOTE_BACKEND_NOT_REPORTED = 'backend_not_reported'
NOTE_NAMES_REFUSED = 'names_refused'
NOTE_TOO_MANY_MODELS = 'too_many_models'
NOTE_REPORT_TOO_LARGE = 'report_too_large'
# LiteLLM: the key may not read /model/info (status), so limits and prices are unknown; an entry
# of it that cannot be used (names)
NOTE_DETAILS_UNREADABLE = 'details_unreadable'
NOTE_ENTRY_IGNORED = 'entry_ignored'
# Not from discovery: a log line that carried no note, as whoever collects notes may keep it (text)
NOTE_MESSAGE = 'message'


@dataclass(frozen=True)
class DiscoveryNote:
    kind: str
    backend: str
    names: tuple = ()
    problem: BackendUnavailable | None = None
    status: int | None = None
    text: str = ''


def _noted(kind: str, backend_name: str, **details) -> dict:
    """The ``extra`` of a log call that carries a DiscoveryNote."""
    return {NOTE_ATTRIBUTE: DiscoveryNote(kind=kind, backend=backend_name, **details)}


async def list_backend_models(backend: Backend, http: httpx.AsyncClient) -> list:
    """Models of one backend as report entries, those its configuration shares. Raises BackendUnavailable."""
    try:
        if backend.kind == KIND_LITELLM:
            models = await _discover_litellm(backend, http)
        elif backend.kind == KIND_OLLAMA:
            models = await _discover_ollama(backend, http)
        else:
            models = await _list_models(backend, http)
    except httpx.TimeoutException as ex:
        raise BackendUnavailable(f'no answer within {DISCOVERY_TIMEOUT_SECONDS:g} s', timed_out=True) from ex
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as ex:
        raise BackendUnavailable(describe_error(ex), is_down=_shows_backend_down(ex),
                                 answered=not isinstance(ex, NO_ANSWER_ERRORS),
                                 status=ex.response.status_code if isinstance(ex, httpx.HTTPStatusError) else None
                                 ) from ex
    if backend.context_length and backend.kind != KIND_OLLAMA:  # Ollama applies it itself, capped
        for model in models:
            model['context_length'] = backend.context_length
    return [model for model in models if backend.allows_model(model['name'])]


async def discover_backend(backend: Backend, http: httpx.AsyncClient) -> list | None:
    """Models of one backend as report entries; None when they could not be checked.

    A backend known to be down (it refuses connections, or its host name does not resolve) has
    no models to offer: []. A timeout or an unusable answer says nothing about its models (a
    busy Ollama may answer late): None, reported as unchecked.
    """
    try:
        return await list_backend_models(backend, http)
    except BackendUnavailable as ex:
        if ex.timed_out:
            logger.warning('Backend %s: model discovery timed out after %g s', backend.name, DISCOVERY_TIMEOUT_SECONDS,
                           extra=_noted(NOTE_TIMED_OUT, backend.name, problem=ex))
            return None
        if ex.is_down:
            logger.warning('Backend %s is down, no models to report: %s', backend.name, ex,
                           extra=_noted(NOTE_DOWN, backend.name, problem=ex))
            return []
        logger.warning('Backend %s: model discovery failed: %s', backend.name, ex,
                       extra=_noted(NOTE_FAILED, backend.name, problem=ex))
        return None


async def shareable_model_names(backend: Backend, http: httpx.AsyncClient) -> list:
    """Names of the models ``backend`` could share, as a models report would carry them (the same
    discovery, trimmed to what the gateway accepts). Raises BackendUnavailable."""
    models = await list_backend_models(backend, http)
    report = fit_report_to_gateway([{'name': backend.name, 'kind': backend.kind, 'models': models}])
    return [model['name'] for backend_report in report for model in backend_report['models']]


async def build_models_report(backends: dict, http: httpx.AsyncClient,
                              config_fields: Mapping[tuple[str, str], Mapping] | None = None) -> list:
    """The models report of ``backends``. ``config_fields``: {(backend, model): {field: value}} of what
    config.toml holds for a model, as far as the gateway takes it (its price, capability models.prices, and its
    description, models.descriptions; see config_settings.ConfigSettings.report_fields). Added before the
    report is fitted to the gateway's limits, so that it stays within them with those fields."""
    report = []
    for backend in backends.values():
        models = await discover_backend(backend, http)
        # Each refresh: the owner is shown the report as sent (events.models_reported) when it changed
        if models is None:
            logger.debug('Backend %s (%s): could not be checked; its models stay as they are', backend.name,
                         backend.kind)
            report.append({'name': backend.name, 'kind': backend.kind, protocol.BACKEND_CHECKED_FIELD: False,
                           'models': []})
            continue
        logger.debug('Backend %s (%s): %d model(s)', backend.name, backend.kind, len(models))
        for model in models if config_fields else ():
            model.update(config_fields.get((backend.name, model['name']), {}))
        report.append({'name': backend.name, 'kind': backend.kind, 'models': models})
    return fit_report_to_gateway(report)


def _shows_backend_down(ex: BaseException) -> bool:
    """Whether a failure proves that nothing serves the backend: the connection was refused, or the
    host name does not exist. The operating system's error is in the cause chain of httpx's; for a
    host with several addresses it is a group of one error per address, and all must say so.
    A temporary name lookup failure proves nothing."""
    seen = set()
    while ex is not None and id(ex) not in seen:
        seen.add(id(ex))
        if isinstance(ex, BaseExceptionGroup):
            return all(_shows_backend_down(member) for member in ex.exceptions)
        if isinstance(ex, ConnectionRefusedError):
            return True
        if isinstance(ex, socket.gaierror):
            return ex.errno in NAME_NOT_FOUND_ERRORS
        ex = ex.__cause__ or ex.__context__
    return False


def without_unchecked_marker(report: list) -> list:
    """The report for a gateway without the models.unchecked capability: an unchecked backend is
    just a backend without models."""
    return [{key: value for key, value in backend.items() if key != protocol.BACKEND_CHECKED_FIELD}
            for backend in report]


def fit_report_to_gateway(report: list) -> list:
    """The report trimmed to what the gateway accepts; every model left out is named in a warning.

    In this order: backends beyond protocol.MAX_BACKENDS_PER_REPORT, model names the gateway
    would refuse (too long, not printable), models beyond protocol.MAX_MODELS_PER_BACKEND, and
    finally models from the end of the report until the frame is at most MODELS_REPORT_MAX_BYTES.
    """
    for backend in report[protocol.MAX_BACKENDS_PER_REPORT:]:
        _warn_dropped(backend, [model['name'] for model in backend['models']], NOTE_BACKEND_NOT_REPORTED,
                      f'only the first {protocol.MAX_BACKENDS_PER_REPORT} backends can be reported')
    fitted = []
    for backend in report[:protocol.MAX_BACKENDS_PER_REPORT]:
        named = [model for model in backend['models'] if _is_reportable_name(model['name'])]
        _warn_dropped(backend, [model['name'] for model in backend['models'] if not _is_reportable_name(model['name'])],
                      NOTE_NAMES_REFUSED,
                      f'names must be printable and at most {protocol.MODEL_NAME_MAX_LENGTH} characters long')
        _warn_dropped(backend, [model['name'] for model in named[protocol.MAX_MODELS_PER_BACKEND:]],
                      NOTE_TOO_MANY_MODELS,
                      f'at most {protocol.MAX_MODELS_PER_BACKEND} models per backend can be reported')
        fitted.append({**backend, 'models': named[:protocol.MAX_MODELS_PER_BACKEND]})
    _fit_frame_size(fitted)
    return fitted


def models_frame_size(report: list) -> int:
    return len(json.dumps({'type': protocol.FRAME_MODELS, 'backends': report}))


def _fit_frame_size(report: list):
    """Drop models from the end of the report until its frame fits into MODELS_REPORT_MAX_BYTES."""
    size = models_frame_size(report)
    for backend in reversed(report):
        dropped = []
        while backend['models'] and size > MODELS_REPORT_MAX_BYTES:
            model = backend['models'].pop()
            size -= len(json.dumps(model)) + (len(JSON_ITEM_SEPARATOR) if backend['models'] else 0)
            dropped.append(model['name'])
        _warn_dropped(backend, dropped[::-1], NOTE_REPORT_TOO_LARGE,
                      f'the report has to stay within the gateway\'s size limit of '
                      f'{protocol.MAX_GATEWAY_INBOUND_FRAME_BYTES} bytes')


def _is_reportable_name(name) -> bool:
    return isinstance(name, str) and 0 < len(name) <= protocol.MODEL_NAME_MAX_LENGTH and name.isprintable()


def _warn_dropped(backend: dict, names: list, kind: str, reason: str):
    if not names:
        return
    shown = ', '.join(repr(name) for name in names[:DROPPED_NAMES_SHOWN])
    more = f' and {len(names) - DROPPED_NAMES_SHOWN} more' if len(names) > DROPPED_NAMES_SHOWN else ''
    logger.warning('Backend %s: %d model(s) not reported, %s: %s%s', backend['name'], len(names), reason, shown, more,
                   extra=_noted(kind, backend['name'], names=tuple(names)))


async def _get_json(backend: Backend, http: httpx.AsyncClient, url: str, **kwargs) -> dict:
    response = await http.request(kwargs.pop('method', 'GET'), url, headers=auth_headers(backend),
                                  timeout=DISCOVERY_TIMEOUT_SECONDS, **kwargs)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f'{url} did not return a JSON object')
    return data


def _listed(data: dict, key: str, url: str) -> list:
    """The list an answer holds under ``key``; none at all is an empty one, anything else is no answer to use."""
    listed = data.get(key)
    if listed is None:
        return []
    if not isinstance(listed, list):
        raise ValueError(f'"{key}" of {url} is not a list')
    return listed


async def _list_models(backend: Backend, http: httpx.AsyncClient) -> list:
    data = await _get_json(backend, http, backend.base_url + MODELS_PATH)
    return [{'name': entry['id']} for entry in _listed(data, 'data', backend.base_url + MODELS_PATH)
            if isinstance(entry, dict) and isinstance(entry.get('id'), str)]


async def _discover_litellm(backend: Backend, http: httpx.AsyncClient) -> list:
    """LiteLLM proxy: /model/info carries limits and per-token costs; keys without access fall back to /models."""
    try:
        data = await _get_json(backend, http, backend.base_url + LITELLM_MODEL_INFO_PATH)
    except httpx.HTTPStatusError as ex:
        logger.info('Backend %s: %s not readable (%s), using %s', backend.name, LITELLM_MODEL_INFO_PATH,
                    ex.response.status_code, MODELS_PATH,
                    extra=_noted(NOTE_DETAILS_UNREADABLE, backend.name, status=ex.response.status_code))
        return await _list_models(backend, http)
    models = []
    for entry in _listed(data, 'data', backend.base_url + LITELLM_MODEL_INFO_PATH):
        if not isinstance(entry, dict) or not isinstance(entry.get('model_name'), str):
            continue
        info = entry.get('model_info') or {}
        if not isinstance(info, dict):
            logger.warning('Backend %s: ignoring %s, its model_info is not an object', backend.name,
                           printable(entry['model_name']),
                           extra=_noted(NOTE_ENTRY_IGNORED, backend.name, names=(entry['model_name'],)))
            continue
        if info.get('mode') not in (None, LITELLM_CHAT_MODE):
            continue  # embeddings, image generation, ...
        models.append({
            'name': entry['model_name'],
            'context_length': info.get('max_input_tokens'),
            'max_output_tokens': info.get('max_output_tokens'),
            'input_cost_per_token': info.get('input_cost_per_token'),
            'output_cost_per_token': info.get('output_cost_per_token'),
            'supports_reasoning': bool(info.get('supports_reasoning')),
        })
    return models


async def _discover_ollama(backend: Backend, http: httpx.AsyncClient) -> list:
    """Ollama: /v1/models lists everything pulled; /api/show tells chat models from embeddings.

    The context window reported is the one requests actually get (see _ollama_context_length).
    """
    root = ollama_root(backend)
    loaded = await ollama_loaded_models(backend, http) or {}
    models = []
    for model in await _list_models(backend, http):
        if not backend.allows_model(model['name']):
            continue  # not shared: do not even ask for its details
        try:
            details = await _get_json(backend, http, root + OLLAMA_SHOW_PATH, method='POST',
                                      json={'model': model['name']})
        except (httpx.HTTPError, ValueError) as ex:
            logger.debug('Backend %s: no details for %s: %s', backend.name, printable(model['name']),
                         describe_error(ex))
            models.append(model)
            continue
        capabilities = details.get('capabilities')
        if isinstance(capabilities, list) and OLLAMA_CAPABILITY_COMPLETION not in capabilities:
            continue  # embedding-only model
        model['context_length'] = _ollama_context_length(
            backend, details, _loaded_context_length(loaded.get(ollama_model_reference(model['name']))))
        model['supports_reasoning'] = isinstance(capabilities, list) and OLLAMA_CAPABILITY_THINKING in capabilities
        models.append(model)
    return models


def ollama_model_reference(name: str) -> str:
    """``name`` as Ollama resolves it: a model named without a tag is its ``latest`` tag.

    Names compare equal only in this form ("llama3.1" and "llama3.1:latest" are one model). The
    tag follows a ':' in the last path part only: "host:5000/org/model" has none.
    """
    last_part = name.rsplit(OLLAMA_PATH_SEPARATOR, 1)[-1]
    return name if OLLAMA_TAG_SEPARATOR in last_part else f'{name}{OLLAMA_TAG_SEPARATOR}{OLLAMA_DEFAULT_TAG}'


async def ollama_loaded_models(backend: Backend, http: httpx.AsyncClient) -> dict | None:
    """{model reference: /api/ps entry} of the models Ollama holds in memory right now; None if unknown.

    A model missing from it is not in memory. One listed is not necessarily ready: /api/ps has
    no loading state, and lists a model from the moment Ollama starts to load it.
    """
    try:
        url = ollama_root(backend) + OLLAMA_PROCESS_LIST_PATH
        loaded = _listed(await _get_json(backend, http, url), 'models', url)
    except (httpx.HTTPError, ValueError) as ex:
        logger.debug('Backend %s: no list of loaded models: %s', backend.name, describe_error(ex))
        return None
    return {ollama_model_reference(entry['name']): entry for entry in loaded
            if isinstance(entry, dict) and isinstance(entry.get('name'), str)}


def ollama_root(backend: Backend) -> str:
    return backend.base_url.removesuffix(OLLAMA_OPENAI_SUFFIX)


def _loaded_context_length(entry):
    """The context window of a loaded model's /api/ps entry (an undocumented field), if it has one."""
    if entry is None or not _is_positive_int(entry.get('context_length')):
        return None
    return entry['context_length']


def _ollama_context_length(backend: Backend, details: dict, loaded_context_length) -> int:
    """The window requests through the OpenAI API get, which cannot set num_ctx themselves.

    First match wins: the backend's configured context_length, the window of the loaded model
    (/api/ps), num_ctx of the Modelfile, Ollama's default. Never more than the architecture
    supports (model_info.<arch>.context_length).
    """
    window = backend.context_length or loaded_context_length or _modelfile_num_ctx(details.get('parameters')) \
        or DEFAULT_OLLAMA_CONTEXT_LENGTH
    architecture_maximum = _architecture_context_length(details.get('model_info'))
    return min(window, architecture_maximum) if architecture_maximum else window


def _modelfile_num_ctx(parameters):
    """num_ctx from /api/show's parameters text ("name value" per line), if the Modelfile sets it."""
    if not isinstance(parameters, str):
        return None
    for line in parameters.splitlines():
        name, _, value = line.strip().partition(' ')
        if name == OLLAMA_NUM_CTX_PARAMETER and value.strip().isdigit() and int(value.strip()) > 0:
            return int(value.strip())
    return None


def _architecture_context_length(model_info):
    if not isinstance(model_info, dict):
        return None
    for key, value in model_info.items():
        if key.endswith(OLLAMA_CONTEXT_LENGTH_SUFFIX) and _is_positive_int(value):
            return value
    return None


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
