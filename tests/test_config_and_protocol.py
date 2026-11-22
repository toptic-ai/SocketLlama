import tomllib
from pathlib import Path

import pytest

from socketllama import config as config_module
from socketllama import protocol
from socketllama.client import BACKOFF_FACTOR, JITTER_FLOOR, backoff_delay
from socketllama.config import (
    DEFAULT_PING_INTERVAL_SECONDS, KIND_LITELLM, ConfigError, check_backend_name, check_gateway_url, load_config,
    normalized_backend_url, parse_config,
)

GATEWAY = {'url': 'wss://toptic.example/ws/socketllama/', 'instance': 'me-home', 'token': 'tt_live_x',
           'signing_secret': 'secret'}
OLLAMA = {'name': 'ollama', 'kind': 'ollama', 'base_url': 'http://localhost:11434/v1/', 'models': ['llama3.1:8b']}
# Same vector as the tests of toptic.ai's gateway (tests/test_tunnel_registry.py in the toptic.ai repository):
# both ends must agree
VECTOR_SECRET = 'test-secret'
VECTOR_BODY = b'{"model":"llama3.1:8b"}'
VECTOR_NONCE = 'nonce-1'
VECTOR_SIGNATURE = '460cc46c4feda1559c03875bd183e4ad84b5aaebac8ae4681cf82d8e3c646047'
VECTOR_AUTH_PROOF = '2bb9c583dfd0765d8464073e55e786dd339cc75a0dd4963f4d78c680a8ba3b82'
# PROTOCOL_VERSION of toptic.ai's gateway (ai_helper/service/socketllama_protocol.py in the toptic.ai repository)
GATEWAY_PROTOCOL_VERSION = 2
# MAX_BACKENDS_PER_REPORT and MAX_MODELS_PER_BACKEND, BACKEND_NAME_MAX_LENGTH and UPSTREAM_MODEL_NAME_MAX_LENGTH,
# and MAX_FRAME_CHARS of toptic.ai's gateway (ai_helper/service/socketllama_discovery.py,
# ai_helper/store/socketllama_models.py and ai_helper/service/socketllama_limits.py in the toptic.ai repository)
GATEWAY_REPORT_LIMITS = (8, 200)
GATEWAY_NAME_LIMITS = (64, 255)
GATEWAY_MAX_FRAME_CHARS = 1024 * 1024
MIN_DELAY = 1.0
MAX_DELAY = 60.0
FULL_JITTER = 1.0
NO_JITTER = 0.0
LATE_ATTEMPT = 20


def config(**sections):
    return parse_config({'gateway': GATEWAY, 'backends': [OLLAMA], **sections}, environ={})


def test_minimal_config():
    parsed = config()
    assert parsed.instance == 'me-home'
    assert parsed.backends['ollama'].base_url == 'http://localhost:11434/v1'  # trailing slash removed
    assert parsed.ping_interval == DEFAULT_PING_INTERVAL_SECONDS


def test_secrets_from_environment():
    parsed = parse_config(
        {'gateway': {'url': GATEWAY['url'], 'instance': 'me-home', 'token_env': 'SL_TOKEN',
                     'signing_secret_env': 'SL_SECRET'},
         'backends': [{'name': 'proxy', 'kind': KIND_LITELLM, 'base_url': 'https://llm.example/v1',
                       'api_key_env': 'SL_KEY', 'models': []}]},
        environ={'SL_TOKEN': 'tt_live_env', 'SL_SECRET': 'env-secret', 'SL_KEY': 'sk-env'})
    assert (parsed.token, parsed.signing_secret) == ('tt_live_env', 'env-secret')
    assert parsed.backends['proxy'].api_key == 'sk-env'


def test_missing_environment_secret_is_an_error():
    with pytest.raises(ConfigError, match='SL_TOKEN'):
        parse_config({'gateway': {**GATEWAY, 'token_env': 'SL_TOKEN'}, 'backends': [OLLAMA]}, environ={})


@pytest.mark.parametrize('broken, message', [
    ({'gateway': {**GATEWAY, 'url': 'ws://toptic.example/ws/'}}, 'wss://'),
    ({'gateway': {**GATEWAY, 'url': 'https://toptic.example/'}}, 'wss://'),
    ({'gateway': {**GATEWAY, 'instance': ''}}, 'instance'),
    ({'gateway': {**GATEWAY, 'signing_secret': ''}}, 'signing_secret'),
    ({'backends': []}, 'At least one'),
    ({'backends': [OLLAMA, OLLAMA]}, 'used twice'),
    ({'backends': [{**OLLAMA, 'name': 'a/b'}]}, 'must not contain'),
    ({'backends': [{**OLLAMA, 'kind': 'vllm'}]}, 'kind must be'),
    ({'backends': [{**OLLAMA, 'base_url': 'ftp://host/v1'}]}, 'http'),
    ({'backends': [{**OLLAMA, 'base_url': 'http://127.0.0.1:1143x/v1'}]}, 'Backend ollama: base_url: the port'),
    ({'gateway': {**GATEWAY, 'url': 'wss://[::1'}}, 'gateway.url: .*brackets'),
    ({'backends': [{**OLLAMA, 'base_url': 'http://user:pw@host/v1'}]}, 'api_key'),
    ({'backends': [{**OLLAMA, 'models': 'llama'}]}, 'list of model names'),
    ({'backends': [{key: value for key, value in OLLAMA.items() if key != 'models'}]}, 'models'),
    ({'client': {'ping_interval': 0}}, 'positive'),
    ({'backends': [{**OLLAMA, 'context_length': 0}]}, 'context_length'),
    ({'backends': [{**OLLAMA, 'context_length': 4096.5}]}, 'context_length'),
    ({'client': {'reconnect_min': 5, 'reconnect_max': 1}}, 'reconnect_max'),
    ({'client': {'max_concurrent_requests': 0.5}}, 'max_concurrent_requests'),
    ({'client': {'max_concurrent_requests': 2.0}}, 'max_concurrent_requests'),
    ({'client': {'max_concurrent_requests': 0}}, 'max_concurrent_requests'),
    ({'client': {'max_concurrent_requests': True}}, 'max_concurrent_requests'),
    ({'client': {'max_concurrent_requests': '4'}}, 'max_concurrent_requests'),
    ({'backends': [{**OLLAMA, 'name': 'b' * (protocol.BACKEND_NAME_MAX_LENGTH + 1)}]}, 'characters'),
])
def test_invalid_configs_are_rejected(broken, message):
    with pytest.raises(ConfigError, match=message):
        parse_config({'gateway': GATEWAY, 'backends': [OLLAMA], **broken}, environ={})


def test_max_concurrent_requests_is_a_whole_number_of_at_least_one():
    assert config(client={'max_concurrent_requests': 1}).max_concurrent_requests == 1


def test_report_limits_match_the_gateway():
    """Mirrors of the values of toptic.ai's gateway named above GATEWAY_REPORT_LIMITS."""
    assert (protocol.MAX_BACKENDS_PER_REPORT, protocol.MAX_MODELS_PER_BACKEND) == GATEWAY_REPORT_LIMITS
    assert (protocol.BACKEND_NAME_MAX_LENGTH, protocol.MODEL_NAME_MAX_LENGTH) == GATEWAY_NAME_LIMITS
    assert protocol.MAX_GATEWAY_INBOUND_FRAME_BYTES == GATEWAY_MAX_FRAME_CHARS


def test_plain_ws_needs_an_explicit_opt_in():
    parsed = config(gateway={**GATEWAY, 'url': 'ws://127.0.0.1:8888/ws/socketllama/', 'allow_insecure': True})
    assert parsed.gateway_url.startswith('ws://')


def test_model_allow_list():
    restricted = config().backends['ollama']
    assert restricted.allows_model('llama3.1:8b')
    assert not restricted.allows_model('qwen3:14b')
    assert not restricted.shares_all_models


def test_an_empty_model_list_shares_every_model_but_has_to_be_written_out():
    everything = config(backends=[{**OLLAMA, 'models': []}]).backends['ollama']
    assert everything.allows_model('anything')
    assert everything.shares_all_models


def test_load_config_reports_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match='not found'):
        load_config(tmp_path / 'absent.toml')
    broken = tmp_path / 'broken.toml'
    broken.write_text('[gateway\n')
    with pytest.raises(ConfigError, match='not valid TOML'):
        load_config(broken)


INLINE_SECRET_CONFIG = f'''
[gateway]
url = "{GATEWAY['url']}"
instance = "me-home"
token = "tt_live_inline-token"
signing_secret = "inline-signing-secret"

[[backends]]
name = "ollama"
kind = "ollama"
base_url = "http://localhost:11434/v1"
models = []
'''
OWNER_ONLY = 0o600
WORLD_READABLE = 0o644


def test_inline_secrets_need_a_file_only_its_owner_can_read(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(INLINE_SECRET_CONFIG)
    path.chmod(WORLD_READABLE)
    with pytest.raises(ConfigError, match='chmod 600'):
        load_config(path)
    path.chmod(OWNER_ONLY)
    assert load_config(path).token == 'tt_live_inline-token'


def test_secrets_from_the_environment_need_no_special_file_mode(tmp_path, monkeypatch):
    monkeypatch.setenv('SOCKETLLAMA_TOKEN', 'tt_live_example')
    monkeypatch.setenv('SOCKETLLAMA_SIGNING_SECRET', 'example-secret')
    monkeypatch.setenv('LITELLM_API_KEY', 'sk-example')
    path = tmp_path / 'config.toml'
    path.write_text(Path('config.example.toml').read_text())
    path.chmod(WORLD_READABLE)
    assert load_config(path).token == 'tt_live_example'


def test_the_repr_of_a_config_shows_no_secrets():
    parsed = config(backends=[{**OLLAMA, 'api_key': 'sk-backend-key'}])
    shown = repr(parsed)
    for secret in (GATEWAY['token'], GATEWAY['signing_secret'], 'sk-backend-key'):
        assert secret not in shown


def test_example_config_is_valid(monkeypatch):
    monkeypatch.setenv('SOCKETLLAMA_TOKEN', 'tt_live_example')
    monkeypatch.setenv('SOCKETLLAMA_SIGNING_SECRET', 'example-secret')
    monkeypatch.setenv('LITELLM_API_KEY', 'sk-example')
    parsed = load_config('config.example.toml')
    assert set(parsed.backends) == {'ollama', 'litellm'}


def test_signature_matches_the_gateway_implementation():
    signature = protocol.sign_request(VECTOR_SECRET, VECTOR_NONCE, 'req-1', 'ollama', 'POST', '/chat/completions',
                                      VECTOR_BODY)
    assert signature == VECTOR_SIGNATURE


def test_signature_covers_every_field_and_the_connection():
    frame = {'id': 'req-1', 'backend': 'ollama', 'method': 'POST', 'path': '/chat/completions',
             'signature': VECTOR_SIGNATURE}
    assert protocol.signature_is_valid(VECTOR_SECRET, VECTOR_NONCE, frame, VECTOR_BODY)
    assert not protocol.signature_is_valid('other-secret', VECTOR_NONCE, frame, VECTOR_BODY)
    assert not protocol.signature_is_valid(VECTOR_SECRET, 'nonce-of-a-later-connection', frame, VECTOR_BODY)
    assert not protocol.signature_is_valid(VECTOR_SECRET, VECTOR_NONCE, frame, VECTOR_BODY + b' ')
    for field, value in (('id', 'req-2'), ('backend', 'litellm'), ('method', 'GET'), ('path', '/models')):
        assert not protocol.signature_is_valid(VECTOR_SECRET, VECTOR_NONCE, {**frame, field: value}, VECTOR_BODY)
    assert not protocol.signature_is_valid(VECTOR_SECRET, VECTOR_NONCE, {**frame, 'signature': None}, VECTOR_BODY)


@pytest.mark.parametrize('signature', ['ä' * len(VECTOR_SIGNATURE), VECTOR_SIGNATURE.upper(), 12345, ['x']])
def test_a_malformed_signature_is_invalid_not_an_error(signature):
    frame = {'id': 'req-1', 'backend': 'ollama', 'method': 'POST', 'path': '/chat/completions',
             'signature': signature}
    assert not protocol.signature_is_valid(VECTOR_SECRET, VECTOR_NONCE, frame, VECTOR_BODY)


def test_protocol_version_matches_the_gateway():
    assert protocol.PROTOCOL_VERSION == GATEWAY_PROTOCOL_VERSION


def test_backoff_grows_and_is_capped():
    assert backoff_delay(0, MIN_DELAY, MAX_DELAY, FULL_JITTER) == MIN_DELAY
    assert backoff_delay(1, MIN_DELAY, MAX_DELAY, FULL_JITTER) == MIN_DELAY * BACKOFF_FACTOR
    assert backoff_delay(LATE_ATTEMPT, MIN_DELAY, MAX_DELAY, FULL_JITTER) == MAX_DELAY
    assert backoff_delay(LATE_ATTEMPT, MIN_DELAY, MAX_DELAY, NO_JITTER) == MAX_DELAY * JITTER_FLOOR


def test_auth_proof_matches_the_gateway_implementation():
    assert protocol.sign_auth(VECTOR_SECRET, 'nonce-1', 'me-home') == VECTOR_AUTH_PROOF
    assert protocol.sign_auth(VECTOR_SECRET, 'nonce-2', 'me-home') != VECTOR_AUTH_PROOF
    assert protocol.sign_auth(VECTOR_SECRET, 'nonce-1', 'other') != VECTOR_AUTH_PROOF


@pytest.mark.parametrize('url, allow_insecure, message', [
    ('https://toptic.example/', False, 'wss://'),
    ('ws://127.0.0.1:8888/ws/socketllama/', False, 'wss://'),
    ('wss://', False, 'WebSocket URL'),
    ('wss://[::1', False, 'brackets'),
    ('wss://toptic.example:44x/ws/', False, 'port'),
    ('wss://toptic.example:65536/ws/', False, 'port'),
    ('toptic.example', True, 'WebSocket URL'),
])
def test_a_single_gateway_url_is_checked_like_the_one_of_a_file(url, allow_insecure, message):
    with pytest.raises(ConfigError, match=message):
        check_gateway_url(url, allow_insecure=allow_insecure)


def test_a_single_gateway_url_that_is_fine_passes():
    check_gateway_url(GATEWAY['url'], allow_insecure=False)
    check_gateway_url('ws://127.0.0.1:8888/ws/socketllama/', allow_insecure=True)


@pytest.mark.parametrize('name, message', [
    ('a/b', 'must not contain'),
    ('tab\there', 'must not contain'),
    ('b' * (protocol.BACKEND_NAME_MAX_LENGTH + 1), 'characters'),
])
def test_a_single_backend_name_is_checked_like_the_one_of_a_file(name, message):
    with pytest.raises(ConfigError, match=message):
        check_backend_name(name)


@pytest.mark.parametrize('base_url, message', [
    ('ftp://host/v1', 'http'),
    ('localhost:11434/v1', 'http'),
    ('http://[fe80::1/v1', 'brackets'),
    ('http://127.0.0.1:1143x/v1', 'port'),
    ('http://127.0.0.1:99999/v1', 'port'),
    ('http://user:pw@host/v1', 'api_key'),
])
def test_a_single_backend_address_is_checked_like_the_one_of_a_file(base_url, message):
    with pytest.raises(ConfigError, match=message):
        normalized_backend_url('ollama', base_url)


def test_a_backend_address_loses_its_trailing_slash():
    assert normalized_backend_url('ollama', 'http://localhost:11434/v1/') == 'http://localhost:11434/v1'


@pytest.mark.parametrize('check, reason', [
    (lambda: check_gateway_url('toptic.example', allow_insecure=True), config_module.REASON_GATEWAY_URL_MALFORMED),
    (lambda: check_gateway_url('wss://[::1', allow_insecure=True), config_module.REASON_ADDRESS_BRACKETS),
    (lambda: check_gateway_url('wss://toptic.example:44x/', allow_insecure=True), config_module.REASON_ADDRESS_PORT),
    (lambda: normalized_backend_url('b', 'http://[fe80::1/v1'), config_module.REASON_ADDRESS_BRACKETS),
    (lambda: normalized_backend_url('b', 'http://host:65536/v1'), config_module.REASON_ADDRESS_PORT),
    (lambda: check_gateway_url('https://toptic.example/', allow_insecure=True),
     config_module.REASON_GATEWAY_URL_SCHEME),
    (lambda: check_gateway_url('ws://toptic.example/', allow_insecure=False), config_module.REASON_GATEWAY_URL_SCHEME),
    (lambda: check_backend_name('a/b'), config_module.REASON_BACKEND_NAME_CHARACTERS),
    (lambda: check_backend_name('b' * (protocol.BACKEND_NAME_MAX_LENGTH + 1)),
     config_module.REASON_BACKEND_NAME_TOO_LONG),
    (lambda: normalized_backend_url('b', 'host:8000/v1'), config_module.REASON_BACKEND_URL_MALFORMED),
    (lambda: normalized_backend_url('b', 'http://host:80x/v1'), config_module.REASON_ADDRESS_PORT),
    (lambda: normalized_backend_url('b', 'http://me:pw@host/v1'), config_module.REASON_BACKEND_URL_CREDENTIALS),
])
def test_a_single_value_that_breaks_a_rule_names_the_rule_for_a_caller_with_its_own_words(check, reason):
    with pytest.raises(ConfigError) as raised:
        check()
    assert raised.value.reason == reason


def test_other_configuration_errors_name_no_rule():
    with pytest.raises(ConfigError) as raised:
        parse_config({'gateway': GATEWAY, 'backends': []}, environ={})
    assert raised.value.reason is None


def test_a_port_at_the_ends_of_the_range_is_one():
    assert normalized_backend_url('b', 'http://host:0/v1') == 'http://host:0/v1'
    assert normalized_backend_url('b', 'http://host:65535/v1') == 'http://host:65535/v1'
    assert normalized_backend_url('b', 'http://[::1]:11434/v1') == 'http://[::1]:11434/v1'


# ----- prices -----

# MAX_PRICE_PER_UNIT and SOCKETLLAMA_MAX_MODELS_PER_INSTANCE of toptic.ai's gateway
# (ai_helper/service/machine_prices.py and ai_helper/settings.py in the toptic.ai repository)
GATEWAY_MAX_PRICE_PER_UNIT = 1000.0
GATEWAY_MAX_MODELS_PER_INSTANCE = 100
PRICED = 'llama3.1:8b'
PRICE = {'input': 0.8, 'output': 2}


def priced(prices, **backend) -> dict:
    return {**OLLAMA, 'prices_per_1m_tokens': prices, **backend}


def test_price_bounds_match_the_gateway():
    assert protocol.MAX_PRICE_PER_UNIT == GATEWAY_MAX_PRICE_PER_UNIT
    assert protocol.MAX_SETTINGS_PER_FRAME == GATEWAY_MAX_MODELS_PER_INSTANCE


def test_a_backend_holds_the_prices_of_its_table():
    backend = config(backends=[priced({PRICED: PRICE})]).backends['ollama']
    assert backend.prices == {PRICED: config_module.ModelPrice(input=0.8, output=2.0)}
    assert isinstance(backend.prices[PRICED].output, float)


def test_without_a_table_a_backend_has_no_prices():
    assert config().backends['ollama'].prices == {}


@pytest.mark.parametrize('kind', config_module.BACKEND_KINDS)
def test_every_kind_of_backend_may_have_prices(kind):
    backend = config(backends=[priced({PRICED: PRICE}, kind=kind)]).backends['ollama']
    assert PRICED in backend.prices


def test_a_backend_that_shares_every_model_may_price_any():
    assert 'anything' in config(backends=[priced({'anything': PRICE}, models=[])]).backends['ollama'].prices


@pytest.mark.parametrize('prices, message', [
    ('cheap', 'must be a table'),
    ({PRICED: 0.8}, 'must be { input'),
    ({PRICED: {'input': 0.8}}, 'must be { input'),
    ({PRICED: {'input': 0.8, 'output': 2.0, 'currency': 'EUR'}}, 'must be { input'),
    ({PRICED: {'input': True, 'output': 2.0}}, 'input must be a number'),
    ({PRICED: {'input': '0.8', 'output': 2.0}}, 'input must be a number'),
    ({PRICED: {'input': -0.1, 'output': 2.0}}, 'input must be a number from 0'),
    ({PRICED: {'input': 0.8, 'output': GATEWAY_MAX_PRICE_PER_UNIT + 1}}, 'output must be a number from 0 to 1000'),
    ({PRICED: {'input': 0.8, 'output': 10 ** 400}}, 'output must be a number'),
    ({PRICED: {'input': float('inf'), 'output': 2.0}}, 'input must be a number'),
    ({PRICED: {'input': float('nan'), 'output': 2.0}}, 'input must be a number'),
])
def test_a_price_that_cannot_be_one_is_refused(prices, message):
    with pytest.raises(ConfigError, match=message.replace('{', '\\{')):
        config(backends=[priced(prices)])


def test_the_bounds_themselves_are_prices():
    bounds = {'input': protocol.MIN_PRICE_PER_UNIT, 'output': protocol.MAX_PRICE_PER_UNIT}
    assert config(backends=[priced({PRICED: bounds})]).backends['ollama'].prices[PRICED].output == 1000.0


def test_prices_are_the_same_as_toptic_compares_them():
    price = config_module.ModelPrice(input=0.1 + 0.2, output=1.0)
    assert price.same_as(config_module.ModelPrice(input=0.3, output=1.0))
    assert not price.same_as(config_module.ModelPrice(input=0.31, output=1.0))
    assert not price.same_as(None)


def test_the_file_a_configuration_was_read_from_is_known(tmp_path, monkeypatch):
    monkeypatch.setenv('SOCKETLLAMA_TOKEN', 'tt_live_example')
    monkeypatch.setenv('SOCKETLLAMA_SIGNING_SECRET', 'example-secret')
    monkeypatch.setenv('LITELLM_API_KEY', 'sk-example')
    assert load_config('config.example.toml').path == Path('config.example.toml')
    assert config().path is None


def test_the_price_table_as_the_example_configuration_shows_it_is_valid(monkeypatch):
    text = Path('config.example.toml').read_text()
    for commented in ('# [backends.prices_per_1m_tokens]', '# "llama3.1:8b" = { input = 0.80, output = 2.00 }'):
        assert text.count(commented) == 1
        text = text.replace(commented, commented.removeprefix('# '))
    for name in ('SOCKETLLAMA_TOKEN', 'SOCKETLLAMA_SIGNING_SECRET', 'LITELLM_API_KEY'):
        monkeypatch.setenv(name, 'example')
    ollama = parse_config(tomllib.loads(text)).backends['ollama']
    assert ollama.prices == {PRICED: config_module.ModelPrice(input=0.8, output=2.0)}


def test_a_price_line_for_a_model_no_longer_shared_says_it_can_simply_be_deleted():
    """What is left when the owner takes a model out of models after the client wrote its price line."""
    with pytest.raises(ConfigError) as refused:
        config(backends=[priced({'qwen3:14b': PRICE})])
    assert str(refused.value) == (
        'Backend ollama: the line "qwen3:14b" = ... under [backends.prices_per_1m_tokens] prices a model that is '
        'not among its models. If you no longer share that model, simply delete the line; if you do, check the '
        'name or add it to models')
