"""The settings that decide how long Ollama keeps a backend's models in memory, as the configuration takes them."""
import math
import tomllib
from pathlib import Path

import pytest

from socketllama import config as config_module
from socketllama.config import (
    KIND_LITELLM, KIND_OLLAMA, KIND_OPENAI, ConfigError, check_keep_loaded, keep_loaded_seconds, parse_config,
)

GATEWAY = {'url': 'wss://toptic.example/ws/socketllama/', 'instance': 'me-home', 'token': 'tt_live_x',
           'signing_secret': 'secret'}
# Never contacted: these tests only parse
OLLAMA = {'name': 'ollama', 'kind': KIND_OLLAMA, 'base_url': 'http://ollama.invalid/v1', 'models': ['llama3.1:8b']}
MINUTE = 60.0
HOUR = 3600.0
# Digits Python's \d takes and Go's duration syntax does not (it takes 0-9 only)
FULLWIDTH_FIVE = '\N{FULLWIDTH DIGIT FIVE}'
ARABIC_INDIC_THREE = '\N{ARABIC-INDIC DIGIT THREE}'


def backend(**settings):
    return parse_config({'gateway': GATEWAY, 'backends': [{**OLLAMA, **settings}]}, environ={}).backends['ollama']


def test_without_the_settings_the_client_leaves_the_memory_to_ollama():
    plain = backend()
    assert plain.keep_loaded is None and plain.load_at_start is False


@pytest.mark.parametrize('value, seconds', [
    ('30m', 30 * MINUTE),
    ('2h', 2 * HOUR),
    ('24h', 24 * HOUR),
    ('1h30m', 1.5 * HOUR),
    ('1.5h', 1.5 * HOUR),
    ('90s', 90.0),
    ('500ms', 0.5),
    ('+5m', 5 * MINUTE),
    ('0', 0.0),
    ('0s', 0.0),
    (3600, HOUR),
    (0.5, 0.5),
    (0, 0.0),
])
def test_a_time_is_given_the_ways_ollama_takes_it(value, seconds):
    assert keep_loaded_seconds(value) == pytest.approx(seconds)
    kept = backend(keep_loaded=value).keep_loaded
    assert kept == value and type(kept) is type(value)  # passed on to Ollama as it was written


@pytest.mark.parametrize('value', [-1, -0.5, '-1m', '-1s', '-24h'])
def test_a_negative_time_means_until_ollama_stops(value):
    assert keep_loaded_seconds(value) < 0
    assert backend(keep_loaded=value).keep_loaded == value


@pytest.mark.parametrize('value', [
    '', ' ', '30', '30 m', '30min', 'm', '1h 30m', 'forever', '1d', '--1m', '1.2.3s', '.s', 'h1',
    f'{FULLWIDTH_FIVE}m', f'{ARABIC_INDIC_THREE}h', f'1.{FULLWIDTH_FIVE}s', f'1h{FULLWIDTH_FIVE}m',
    True, False, [30], {'minutes': 30}, math.nan,
])
def test_anything_else_is_refused_with_the_forms_that_are_taken(value):
    with pytest.raises(ValueError):
        keep_loaded_seconds(value)
    with pytest.raises(ConfigError, match='Backend ollama: keep_loaded .*"30m".*seconds.*-1') as raised:
        backend(keep_loaded=value)
    assert raised.value.reason == config_module.REASON_KEEP_LOADED


def test_a_single_time_is_checked_like_the_one_of_a_file():
    check_keep_loaded('ollama', '45m')
    with pytest.raises(ConfigError, match='keep_loaded'):
        check_keep_loaded('ollama', '45 minutes')


@pytest.mark.parametrize('kind', [KIND_LITELLM, KIND_OPENAI])
@pytest.mark.parametrize('setting', [{'keep_loaded': '30m'}, {'load_at_start': True}, {'load_at_start': False}])
def test_backends_that_do_not_load_models_themselves_refuse_the_settings(kind, setting):
    key = next(iter(setting))
    with pytest.raises(ConfigError, match=f'Backend ollama: {key} is only for backends of kind ollama'):
        backend(kind=kind, **setting)


def test_models_can_be_loaded_at_start():
    loading = backend(load_at_start=True, keep_loaded='2h')
    assert loading.load_at_start is True and loading.keep_loaded == '2h'
    assert backend(load_at_start=True).keep_loaded is None  # for as long as Ollama itself keeps them
    assert backend(load_at_start=False).load_at_start is False


@pytest.mark.parametrize('value', ['yes', 1, 0, 'true', None])
def test_loading_at_start_is_true_or_false(value):
    with pytest.raises(ConfigError, match='Backend ollama: load_at_start must be true or false'):
        backend(load_at_start=value)


def test_loading_whatever_the_server_has_is_refused():
    with pytest.raises(ConfigError, match='Backend ollama: load_at_start needs the models listed'):
        backend(load_at_start=True, models=[])
    assert backend(load_at_start=False, models=[]).shares_all_models
    assert backend(keep_loaded='30m', models=[]).keep_loaded == '30m'


@pytest.mark.parametrize('value', [0, 0.0, '0', '0s', '0m'])
def test_loading_at_start_what_is_unloaded_at_once_is_refused(value):
    with pytest.raises(ConfigError, match='Backend ollama: load_at_start with keep_loaded = '):
        backend(load_at_start=True, keep_loaded=value)


@pytest.mark.parametrize('value', ['9' * 5000 + 'x', '9' * 5000, '1m' * 2000 + 'x', '.' * 5000])
def test_a_text_of_any_length_is_judged_at_once(value):
    """Not by trying every way to split it: that would take longer than anybody waits."""
    with pytest.raises(ValueError):
        keep_loaded_seconds(value)


def test_the_settings_as_the_example_configuration_shows_them_are_valid(monkeypatch):
    text = Path('config.example.toml').read_text()
    for commented in ('# keep_loaded = "30m"', '# load_at_start = true'):
        assert text.count(commented) == 1
        text = text.replace(commented, commented.removeprefix('# '))
    for name in ('SOCKETLLAMA_TOKEN', 'SOCKETLLAMA_SIGNING_SECRET', 'LITELLM_API_KEY'):
        monkeypatch.setenv(name, 'example')
    ollama = parse_config(tomllib.loads(text)).backends['ollama']
    assert (ollama.keep_loaded, ollama.load_at_start) == ('30m', True)


@pytest.mark.parametrize('value', ['2562047h', 9223372036, 9223372036.0, '-2562047h', -9223372036, '153722867m'])
def test_the_longest_time_ollama_can_count_is_still_taken(value):
    assert backend(keep_loaded=value).keep_loaded == value


@pytest.mark.parametrize('value', [
    '2562048h', '3000000h', '-3000000h', '2562047h47m17s', 9223372037, 10 ** 30, -(10 ** 30), 1e300, -1e300,
    math.inf, -math.inf, '9' * 400 + 'h', '9' * 5000 + 'h', '1000000h' * 1000000,
])
def test_a_time_beyond_what_ollama_can_count_is_refused_and_the_limit_is_named(value):
    with pytest.raises(ValueError):
        keep_loaded_seconds(value)
    with pytest.raises(ConfigError, match='Backend ollama: keep_loaded is longer than Ollama can count.*at most '
                                          '2562047h.*-1') as raised:
        backend(keep_loaded=value)
    assert raised.value.reason == config_module.REASON_KEEP_LOADED_TOO_LONG


def test_a_whole_number_of_seconds_is_judged_as_the_whole_number_it_is():
    """Not through a float, which cannot tell the largest time from the one after it."""
    assert keep_loaded_seconds(config_module.MAX_KEEP_LOADED_SECONDS) == config_module.MAX_KEEP_LOADED_SECONDS
    assert type(keep_loaded_seconds(config_module.MAX_KEEP_LOADED_SECONDS)) is int
    with pytest.raises(ValueError):
        keep_loaded_seconds(config_module.MAX_KEEP_LOADED_SECONDS + 1)


@pytest.mark.parametrize('value, seconds', [('2562047h47m16s', 9223372036), ('1.5h', 5400), ('.5s', 0.5)])
def test_a_text_is_counted_exactly(value, seconds):
    assert keep_loaded_seconds(value) == seconds
