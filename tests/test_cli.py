from socketllama import cli
from socketllama.console import LogFormatter

from .fakes import INSTANCE, MODEL, SIGNING_SECRET, TOKEN

OWNER_ONLY = 0o600


def write_config(tmp_path, backend_url: str) -> str:
    path = tmp_path / 'config.toml'
    path.write_text(f'''
[gateway]
url = "wss://toptic.example/ws/socketllama/"
instance = "{INSTANCE}"
token = "{TOKEN}"
signing_secret = "{SIGNING_SECRET}"

[[backends]]
name = "ollama"
kind = "ollama"
base_url = "{backend_url}"
models = ["{MODEL}"]
''')
    path.chmod(OWNER_ONLY)  # the file holds secrets inline
    return str(path)


def test_missing_config_is_a_config_error(tmp_path, capsys):
    assert cli.main(['check', '-c', str(tmp_path / 'absent.toml')]) == cli.EXIT_CONFIG_ERROR
    assert 'not found' in capsys.readouterr().err


async def test_check_lists_models_without_contacting_the_gateway(tmp_path, backend, capsys):
    config = cli.load_config(write_config(tmp_path, backend.base_url))
    assert await cli.check(config) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert f'- {MODEL}' in output
    assert TOKEN not in output and SIGNING_SECRET not in output


MEMORY_SETTINGS = '''keep_loaded = "2h"
load_at_start = true
'''


async def test_check_shows_what_an_ollama_backend_does_about_its_memory(tmp_path, backend, capsys):
    path = write_config(tmp_path, backend.base_url)
    with open(path, 'a') as config_file:
        config_file.write(MEMORY_SETTINGS)
    assert await cli.check(cli.load_config(path)) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert f'  - {MODEL}\n  kept in memory after a request through toptic.ai: for 2h\n' in output
    assert '  loaded when the client starts: yes\n' in output
    assert '/api/generate' not in [request['path'] for request in backend.requests]  # check loads nothing


async def test_check_says_when_the_memory_is_left_to_ollama(tmp_path, backend, capsys):
    assert await cli.check(cli.load_config(write_config(tmp_path, backend.base_url))) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert '  kept in memory after a request through toptic.ai: as long as Ollama itself keeps them\n' in output
    assert '  loaded when the client starts: no\n' in output


async def test_check_says_nothing_about_memory_for_backends_that_have_no_say_in_it(tmp_path, backend, capsys):
    path = tmp_path / 'config.toml'
    path.write_text(f'''
[gateway]
url = "wss://toptic.example/ws/socketllama/"
instance = "{INSTANCE}"
token = "{TOKEN}"
signing_secret = "{SIGNING_SECRET}"

[[backends]]
name = "proxy"
kind = "litellm"
base_url = "{backend.base_url}"
models = []
''')
    path.chmod(OWNER_ONLY)
    assert await cli.check(cli.load_config(path)) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert 'backend proxy (litellm)' in output and 'memory' not in output and 'loaded' not in output


def _logging_configured_by(monkeypatch, argv: list) -> dict:
    """What main() hands to logging.basicConfig for ``argv`` (a command that ends at once)."""
    configured = {}
    monkeypatch.setattr(cli.logging, 'basicConfig', lambda **options: configured.update(options))
    cli.main(argv)
    return configured


def test_the_console_shows_events_by_default_in_the_clients_own_format(monkeypatch, tmp_path):
    configured = _logging_configured_by(monkeypatch, ['check', '-c', str(tmp_path / 'absent.toml')])
    assert configured['level'] == cli.logging.INFO
    [handler] = configured['handlers']
    assert isinstance(handler.formatter, LogFormatter)


def test_verbose_adds_the_details(monkeypatch, tmp_path):
    configured = _logging_configured_by(monkeypatch, ['-v', 'check', '-c', str(tmp_path / 'absent.toml')])
    assert configured['level'] == cli.logging.DEBUG


NOT_PULLED = 'not-pulled:1b'
PRICES = f'''models = ["{MODEL}", "{NOT_PULLED}"]

[backends.prices_per_1m_tokens]
"{MODEL}" = {{ input = 0.8, output = 2 }}
"{NOT_PULLED}" = {{ input = 0, output = 0 }}
'''


async def test_check_shows_the_price_of_each_model_in_the_configuration(tmp_path, backend, capsys):
    path = tmp_path / 'config.toml'
    path.write_text(open(write_config(tmp_path, backend.base_url)).read().replace(f'models = ["{MODEL}"]\n', PRICES))
    assert await cli.check(cli.load_config(path)) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert f'  - {MODEL}: $0.80 in, $2.00 out per 1M tokens\n' in output
    # a price for a model the backend does not offer now does nothing until it does: said, not hidden
    assert f'  priced in the configuration, but not offered now: {NOT_PULLED}\n' in output


async def test_check_shows_no_price_for_a_model_without_one(tmp_path, backend, capsys):
    assert await cli.check(cli.load_config(write_config(tmp_path, backend.base_url))) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert f'  - {MODEL}\n' in output and 'per 1M tokens' not in output and 'priced' not in output


DESCRIPTIONS = f'''models = ["{MODEL}", "{NOT_PULLED}"]

[backends.descriptions]
"{MODEL}" = "Quick answers, in English and German"
"{NOT_PULLED}" = "Pulled next week"
'''


async def test_check_shows_the_description_of_each_model_in_the_configuration(tmp_path, backend, capsys):
    path = tmp_path / 'config.toml'
    path.write_text(open(write_config(tmp_path, backend.base_url)).read().replace(f'models = ["{MODEL}"]\n',
                                                                                  DESCRIPTIONS))
    assert await cli.check(cli.load_config(path)) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert f'  - {MODEL}\n    description: Quick answers, in English and German\n' in output
    assert f'  described in the configuration, but not offered now: {NOT_PULLED}\n' in output


async def test_check_shows_no_description_for_a_model_without_one(tmp_path, backend, capsys):
    assert await cli.check(cli.load_config(write_config(tmp_path, backend.base_url))) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert 'description:' not in output and 'described' not in output
