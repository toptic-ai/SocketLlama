"""``socketllama setup`` from the command line to the written file, against fake backends."""
import asyncio
import io
import logging
import os
import signal
import socket
import stat
import sys

import pytest

from socketllama import cli
from socketllama.backends import NOTE_DOWN, NOTE_MESSAGE, BackendUnavailable, DiscoveryNote
from socketllama.backends import logger as client_logger
from socketllama.config import KIND_LITELLM, KIND_OLLAMA, ConfigError, load_config
from socketllama.console import Terminal

from .fakes import (
    BACKEND_API_KEY, EPHEMERAL_PORT, HTTP_FORBIDDEN, INSTANCE, LOCALHOST, MODEL, SIGNING_SECRET, TOKEN, FakeBackend,
    ScriptedConsole, ScriptedDiscovery,
)

GATEWAY_URL = 'wss://toptic.example/ws/socketllama/'
# Never contacted: these tests ask backends through the scripted discovery or the fake backend only
OLLAMA_URL = 'http://ollama.invalid/v1'
LITELLM_URL = 'http://litellm.invalid/v1'
BACKEND_URLS = {KIND_OLLAMA: OLLAMA_URL, KIND_LITELLM: LITELLM_URL}
OWNER_ONLY = 0o600
WORLD_READABLE = 0o644
NOT_READABLE = 0o000
BACKUP_AGAIN = '1'
WITHOUT_BACKUP = '2'
STOP_WITHOUT_WRITING = '3'
READ_ONLY_DIRECTORY = 0o500
OLD_CONTENT = '# the configuration that was here before\n'
PASTE_BLOCK = '1'
END_OF_BLOCK = 'END'
TRY_AGAIN = '1'
OTHER_ADDRESS = '2'
ADD_LITELLM = '1'
NO_MORE_BACKENDS = '4'
DEFAULT = ''
YES = 'y'
NO = 'n'
PAGE_BLOCK = f'''[gateway]
url = "{GATEWAY_URL}"
instance = "{INSTANCE}"
token = "{TOKEN}"
signing_secret = "{SIGNING_SECRET}"

[[backends]]
name = "ollama"
kind = "ollama"
base_url = "http://127.0.0.1:11434/v1"
models = ["NAME-OF-A-MODEL-FROM-OLLAMA-LIST"]'''
PASTED_GATEWAY = [PASTE_BLOCK, *PAGE_BLOCK.splitlines(), END_OF_BLOCK]
# Up to the question whether to write: the block, the first model, no further backend
# Answers to the questions about an Ollama backend's memory that leave it to Ollama: how long models
# stay, and whether they are loaded at start (the second is not asked of a backend that shares everything)
LEAVE_MEMORY = [DEFAULT, DEFAULT]
LEAVE_MEMORY_OF_ALL = [DEFAULT]
UNTIL_WRITING = [*PASTED_GATEWAY, '1', *LEAVE_MEMORY, NO_MORE_BACKENDS]
FAKE_PROGRAM = '/opt/socketllama/.venv/bin/socketllama'
RUN_RESULT = 17


def scripted_setup(path, answers, discovery=None) -> tuple:
    """(exit code, console) of a setup run with scripted answers and scripted backends."""
    console = ScriptedConsole(answers)
    discovery = discovery or ScriptedDiscovery({OLLAMA_URL: [[MODEL, 'qwen3:14b']]})
    return cli.setup(str(path), console, discovery, BACKEND_URLS), console


def refused_url() -> str:
    """An address on this computer where nothing listens."""
    with socket.socket() as probe:
        probe.bind((LOCALHOST, EPHEMERAL_PORT))
        return f'http://{LOCALHOST}:{probe.getsockname()[1]}/v1'


@pytest.fixture(autouse=True)
def no_real_check(request, monkeypatch):
    """The check of the new file asks the configured backends: only the tests with the fake backend let it."""
    checked = []
    if {'backend', 'real_check'} & set(request.fixturenames):
        return checked

    async def fake_show_models(config):
        checked.append(config)
        return cli.EXIT_OK
    monkeypatch.setattr(cli, 'show_models', fake_show_models)
    return checked


@pytest.fixture
def real_check():
    """Asked for by a test whose backends are fake ones of its own: the check of the new file is the real one."""


@pytest.fixture
def no_real_run(monkeypatch):
    """Starting the client is recorded instead of done."""
    started = []

    async def fake_run(config):
        started.append(config)
        return RUN_RESULT
    monkeypatch.setattr(cli, 'run', fake_run)
    return started


def test_setup_refuses_without_a_terminal(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'stdin', io.StringIO('1\n'))
    path = tmp_path / 'config.toml'
    assert cli.main(['setup', '-c', str(path)]) == cli.EXIT_NOT_A_TERMINAL
    captured = capsys.readouterr()
    assert 'terminal' in captured.err and 'config.example.toml' in captured.err
    assert captured.out == ''
    assert not path.exists()


def test_setup_on_a_terminal_shows_the_banner_first_and_asks_on_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'is_interactive', lambda: True)
    monkeypatch.setattr(cli, 'show_banner', lambda stream: calls.append(('banner', stream)))
    monkeypatch.setattr(cli, 'setup', lambda path, console: calls.append(('setup', path, type(console))) or cli.EXIT_OK)
    assert cli.main(['setup']) == cli.EXIT_OK
    assert calls == [('banner', sys.stdout), ('setup', cli.DEFAULT_CONFIG_PATH, Terminal)]


def test_setup_writes_a_configuration_the_client_accepts(tmp_path, capsys):
    path = tmp_path / 'config.toml'
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, NO])
    assert code == cli.EXIT_OK and console.unused_answers == []
    assert stat.S_IMODE(path.stat().st_mode) == OWNER_ONLY
    config = load_config(path)
    assert (config.gateway_url, config.instance, config.token, config.signing_secret) == (
        GATEWAY_URL, INSTANCE, TOKEN, SIGNING_SECRET)
    assert config.backends['ollama'].models == (MODEL,)
    assert path.read_text().startswith('#')
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']
    shown = console.output + capsys.readouterr().out
    assert TOKEN not in shown and SIGNING_SECRET not in shown


async def test_setup_lists_what_a_real_discovery_finds_and_checks_the_new_file(tmp_path, backend, capsys, monkeypatch):
    """The discovery the command really uses, against the fake backend; setup runs beside its event loop."""
    monkeypatch.setattr(sys, 'argv', [FAKE_PROGRAM, 'setup'])
    path = tmp_path / 'config.toml'
    # the one fake backend answers as Ollama and as a LiteLLM proxy
    console = ScriptedConsole([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                               ADD_LITELLM, DEFAULT, backend.base_url, BACKEND_API_KEY, 'all',
                               NO_MORE_BACKENDS, YES, NO])
    code = await asyncio.to_thread(cli.setup, str(path), console,
                                   backend_urls={KIND_OLLAMA: backend.base_url, KIND_LITELLM: LITELLM_URL})
    assert code == cli.EXIT_OK and console.unused_answers == []
    assert f'1) {MODEL}' in console.output and '1) gpt-proxy' in console.output
    config = load_config(path)
    assert config.backends['ollama'].models == (MODEL,)
    assert config.backends['litellm'].models == ('gpt-proxy',)
    assert config.backends['litellm'].api_key == BACKEND_API_KEY
    checked = capsys.readouterr().out
    assert f'backend ollama (ollama): 1 model(s)\n  - {MODEL}' in checked
    assert 'backend litellm (litellm): 1 model(s)\n  - gpt-proxy' in checked
    assert f'{FAKE_PROGRAM} run -c {path}' in console.output
    for secret in (TOKEN, SIGNING_SECRET, BACKEND_API_KEY):
        assert secret not in console.output + checked


async def test_an_address_that_refuses_is_reported_and_another_one_works(tmp_path, backend):
    path = tmp_path / 'config.toml'
    refused = refused_url()
    console = ScriptedConsole([*PASTED_GATEWAY, TRY_AGAIN, OTHER_ADDRESS, backend.base_url, '1', *LEAVE_MEMORY,
                               NO_MORE_BACKENDS,
                               YES, NO])
    code = await asyncio.to_thread(cli.setup, str(path), console,
                                   backend_urls={KIND_OLLAMA: refused, KIND_LITELLM: LITELLM_URL})
    assert code == cli.EXIT_OK and console.unused_answers == []
    assert console.output.count(f'Nothing answers at {refused}') == 2
    assert load_config(path).backends['ollama'].base_url == backend.base_url


def test_setup_names_the_command_and_starts_the_client_when_asked(tmp_path, no_real_run, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [FAKE_PROGRAM, 'setup'])
    path = tmp_path / 'config.toml'
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, YES])
    assert code == RUN_RESULT
    assert [config.instance for config in no_real_run] == [INSTANCE]
    assert f'{FAKE_PROGRAM} run -c {path}' in console.output


def test_setup_does_not_start_the_client_unasked(tmp_path, no_real_run):
    code, _ = scripted_setup(tmp_path / 'config.toml', [*UNTIL_WRITING, YES, NO])
    assert code == cli.EXIT_OK and no_real_run == []


def test_setup_checks_the_file_it_wrote(tmp_path, no_real_check):
    code, console = scripted_setup(tmp_path / 'config.toml', [*UNTIL_WRITING, YES, NO])
    assert code == cli.EXIT_OK
    assert [config.instance for config in no_real_check] == [INSTANCE]
    assert 'not toptic.ai' in console.output


def test_the_start_command_is_the_one_this_program_was_started_with(monkeypatch):
    monkeypatch.setattr(sys, 'argv', [FAKE_PROGRAM, 'setup'])
    assert cli.start_command('/etc/sl/config.toml') == f'{FAKE_PROGRAM} run -c /etc/sl/config.toml'
    assert cli.start_command('/etc/sl/my config.toml') == f"{FAKE_PROGRAM} run -c '/etc/sl/my config.toml'"
    monkeypatch.setattr(sys, 'argv', ['/somewhere/socketllama/cli.py', 'setup'])
    assert cli.start_command('/etc/sl/config.toml') == (
        f'{sys.executable} -m socketllama.cli run -c /etc/sl/config.toml')


def test_the_start_command_works_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['.venv/bin/socketllama', 'setup'])
    assert cli.start_command('config.toml') == (
        f'{tmp_path}/.venv/bin/socketllama run -c {tmp_path}/config.toml')


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, EOFError])
@pytest.mark.parametrize('answers_before', [[], PASTED_GATEWAY[:3], PASTED_GATEWAY, UNTIL_WRITING])
def test_an_interruption_leaves_no_file_and_has_its_own_exit_code(tmp_path, interruption, answers_before):
    code, console = scripted_setup(tmp_path / 'config.toml', [*answers_before, interruption()])
    assert code == cli.EXIT_SETUP_CANCELLED
    assert list(tmp_path.iterdir()) == []
    assert 'Nothing was written' in console.shown[-1]


def test_an_interruption_leaves_an_existing_file_as_it_was(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    code, _ = scripted_setup(path, [*UNTIL_WRITING, YES, KeyboardInterrupt()])
    assert code == cli.EXIT_SETUP_CANCELLED
    assert path.read_text() == OLD_CONTENT
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']


def test_an_interruption_while_the_file_is_written_leaves_no_partial_file(tmp_path, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(cli.os, 'fsync', interrupted)
    code, console = scripted_setup(tmp_path / 'config.toml', [*UNTIL_WRITING, YES])
    assert code == cli.EXIT_SETUP_CANCELLED
    assert list(tmp_path.iterdir()) == []
    assert 'Nothing was written' in console.shown[-1]


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, EOFError])
def test_an_interruption_after_writing_says_that_the_file_is_there_and_setup_did_its_work(tmp_path, monkeypatch,
                                                                                         interruption):
    monkeypatch.setattr(sys, 'argv', [FAKE_PROGRAM, 'setup'])
    path = tmp_path / 'config.toml'
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, interruption()])
    assert code == cli.EXIT_OK
    assert load_config(path).instance == INSTANCE
    assert 'Nothing was written' not in console.output
    assert str(path) in console.shown[-1] and 'run -c' in console.shown[-1]


def test_declining_at_the_end_writes_nothing(tmp_path):
    code, console = scripted_setup(tmp_path / 'config.toml', [*UNTIL_WRITING, NO])
    assert code == cli.EXIT_SETUP_CANCELLED
    assert list(tmp_path.iterdir()) == []
    assert 'Nothing was written' in console.shown[-1]


def test_an_existing_file_is_not_replaced_without_a_yes(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    code, console = scripted_setup(path, [*UNTIL_WRITING, DEFAULT])
    assert code == cli.EXIT_SETUP_CANCELLED
    assert path.read_text() == OLD_CONTENT
    assert f'{path} already exists' in console.output


def test_a_replaced_file_is_kept_as_a_named_backup(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, DEFAULT, NO])
    assert code == cli.EXIT_OK
    backups = [entry for entry in tmp_path.iterdir() if entry != path]
    assert len(backups) == 1 and backups[0].name.startswith('config.toml.backup-')
    assert backups[0].read_text() == OLD_CONTENT
    assert str(backups[0]) in console.output
    assert load_config(path).instance == INSTANCE


def test_a_replaced_file_is_not_kept_when_the_backup_is_declined(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    code, _ = scripted_setup(path, [*UNTIL_WRITING, YES, NO, NO])
    assert code == cli.EXIT_OK
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']
    assert load_config(path).instance == INSTANCE


@pytest.mark.parametrize('target, reason', [
    ('absent/config.toml', 'does not exist'),
    ('locked/config.toml', 'may not'),
    ('folder', 'is a directory'),
])
def test_a_file_that_could_not_be_written_is_refused_before_any_question(tmp_path, capsys, target, reason):
    (tmp_path / 'folder').mkdir()
    locked = tmp_path / 'locked'
    locked.mkdir(mode=READ_ONLY_DIRECTORY)
    code, console = scripted_setup(tmp_path / target, [])
    assert code == cli.EXIT_CONFIG_ERROR
    assert console.shown == []
    assert reason in capsys.readouterr().err


def test_run_shows_the_banner_and_check_does_not(tmp_path, no_real_run, monkeypatch):
    path = tmp_path / 'config.toml'
    assert scripted_setup(path, [*UNTIL_WRITING, YES, NO])[0] == cli.EXIT_OK
    streams = []
    monkeypatch.setattr(cli, 'show_banner', streams.append)
    assert cli.main(['run', '-c', str(path)]) == RUN_RESULT
    assert streams == [sys.stdout]

    assert cli.main(['check', '-c', str(path)]) == cli.EXIT_OK
    assert streams == [sys.stdout]


def test_a_configuration_error_shows_no_banner(tmp_path, monkeypatch):
    streams = []
    monkeypatch.setattr(cli, 'show_banner', streams.append)
    assert cli.main(['run', '-c', str(tmp_path / 'absent.toml')]) == cli.EXIT_CONFIG_ERROR
    assert streams == []


def test_the_exit_codes_are_distinct():
    codes = [cli.EXIT_OK, cli.EXIT_CONFIG_ERROR, cli.EXIT_GATEWAY_REFUSED, cli.EXIT_SETUP_CANCELLED,
             cli.EXIT_NOT_A_TERMINAL]
    assert len(set(codes)) == len(codes)


def test_a_file_that_cannot_be_written_in_the_end_is_an_error_not_a_crash(tmp_path, monkeypatch, capsys):
    def no_space(*args):
        raise OSError('No space left on device')
    monkeypatch.setattr(cli.os, 'fsync', no_space)
    path = tmp_path / 'config.toml'
    code, _ = scripted_setup(path, [*UNTIL_WRITING, YES])
    assert code == cli.EXIT_CONFIG_ERROR
    assert list(tmp_path.iterdir()) == []
    assert f'{path} could not be written' in capsys.readouterr().err


# ----- the truth about what was written -----

def test_ctrl_c_at_the_instant_the_file_lands_is_still_told_truthfully(tmp_path, monkeypatch):
    """The interruption waits until the file is written and known to be: no moment in which it is
    there and setup says it is not."""
    monkeypatch.setattr(sys, 'argv', [FAKE_PROGRAM, 'setup'])
    real_write = cli.write_config_file

    def interrupted_write(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        return real_write(*args, **kwargs)
    monkeypatch.setattr(cli, 'write_config_file', interrupted_write)
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, YES])
    assert code == cli.EXIT_OK
    assert load_config(path).instance == INSTANCE
    backups = [entry for entry in tmp_path.iterdir() if entry != path]
    assert len(backups) == 1 and backups[0].read_text() == OLD_CONTENT
    assert 'Nothing was written' not in console.output
    assert str(path) in console.shown[-1] and str(backups[0]) in console.shown[-1]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_a_configuration_error_before_writing_does_not_claim_a_file(tmp_path, monkeypatch, capsys):
    def broken(data):
        raise ConfigError('gateway: token is required')
    monkeypatch.setattr(cli, 'render_config', broken)
    code, _ = scripted_setup(tmp_path / 'config.toml', [*UNTIL_WRITING, YES])
    assert code == cli.EXIT_CONFIG_ERROR
    message = capsys.readouterr().err
    assert 'Nothing was written' in message and 'just wrote' not in message
    assert list(tmp_path.iterdir()) == []


def test_a_configuration_error_in_the_file_written_says_that_it_is_there(tmp_path, monkeypatch, capsys):
    def broken(path):
        raise ConfigError('gateway: token is required')
    monkeypatch.setattr(cli, 'load_config', broken)
    path = tmp_path / 'config.toml'
    code, _ = scripted_setup(path, [*UNTIL_WRITING, YES])
    assert code == cli.EXIT_CONFIG_ERROR
    assert f'{path}, which setup just wrote' in capsys.readouterr().err


# ----- the backup -----

def test_the_backup_of_a_file_others_could_read_is_for_its_owner_only_and_the_risk_is_named(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(WORLD_READABLE)
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, DEFAULT, NO])
    assert code == cli.EXIT_OK
    backup = next(entry for entry in tmp_path.iterdir() if entry != path)
    assert stat.S_IMODE(backup.stat().st_mode) == OWNER_ONLY
    assert stat.S_IMODE(path.stat().st_mode) == OWNER_ONLY
    assert 'may have been read' in console.output and 'Profile → API tokens' in console.output


def test_a_backup_that_cannot_be_made_does_not_cost_the_answers(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(NOT_READABLE)
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, DEFAULT, BACKUP_AGAIN, WITHOUT_BACKUP, NO])
    assert code == cli.EXIT_OK and console.unused_answers == []
    assert console.output.count('could not be made') == 2
    assert 'Permission denied' in console.output and 'could not be written' not in console.output
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']
    assert load_config(path).instance == INSTANCE


def test_stopping_after_a_failed_backup_changes_nothing(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(NOT_READABLE)
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, DEFAULT, STOP_WITHOUT_WRITING])
    assert code == cli.EXIT_SETUP_CANCELLED
    assert 'Nothing was written' in console.shown[-1]
    path.chmod(OWNER_ONLY)
    assert path.read_text() == OLD_CONTENT
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']


# ----- a link in place of the file -----

def test_a_linked_configuration_is_written_through_and_the_link_stays(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [FAKE_PROGRAM, 'setup'])
    real = tmp_path / 'elsewhere' / 'real.toml'
    real.parent.mkdir()
    real.write_text(OLD_CONTENT)
    link = tmp_path / 'config.toml'
    link.symlink_to(real)
    code, console = scripted_setup(link, [*UNTIL_WRITING, YES, DEFAULT, NO])
    assert code == cli.EXIT_OK
    assert link.is_symlink() and load_config(real).instance == INSTANCE
    assert stat.S_IMODE(real.stat().st_mode) == OWNER_ONLY
    backups = [entry for entry in real.parent.iterdir() if entry != real]
    assert len(backups) == 1 and backups[0].read_text() == OLD_CONTENT
    assert f'{link} is a link to {real}; {real} will be replaced' in console.output
    assert f'run -c {link}' in console.output


def test_a_link_into_a_directory_that_cannot_be_written_is_refused_before_any_question(tmp_path, capsys):
    locked = tmp_path / 'locked'
    locked.mkdir(mode=READ_ONLY_DIRECTORY)
    link = tmp_path / 'config.toml'
    link.symlink_to(locked / 'real.toml')
    code, console = scripted_setup(link, [])
    assert code == cli.EXIT_CONFIG_ERROR and console.shown == []
    assert f'may not create files in {locked}' in capsys.readouterr().err


# ----- only setup's own sentences while it talks -----

def test_what_the_client_logs_is_collected_as_notes_and_the_log_is_as_before_afterwards(caplog):
    with caplog.at_level(logging.INFO):
        with cli.discovery_notes() as notes:
            client_logger.warning('Backend %s is down', 'b', extra={'discovery_note': DiscoveryNote(NOTE_DOWN, 'b')})
            client_logger.warning('Something %s foresaw', 'nobody\x9b')
            client_logger.info('Backend b (ollama): 1 model(s)')
        client_logger.warning('logged as ever')
    assert [(note.kind, note.backend, note.text) for note in notes] == [
        (NOTE_DOWN, 'b', 'Backend b is down'), (NOTE_MESSAGE, '', 'Something nobody\\x9b foresaw')]
    assert [record.getMessage() for record in caplog.records] == ['logged as ever']


async def test_setup_says_in_its_own_words_what_discovery_logs(tmp_path, caplog, capsys, real_check):
    """A proxy whose key may not read the model details: logged by the running client, said by setup."""
    path = tmp_path / 'config.toml'
    async with FakeBackend(model_info_status=HTTP_FORBIDDEN) as proxy:
        console = ScriptedConsole([*PASTED_GATEWAY, 'none', ADD_LITELLM, DEFAULT, DEFAULT, BACKEND_API_KEY, 'future',
                                   YES, NO_MORE_BACKENDS, YES, NO])
        with caplog.at_level(logging.INFO):
            code = await asyncio.to_thread(cli.setup, str(path), console,
                                           backend_urls={KIND_OLLAMA: proxy.base_url, KIND_LITELLM: proxy.base_url})
            assert code == cli.EXIT_OK and console.unused_answers == []
            assert [record for record in caplog.records if record.name == client_logger.name] == []
            # noted while asking and again by the check of the written file: said once
            assert console.output.count('The API key may not read the model details of litellm (HTTP 403') == 1
            assert 'list the models to share instead' not in console.output + capsys.readouterr().out
            # the same file checked by hand is told what sharing everything means
            assert await cli.check(load_config(path)) == cli.EXIT_OK
    assert 'list the models to share instead' in caplog.text


async def test_a_backend_that_is_gone_when_the_new_file_is_checked_is_told_in_a_sentence(tmp_path, real_check):
    path = tmp_path / 'config.toml'
    refused = refused_url()

    def gone_by_now(backend):
        return cli.Discovered(names=[MODEL])
    console = ScriptedConsole([*PASTED_GATEWAY, '1', *LEAVE_MEMORY, NO_MORE_BACKENDS, YES, NO])
    code = await asyncio.to_thread(cli.setup, str(path), console, gone_by_now,
                                   {KIND_OLLAMA: refused, KIND_LITELLM: LITELLM_URL})
    assert code == cli.EXIT_OK
    assert 'Backend ollama does not answer: nothing listens at its address.' in (
        console.output.split('Checking it')[1])


@pytest.mark.parametrize('failure', [OSError('No space left on device'), None])
def test_ctrl_c_during_a_write_that_fails_ends_setup_instead_of_being_lost(tmp_path, monkeypatch, failure):
    """With None the backup is what fails (the old file cannot be read): no menu about it then."""
    def interrupted_write(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGINT)
        if failure:
            raise failure
        return real_write(*args, **kwargs)
    real_write = cli.write_config_file
    monkeypatch.setattr(cli, 'write_config_file', interrupted_write)
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(NOT_READABLE if failure is None else OWNER_ONLY)
    code, console = scripted_setup(path, [*UNTIL_WRITING, YES, YES])
    assert code == cli.EXIT_SETUP_CANCELLED and console.unused_answers == []
    assert 'Nothing was written' in console.shown[-1]
    assert 'could not be made' not in console.output
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']


def test_a_link_to_itself_is_refused_with_the_reason_before_any_question(tmp_path, capsys):
    link = tmp_path / 'config.toml'
    link.symlink_to(link)
    code, console = scripted_setup(link, [])
    assert code == cli.EXIT_CONFIG_ERROR and console.shown == []
    message = capsys.readouterr().err
    assert str(link) in message and 'symbolic links' in message


def test_others_being_able_to_read_the_old_file_is_said_before_the_secrets_are_asked_for(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(WORLD_READABLE)
    _, console = scripted_setup(path, [KeyboardInterrupt()])
    assert 'may have been read' in console.output and 'before you go on' in console.output


async def test_what_is_decided_about_memory_is_written_explained_and_shown_by_the_check(tmp_path, backend, capsys):
    path = tmp_path / 'config.toml'
    console = ScriptedConsole([*PASTED_GATEWAY, '1', YES, '2', YES, NO_MORE_BACKENDS, YES, NO])
    code = await asyncio.to_thread(cli.setup, str(path), console,
                                   backend_urls={KIND_OLLAMA: backend.base_url, KIND_LITELLM: LITELLM_URL})
    assert code == cli.EXIT_OK and console.unused_answers == []
    written = load_config(path).backends['ollama']
    assert (written.keep_loaded, written.load_at_start) == ('2h', True)
    text = path.read_text()
    assert 'keep_loaded = "2h"' in text and 'load_at_start = true' in text and 'OLLAMA_KEEP_ALIVE' in text
    checked = capsys.readouterr().out
    assert 'kept in memory after a request through toptic.ai: for 2h' in checked
    assert 'loaded when the client starts: yes' in checked
    assert '/api/generate' not in [request['path'] for request in backend.requests]  # setup itself loads nothing
