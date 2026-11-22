"""The configuration file setup writes: valid for load_config, explained by comments, never readable by others."""
import fnmatch
import os
import stat
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from socketllama import config_writer
from socketllama.config import load_config, parse_config
from socketllama.config_writer import (
    BackupFailed, backup_candidates, inspect_target, render_config, write_config_file,
)

from .fakes import INSTANCE, MODEL, SIGNING_SECRET, TOKEN

OWNER_ONLY = 0o600
WORLD_READABLE = 0o644
GROUP_READABLE = 0o640
NOT_READABLE = 0o000
NO_UMASK = 0
EVERYTHING_MASKED = 0o777
GATEWAY = {'url': 'wss://toptic.example/ws/socketllama/', 'instance': INSTANCE, 'token': TOKEN,
           'signing_secret': SIGNING_SECRET}
OLLAMA = {'name': 'ollama', 'kind': 'ollama', 'base_url': 'http://backend.invalid/v1', 'models': [MODEL, 'qwen3:14b']}
PROXY = {'name': 'litellm', 'kind': 'litellm', 'base_url': 'https://llm.invalid/v1', 'api_key': 'sk-proxy-key',
         'models': []}
DATA = {'gateway': GATEWAY, 'backends': [OLLAMA, PROXY]}
HOSTILE = 'quote " backslash \\ newline \n tab \t bell \x07 delete \x7f ünïcödé 🦙 # [gateway]'
OLD_CONTENT = '# the file that was here before\n'
NEW_CONTENT = render_config(DATA)
WRITTEN_AT = datetime(2026, 11, 18, 9, 30, 5)
BACKUP_NAME = 'config.toml.backup-20261118-093005'


def file_mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def comment_above(lines: list, line: str) -> str:
    """The comment lines directly above ``line``, as one text."""
    above = lines[:lines.index(line)]
    comment = []
    while above and above[-1].startswith('#'):
        comment.insert(0, above.pop().removeprefix('#').strip())
    return ' '.join(comment)


def test_the_rendered_file_reads_back_as_the_data():
    assert tomllib.loads(render_config(DATA)) == DATA


def test_hostile_values_read_back_unchanged():
    data = {'gateway': {**GATEWAY, 'instance': HOSTILE, 'token': HOSTILE, 'signing_secret': HOSTILE,
                        'allow_insecure': True},
            'client': {'ping_interval': 12.5, 'max_concurrent_requests': 2},
            'backends': [{**PROXY, 'api_key': HOSTILE, 'models': [HOSTILE, MODEL], 'context_length': 16384}]}
    assert tomllib.loads(render_config(data)) == data


def test_the_rendered_file_is_a_valid_configuration():
    config = parse_config(tomllib.loads(render_config(DATA)), environ={})
    assert config.backends['ollama'].models == (MODEL, 'qwen3:14b')
    assert config.backends['litellm'].shares_all_models


def test_comments_explain_the_sections_and_what_the_model_lists_mean():
    lines = render_config(DATA).splitlines()
    assert lines[0].startswith('#')
    assert any(line.startswith('#') and 'chmod 600' in line for line in lines)
    before_gateway = lines[:lines.index('[gateway]')]
    assert before_gateway
    assert 'Only these models are shared' in comment_above(lines, 'models = ["llama3.1:8b", "qwen3:14b"]')
    assert 'including any you pull later' in comment_above(lines, 'models = []')


def test_the_gateway_comment_points_to_my_machines_on_toptic():
    comments = ' '.join(line.removeprefix('#').strip() for line in render_config(DATA).splitlines()
                        if line.startswith('#'))
    assert "machine's page on toptic.ai (account menu, My Machines)" in comments
    assert 'SocketLlama' not in comments


def test_every_comment_line_fits_the_width_comments_are_wrapped_to():
    assert all(len(line) <= config_writer.COMMENT_WIDTH for line in render_config(DATA).splitlines()
               if line.startswith('#'))


def test_a_section_the_configuration_does_not_have_is_refused():
    with pytest.raises(ValueError, match='surprise'):
        render_config({**DATA, 'surprise': {}})


def test_a_new_file_is_written_for_its_owner_only(tmp_path):
    path = tmp_path / 'config.toml'
    previous_umask = os.umask(NO_UMASK)
    try:
        assert write_config_file(path, NEW_CONTENT, keep_backup=True) is None
    finally:
        os.umask(previous_umask)
    assert path.read_text() == NEW_CONTENT
    assert file_mode(path) == OWNER_ONLY
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']
    assert load_config(path).token == TOKEN


def test_the_file_is_never_readable_by_others_not_even_before_it_gets_its_name(tmp_path, monkeypatch):
    path = tmp_path / 'config.toml'
    modes = []
    real_replace = os.replace
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(**kwargs):
        descriptor, name = real_mkstemp(**kwargs)
        modes.append(stat.S_IMODE(os.stat(name).st_mode))
        return descriptor, name

    def recording_replace(source, target):
        modes.append(stat.S_IMODE(os.stat(source).st_mode))
        real_replace(source, target)
    monkeypatch.setattr(tempfile, 'mkstemp', recording_mkstemp)
    monkeypatch.setattr(os, 'replace', recording_replace)
    previous_umask = os.umask(NO_UMASK)
    try:
        write_config_file(path, NEW_CONTENT, keep_backup=False)
    finally:
        os.umask(previous_umask)
    assert modes == [OWNER_ONLY, OWNER_ONLY]  # when it is created, and when it gets its name


def test_while_the_backup_is_made_the_file_never_leaves_its_place(tmp_path, monkeypatch):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    in_place = []
    real_replace = os.replace

    def recording_replace(source, target):
        in_place.append(path.read_text())
        real_replace(source, target)
    monkeypatch.setattr(os, 'replace', recording_replace)
    write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    assert in_place == [OLD_CONTENT]


def test_the_file_that_was_there_is_kept_as_a_backup(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(OWNER_ONLY)
    backup = write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    assert backup == tmp_path / BACKUP_NAME
    assert backup.read_text() == OLD_CONTENT
    assert file_mode(backup) == OWNER_ONLY
    assert path.read_text() == NEW_CONTENT
    assert file_mode(path) == OWNER_ONLY


def test_an_earlier_backup_is_never_overwritten(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    earlier = tmp_path / BACKUP_NAME
    earlier.write_text('an earlier backup')
    backup = write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    assert backup != earlier and backup.name.startswith(BACKUP_NAME)
    assert earlier.read_text() == 'an earlier backup'
    assert backup.read_text() == OLD_CONTENT


def test_backup_names_say_what_they_are_and_when_and_never_run_out(tmp_path):
    names = backup_candidates(tmp_path / 'config.toml', WRITTEN_AT)
    assert [next(names).name for _ in range(3)] == [BACKUP_NAME, f'{BACKUP_NAME}-2', f'{BACKUP_NAME}-3']


@pytest.mark.parametrize('old_mode', [OWNER_ONLY, WORLD_READABLE])
def test_the_backup_is_for_its_owner_only_whatever_the_old_file_allowed(tmp_path, monkeypatch, old_mode):
    """Checked whenever something is flushed to disk or renamed: not even for a moment."""
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(old_mode)
    seen = []
    real_fsync, real_replace = os.fsync, os.replace

    def record():
        seen.extend(file_mode(entry) for entry in tmp_path.iterdir() if entry != path)

    def recording_fsync(descriptor):
        record()
        real_fsync(descriptor)

    def recording_replace(source, target):
        record()
        real_replace(source, target)
    monkeypatch.setattr(os, 'fsync', recording_fsync)
    monkeypatch.setattr(os, 'replace', recording_replace)
    previous_umask = os.umask(NO_UMASK)
    try:
        backup = write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    finally:
        os.umask(previous_umask)
    assert file_mode(backup) == OWNER_ONLY and backup.read_text() == OLD_CONTENT
    assert seen and set(seen) == {OWNER_ONLY}


def test_the_backup_needs_no_hard_links(tmp_path, monkeypatch):
    """Not every file system has them, and Linux refuses one to a file of another user."""
    def no_links(*args, **kwargs):
        raise PermissionError('Operation not permitted')
    monkeypatch.setattr(os, 'link', no_links)
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    backup = write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    assert backup.read_text() == OLD_CONTENT and path.read_text() == NEW_CONTENT


def test_a_backup_that_cannot_be_made_is_named_as_what_failed_and_changes_nothing(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(NOT_READABLE)
    with pytest.raises(BackupFailed, match='Permission denied'):
        write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']
    path.chmod(OWNER_ONLY)
    assert path.read_text() == OLD_CONTENT


def test_an_interrupted_backup_leaves_no_half_copy(tmp_path, monkeypatch):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(config_writer.shutil, 'copyfileobj', interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']
    assert path.read_text() == OLD_CONTENT


# ----- what is at the place the file goes to -----

def test_a_place_without_a_file():
    target = inspect_target('config-that-is-not-there.toml')
    assert (target.named, target.file) == (Path('config-that-is-not-there.toml'),) * 2
    assert not target.exists and not target.is_link and not target.readable_by_others


@pytest.mark.parametrize('mode, readable_by_others', [
    (OWNER_ONLY, False), (WORLD_READABLE, True), (GROUP_READABLE, True), (0o604, True),
    # others may write or run it, which is not reading it
    (0o602, False), (0o610, False), (0o620, False), (0o611, False),
])
def test_a_file_in_place_tells_whether_others_could_read_it(tmp_path, mode, readable_by_others):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(mode)
    target = inspect_target(path)
    assert target.exists and target.file == path and not target.is_link
    assert target.readable_by_others is readable_by_others


def test_a_link_names_the_file_it_leads_to(tmp_path):
    real = tmp_path / 'elsewhere' / 'real.toml'
    real.parent.mkdir()
    real.write_text(OLD_CONTENT)
    real.chmod(WORLD_READABLE)
    link = tmp_path / 'config.toml'
    link.symlink_to(real)
    target = inspect_target(link)
    assert (target.named, target.file) == (link, real)
    assert target.is_link and target.exists and target.readable_by_others


def test_a_link_to_nothing_names_the_file_that_would_be_written(tmp_path):
    link = tmp_path / 'config.toml'
    link.symlink_to(tmp_path / 'not-there.toml')
    target = inspect_target(link)
    assert target.is_link and not target.exists and target.file == tmp_path / 'not-there.toml'


def test_without_a_backup_the_old_file_is_replaced(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(WORLD_READABLE)
    assert write_config_file(path, NEW_CONTENT, keep_backup=False) is None
    assert path.read_text() == NEW_CONTENT
    assert file_mode(path) == OWNER_ONLY
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, OSError])
@pytest.mark.parametrize('failing', ['fsync', 'replace'])
def test_an_interrupted_write_leaves_no_partial_file_and_the_old_one_in_place(tmp_path, monkeypatch, interruption,
                                                                             failing):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)

    def interrupted(*args):
        raise interruption()
    monkeypatch.setattr(os, failing, interrupted)
    with pytest.raises(interruption):
        write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']
    assert path.read_text() == OLD_CONTENT


def test_an_interrupted_first_write_leaves_nothing(tmp_path, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(os, 'fsync', interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_config_file(tmp_path / 'config.toml', NEW_CONTENT, keep_backup=True)
    assert list(tmp_path.iterdir()) == []


def test_a_path_without_a_directory_is_written_in_the_current_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_config_file('config.toml', NEW_CONTENT, keep_backup=True)
    assert (tmp_path / 'config.toml').read_text() == NEW_CONTENT


def test_what_setup_leaves_next_to_config_toml_is_git_ignored_like_the_file(tmp_path, monkeypatch):
    """The backup holds the old secrets, and so does the temporary file if the computer stops while it exists."""
    patterns = [line for line in Path('.gitignore').read_text().splitlines() if line and not line.startswith('#')]
    names = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(**kwargs):
        descriptor, name = real_mkstemp(**kwargs)
        names.append(Path(name).name)
        return descriptor, name
    monkeypatch.setattr(tempfile, 'mkstemp', recording_mkstemp)
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    names.append(write_config_file(path, NEW_CONTENT, keep_backup=True).name)
    assert len(names) == 2
    for name in ['config.toml', *names]:
        assert any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns), name


def test_a_link_to_itself_cannot_be_looked_at(tmp_path):
    link = tmp_path / 'config.toml'
    link.symlink_to(link)
    with pytest.raises(OSError, match='symbolic links'):
        inspect_target(link)


def test_whatever_the_umask_the_new_file_and_the_backup_are_for_their_owner(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    previous_umask = os.umask(EVERYTHING_MASKED)
    try:
        backup = write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT)
    finally:
        os.umask(previous_umask)
    assert file_mode(backup) == OWNER_ONLY and backup.read_text() == OLD_CONTENT
    assert file_mode(path) == OWNER_ONLY and path.read_text() == NEW_CONTENT


@pytest.mark.parametrize('keep_loaded', ['30m', 7200, -1, 0.5])
def test_the_memory_settings_of_an_ollama_backend_are_written_with_what_they_mean(keep_loaded):
    data = {'gateway': GATEWAY, 'backends': [{**OLLAMA, 'keep_loaded': keep_loaded, 'load_at_start': True}]}
    text = render_config(data)
    assert tomllib.loads(text) == data
    parsed = parse_config(tomllib.loads(text), environ={}).backends['ollama']
    assert (parsed.keep_loaded, parsed.load_at_start) == (keep_loaded, True)
    lines = text.splitlines()
    kept = comment_above(lines, next(line for line in lines if line.startswith('keep_loaded = ')))
    assert 'after a request through toptic.ai' in kept and 'OLLAMA_KEEP_ALIVE' in kept
    assert 'when the client starts' in comment_above(lines, 'load_at_start = true')


# ----- a file written again with the permissions it has (config_settings) -----

KEPT_MODES = (OWNER_ONLY, GROUP_READABLE, WORLD_READABLE)
OWNER_ONLY_UMASK = 0o077


@pytest.mark.parametrize('mode', KEPT_MODES)
def test_a_file_can_be_written_with_the_permissions_it_had_whatever_the_umask(tmp_path, mode):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    previous_umask = os.umask(OWNER_ONLY_UMASK)
    try:
        write_config_file(path, NEW_CONTENT, keep_backup=False, mode=mode)
    finally:
        os.umask(previous_umask)
    assert path.read_text() == NEW_CONTENT
    assert file_mode(path) == mode


def test_with_another_mode_the_file_is_still_its_owners_only_until_it_is_complete(tmp_path, monkeypatch):
    path = tmp_path / 'config.toml'
    modes_when_created = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(**kwargs):
        descriptor, name = real_mkstemp(**kwargs)
        modes_when_created.append(stat.S_IMODE(os.stat(name).st_mode))
        return descriptor, name
    real_fdopen = os.fdopen

    def recording_fdopen(descriptor, *args, **kwargs):
        handle = real_fdopen(descriptor, *args, **kwargs)
        real_handle_write = handle.write

        def write(text):
            modes_when_created.append(stat.S_IMODE(os.fstat(handle.fileno()).st_mode))
            return real_handle_write(text)
        handle.write = write
        return handle
    monkeypatch.setattr(tempfile, 'mkstemp', recording_mkstemp)
    monkeypatch.setattr(os, 'fdopen', recording_fdopen)
    write_config_file(path, NEW_CONTENT, keep_backup=False, mode=WORLD_READABLE)
    # created and written for its owner only; the mode it was given comes once the text is in
    assert modes_when_created == [OWNER_ONLY, OWNER_ONLY]
    assert file_mode(path) == WORLD_READABLE


@pytest.mark.parametrize('failing', ['fsync', 'replace'])
def test_a_failed_write_with_a_kept_mode_leaves_the_old_file_as_it_was(tmp_path, monkeypatch, failing):
    path = tmp_path / 'config.toml'
    path.write_text(OLD_CONTENT)
    path.chmod(GROUP_READABLE)

    def failed(*args):
        raise OSError('disk full')
    monkeypatch.setattr(os, failing, failed)
    with pytest.raises(OSError):
        write_config_file(path, NEW_CONTENT, keep_backup=True, now=WRITTEN_AT, mode=GROUP_READABLE)
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']
    assert path.read_text() == OLD_CONTENT and file_mode(path) == GROUP_READABLE
