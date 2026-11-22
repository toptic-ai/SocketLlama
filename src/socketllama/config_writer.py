"""Writes a configuration file: what ``socketllama setup`` leaves behind.

The file holds the connection token and the signing secret, so it is created for its owner
only and gets its name once it is complete: there is no moment at which others can read it,
and none at which a half-written file, or no file, is in place of one that was there.
"""
import contextlib
import itertools
import os
import shutil
import tempfile
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import KEY_KEEP_LOADED, KEY_LOAD_AT_START, OWNER_ONLY_PERMISSIONS, READ_BY_OTHERS_PERMISSIONS
from .toml_writer import toml_assignment

SECTION_GATEWAY = 'gateway'
SECTION_CLIENT = 'client'
SECTION_BACKENDS = 'backends'
SECTIONS = (SECTION_GATEWAY, SECTION_CLIENT, SECTION_BACKENDS)
# Keys of the [gateway] section and of a [[backends]] entry
KEY_URL = 'url'
KEY_INSTANCE = 'instance'
KEY_TOKEN = 'token'
KEY_SIGNING_SECRET = 'signing_secret'
KEY_ALLOW_INSECURE = 'allow_insecure'
KEY_NAME = 'name'
KEY_KIND = 'kind'
KEY_BASE_URL = 'base_url'
KEY_API_KEY = 'api_key'
KEY_MODELS = 'models'

COMMENT_PREFIX = '# '
COMMENT_WIDTH = 100
BLOCK_SEPARATOR = '\n\n'
FILE_ENCODING = 'utf-8'

FILE_COMMENT = ('socketllama configuration, written by "socketllama setup". It holds your connection '
                'token and your signing secret: keep it readable by you only '
                f'(chmod {OWNER_ONLY_PERMISSIONS:o}), or the client refuses to start. config.example.toml explains '
                'every option; run "socketllama check" after changing this file by hand.')
GATEWAY_COMMENT = ("Where the client connects and which of your machines this is. The values come from the machine's "
                   'page on toptic.ai (account menu, My Machines).')
CLIENT_COMMENT = 'How the client keeps the connection alive and how many requests it relays at the same time.'
BACKENDS_COMMENT = ('The backends below are the complete allow-list: toptic.ai can only name one of them, it can never '
                    'make the client call another address.')
GATEWAY_KEY_COMMENTS = {
    KEY_TOKEN: "The connection token created on the machine's page. Revoke it under Profile, API tokens.",
    KEY_SIGNING_SECRET: "The machine's signing secret. It proves requests come from toptic.ai and is never sent.",
    KEY_ALLOW_INSECURE: 'Permits ws:// (not encrypted): only for a toptic.ai development server on this computer.',
}
BACKEND_KEY_COMMENTS = {
    KEY_BASE_URL: 'The address that serves /chat/completions.',
    KEY_API_KEY: 'Sent to this backend only, never to toptic.ai.',
    KEY_KEEP_LOADED: 'How long Ollama keeps a model in memory after a request through toptic.ai: a time like "30m" '
                     'or "2h", seconds, or -1 for as long as Ollama runs. Without this line Ollama decides (5 '
                     'minutes, or its OLLAMA_KEEP_ALIVE), as it does for everything else that uses it.',
    KEY_LOAD_AT_START: 'The models listed are loaded into memory when the client starts, so the first prompt does '
                       'not wait for it.',
}
LISTED_MODELS_COMMENT = ('Only these models are shared. models = [] would share every model the backend offers, '
                         'including any you pull later.')
EVERY_MODEL_COMMENT = ('Shares every chat model this backend offers, including any you pull later. To share only '
                       'some, list them: models = ["name", "other-name"]')

BACKUP_TIME_FORMAT = '%Y%m%d-%H%M%S'
BACKUP_NAME = '{name}.backup-{time}'
# Appended (from this number on) when a backup of the same second exists
FIRST_BACKUP_COUNTER = 2
TEMPORARY_PREFIX = '.{name}.'
TEMPORARY_SUFFIX = '.tmp'
# What os.fchown takes for "leave the owner as it is"
UNCHANGED_OWNER = -1


def render_config(data: dict) -> str:
    """``data``, shaped like a parsed configuration file, as the text of one, with comments."""
    unknown = [section for section in data if section not in SECTIONS]
    if unknown:
        raise ValueError(f'Not a section of the configuration: {", ".join(unknown)}')
    blocks = [_comment(FILE_COMMENT)]
    blocks.append(_table(f'[{SECTION_GATEWAY}]', GATEWAY_COMMENT, data[SECTION_GATEWAY], GATEWAY_KEY_COMMENTS))
    if data.get(SECTION_CLIENT):
        blocks.append(_table(f'[{SECTION_CLIENT}]', CLIENT_COMMENT, data[SECTION_CLIENT], {}))
    blocks.append(_comment(BACKENDS_COMMENT))
    for backend in data[SECTION_BACKENDS]:
        comments = {**BACKEND_KEY_COMMENTS,
                    KEY_MODELS: LISTED_MODELS_COMMENT if backend.get(KEY_MODELS) else EVERY_MODEL_COMMENT}
        blocks.append(_table(f'[[{SECTION_BACKENDS}]]', '', backend, comments))
    return BLOCK_SEPARATOR.join(blocks) + '\n'


def _table(header: str, comment: str, table: dict, key_comments: dict) -> str:
    lines = [header] + ([_comment(comment)] if comment else [])
    for key, value in table.items():
        if key in key_comments:
            lines.append(_comment(key_comments[key]))
        lines.append(toml_assignment(key, value))
    return '\n'.join(lines)


def _comment(text: str) -> str:
    return '\n'.join(textwrap.wrap(text, width=COMMENT_WIDTH, initial_indent=COMMENT_PREFIX,
                                   subsequent_indent=COMMENT_PREFIX))


class BackupFailed(Exception):
    """The file in place could not be copied to a backup; the message is the operating system's reason.

    Nothing was changed: the file in place is as it was and no new one was written.
    """


@dataclass(frozen=True)
class ConfigTarget:
    """What is at the place a configuration file is to be written to."""
    # The path as it was named
    named: Path
    # The file itself: what ``named`` leads to if it is a symbolic link, else ``named``
    file: Path
    is_link: bool
    exists: bool
    # Whether anybody but the owner may read the file in place: its secrets may have been read
    readable_by_others: bool


def inspect_target(path: str | os.PathLike) -> ConfigTarget:
    """What is at ``path``. OSError if that cannot be found out (a link that leads to itself, for one)."""
    named = Path(path)
    is_link = named.is_symlink()
    file = Path(os.path.realpath(named)) if is_link else named
    exists = os.path.lexists(file)
    return ConfigTarget(named=named, file=file, is_link=is_link, exists=exists,
                        readable_by_others=exists and bool(file.stat().st_mode & READ_BY_OTHERS_PERMISSIONS))


def backup_candidates(path: str | os.PathLike, now: datetime) -> Iterator[Path]:
    """Names next to ``path`` that say what the file is and when it was made: the plain one, then numbered ones."""
    path = Path(path)
    name = BACKUP_NAME.format(name=path.name, time=now.strftime(BACKUP_TIME_FORMAT))
    numbered = (path.with_name(f'{name}-{counter}') for counter in itertools.count(FIRST_BACKUP_COUNTER))
    return itertools.chain([path.with_name(name)], numbered)


def _copy_for_owner_only(source: Path, candidates: Iterator[Path]) -> Path:
    """Copy ``source`` to the first of ``candidates`` that is free. The copy is created for its owner
    only, whatever ``source`` allows, and a name that is taken is never written to."""
    for candidate in candidates:
        try:
            descriptor = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, OWNER_ONLY_PERMISSIONS)
        except FileExistsError:
            continue
        try:
            with os.fdopen(descriptor, 'wb') as copy, open(source, 'rb') as original:
                os.fchmod(copy.fileno(), OWNER_ONLY_PERMISSIONS)  # whatever the umask made of the mode asked for
                shutil.copyfileobj(original, copy)
                copy.flush()
                os.fsync(copy.fileno())
        except BaseException:
            os.unlink(candidate)
            raise
        return candidate


def write_config_file(path: str | os.PathLike, text: str, keep_backup: bool,
                      now: datetime | None = None, mode: int = OWNER_ONLY_PERMISSIONS,
                      group: int | None = None) -> Path | None:
    """Write ``text`` to ``path``, for its owner only unless ``mode`` says otherwise; returns the backup of the file
    that was there.

    With ``keep_backup`` a file in place is first copied to a backup that only its owner can read
    (see backup_candidates); BackupFailed if that cannot be done. Whatever interrupts the
    writing, ``path`` holds the old file or the complete new one, never none and never half of one.
    ``path`` is the file itself: of a symbolic link, what it leads to (see inspect_target).
    ``mode``: the permissions the file gets. Setup writes it for its owner only; a change to a file in
    place (config_settings) passes the mode that file has, so it keeps the permissions its owner gave it.
    Until the file is complete it is its owner's only, whatever ``mode`` says. ``group``: the group the file
    gets (that of the file in place, for such a change); PermissionError if this user may not give it that one.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=TEMPORARY_PREFIX.format(name=path.name),
                                             suffix=TEMPORARY_SUFFIX)
    backup = None
    try:
        with os.fdopen(descriptor, 'w', encoding=FILE_ENCODING) as handle:
            os.fchmod(handle.fileno(), OWNER_ONLY_PERMISSIONS)  # whatever the umask made of mkstemp's mode
            if group is not None and group != os.fstat(handle.fileno()).st_gid:
                os.fchown(handle.fileno(), UNCHANGED_OWNER, group)
            handle.write(text)
            handle.flush()
            if mode != OWNER_ONLY_PERMISSIONS:
                os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        if keep_backup and os.path.lexists(path):
            try:
                backup = _copy_for_owner_only(path, backup_candidates(path, now or datetime.now()))
            except OSError as ex:
                raise BackupFailed(str(ex)) from ex
        os.replace(temporary, path)
    except BaseException:
        for leftover in filter(None, (temporary, backup)):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(leftover)
        raise
    return backup
