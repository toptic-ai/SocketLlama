"""Command line: ``socketllama setup`` asks for everything and writes config.toml,
``socketllama check`` validates it, ``socketllama run`` connects."""
import argparse
import asyncio
import contextlib
import dataclasses
import logging
import os
import shlex
import signal
import sys
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

import httpx

from . import __version__, protocol
from .backends import NOTE_ATTRIBUTE, NOTE_MESSAGE, DiscoveryNote, build_models_report, shareable_model_names
from .backends import logger as client_logger
from .client import WIRE_LOGGER_NAME, FatalGatewayError, TunnelClient, warn_about_shared_backends
from .config import KIND_OLLAMA, Backend, ConfigError, load_config
from .config_writer import BackupFailed, ConfigTarget, inspect_target, render_config, write_config_file
from .console import LogFormatter, Terminal, show_banner
from .events import price_in_words
from .ollama_memory import keep_loaded_in_words
from .setup_wizard import (
    BACKUP_SKIP, BACKUP_STOP, DEFAULT_BACKEND_URLS, Console, Discover, Discovered, SetupDeclined, SetupWizard,
)
from .text import printable

PROGRAM = 'socketllama'
COMMAND_RUN = 'run'
COMMAND_CHECK = 'check'
COMMAND_SETUP = 'setup'
COMMANDS = (
    (COMMAND_SETUP, 'ask for the connection and the models to share, and write the configuration file'),
    (COMMAND_RUN, 'connect to the gateway and relay requests'),
    (COMMAND_CHECK, 'validate the configuration and list the models of each backend'),
)
CONFIG_OPTION = '-c'
DEFAULT_CONFIG_PATH = 'config.toml'
EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_GATEWAY_REFUSED = 3
EXIT_SETUP_CANCELLED = 4
EXIT_NOT_A_TERMINAL = 5
NOISY_LOGGERS = ('httpx', 'httpcore', 'websockets', WIRE_LOGGER_NAME)
NOTHING_WRITTEN = 'Nothing was written.'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM, description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='debug logging (frames are logged as type and size, never their content)')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in COMMANDS:
        command = commands.add_parser(name, help=help_text)
        command.add_argument(CONFIG_OPTION, '--config', default=DEFAULT_CONFIG_PATH,
                             help=f'configuration file (default: {DEFAULT_CONFIG_PATH})')
    return parser


async def run(config) -> int:
    client = TunnelClient(config)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, client.stop)
    try:
        await client.run_forever()
    except FatalGatewayError as ex:
        logging.getLogger('socketllama').error('%s', ex)
        return EXIT_GATEWAY_REFUSED
    return EXIT_OK


async def check(config) -> int:
    warn_about_shared_backends(config)
    return await show_models(config)


async def show_models(config) -> int:
    """Ask every backend of the configuration and list the models it would offer, with their prices and descriptions
    in it."""
    async with httpx.AsyncClient(follow_redirects=False) as http:
        report = await build_models_report(config.backends, http)
    print(f'gateway:  {config.gateway_url}')
    print(f'instance: {config.instance}')
    for backend in report:
        print(f'backend {backend["name"]} ({backend["kind"]}): {len(backend["models"])} model(s)')
        configured = config.backends[backend['name']]
        for model in backend['models']:
            price = configured.prices.get(model['name'])
            print(f'  - {model["name"]}' + ('' if price is None else f': {price_in_words(price)}'))
            description = configured.descriptions.get(model['name'])
            if description is not None:
                print(f'    description: {printable(description)}')
        offered = {model['name'] for model in backend['models']}
        checked = backend.get(protocol.BACKEND_CHECKED_FIELD) is not False
        for in_table, done in ((configured.prices, 'priced'), (configured.descriptions, 'described')):
            not_offered = [name for name in in_table if name not in offered]
            if not_offered and checked:
                names = ', '.join(printable(name) for name in not_offered)
                print(f'  {done} in the configuration, but not offered now: {names}')
        if configured.kind == KIND_OLLAMA:
            print(f'  kept in memory after a request through toptic.ai: {keep_loaded_in_words(configured.keep_loaded)}')
            print(f'  loaded when the client starts: {"yes" if configured.load_at_start else "no"}')
    return EXIT_OK


class NoteCollector(logging.Handler):
    """Keeps what is logged as notes: the DiscoveryNote a record carries, or a note of a warning without one.

    Every note gets the log line as its text, made printable: whoever has no words of its own for
    a note can still say it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.notes: list[DiscoveryNote] = []

    def emit(self, record: logging.LogRecord) -> None:
        note = getattr(record, NOTE_ATTRIBUTE, None)
        text = printable(record.getMessage())
        if note is not None:
            self.notes.append(dataclasses.replace(note, text=text))
        elif record.levelno >= logging.WARNING:
            self.notes.append(DiscoveryNote(kind=NOTE_MESSAGE, backend='', text=text))


@contextlib.contextmanager
def discovery_notes() -> Iterator[list[DiscoveryNote]]:
    """What the client logs during the block, as notes instead of log lines.

    Setup is a conversation: it says in its own sentences what the running client would log
    (SetupWizard.say_notes). The lines themselves are only shown as well with --verbose.
    """
    collector = NoteCollector()
    propagates = client_logger.propagate
    client_logger.addHandler(collector)
    client_logger.propagate = client_logger.isEnabledFor(logging.DEBUG)
    try:
        yield collector.notes
    finally:
        client_logger.propagate = propagates
        client_logger.removeHandler(collector)


@contextlib.contextmanager
def interrupts_deferred() -> Iterator[None]:
    """Ctrl-C during the block takes effect when the block is done.

    For a step that must be over or not have happened when the interruption is told: writing the
    file and knowing that it is written. The interruption is never lost: if the block fails, it
    takes the place of the failure (the person wanted out; what failed is of no use to them).
    Signals reach the main thread only; elsewhere there is nothing to defer.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    interrupted = []
    previous = signal.signal(signal.SIGINT, lambda signum, frame: interrupted.append(signum))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if interrupted:
            raise KeyboardInterrupt


def discover_models(backend: Backend) -> Discovered:
    """Ask a backend which models it could share, for setup (its conversation does not run in an event loop)."""
    async def discover() -> list:
        async with httpx.AsyncClient(follow_redirects=False) as http:
            return await shareable_model_names(backend, http)
    with discovery_notes() as notes:
        names = asyncio.run(discover())
    return Discovered(names=names, notes=notes)


def start_command(config_path: str) -> str:
    """The command line that starts the client with this configuration, the way this program was
    started, with full paths: it works from any directory."""
    started_as = sys.argv[0]
    program = [os.path.abspath(started_as)] if Path(started_as).name == PROGRAM else [sys.executable, '-m',
                                                                                      __spec__.name]
    return shlex.join([*program, COMMAND_RUN, CONFIG_OPTION, os.path.abspath(config_path)])


def unwritable_reason(path: Path) -> str | None:
    """Why setup could not write the file ``path`` in the end, so that it says so before asking anything."""
    directory = path.parent
    if path.is_dir():
        return (f'{path} is a directory: name the configuration file with {CONFIG_OPTION}, '
                f'like {path / DEFAULT_CONFIG_PATH}')
    if not directory.is_dir():
        return f'The directory {directory} does not exist: create it first, or name another file with {CONFIG_OPTION}'
    if not os.access(directory, os.W_OK | os.X_OK):
        return f'You may not create files in {directory}: name a file elsewhere with {CONFIG_OPTION}'
    return None


def write_asking_about_backup(wizard: SetupWizard, target: ConfigTarget, text: str, keep_backup: bool,
                              written: list) -> None:
    """Write the file and add its backup (None without one) to ``written``.

    Ctrl-C waits until both are done, so ``written`` never lacks a file that is on disk. A backup
    that cannot be made is the person's to decide about (SetupDeclined if they stop).
    """
    while True:
        try:
            with interrupts_deferred():
                written.append(write_config_file(target.file, text, keep_backup))
            return
        except BackupFailed as ex:
            action = wizard.backup_failed(target, str(ex))
            if action == BACKUP_STOP:
                raise SetupDeclined
            keep_backup = action != BACKUP_SKIP


def setup(config_path: str, console: Console, discover: Discover = discover_models,
          backend_urls: Mapping = DEFAULT_BACKEND_URLS) -> int:
    """Ask for a configuration, write it, check it as ``check`` does, and start the client if asked to."""
    try:
        target = inspect_target(config_path)
        reason = unwritable_reason(target.file)
    except OSError as ex:
        reason = f'{config_path} cannot be used ({ex}): name another file with {CONFIG_OPTION}'
    if reason:
        print(f'Setup cannot write the configuration: {reason}.', file=sys.stderr)
        return EXIT_CONFIG_ERROR
    wizard = SetupWizard(console, discover, backend_urls)
    command = start_command(config_path)
    # The backup of the file written (None without one), once it is written: empty until then
    written = []
    try:
        plan = wizard.run(target)
        text = render_config(plan.data)
        try:
            write_asking_about_backup(wizard, target, text, plan.keep_backup, written)
        except OSError as ex:
            print(f'{target.file} could not be written: {ex}', file=sys.stderr)
            return EXIT_CONFIG_ERROR
        console.say()
        console.say(f'Wrote {target.file}.{_backup_in_words(written)}')
        config = load_config(target.file)
        console.say()
        console.say(f'Checking it the way "{PROGRAM} {COMMAND_CHECK}" does. This asks your backends, not toptic.ai:')
        with discovery_notes() as notes:
            asyncio.run(show_models(config))
        wizard.say_notes(notes)
        start_now = wizard.offer_start(command)
    except SetupDeclined:
        console.say()
        console.say(NOTHING_WRITTEN)
        return EXIT_SETUP_CANCELLED
    except (KeyboardInterrupt, EOFError):
        console.say()
        if not written:
            console.say(f'Setup stopped. {NOTHING_WRITTEN}')
            return EXIT_SETUP_CANCELLED
        # the configuration is complete: what was left was to look at it and to start the client
        console.say(f'Setup stopped. {target.file} is written.{_backup_in_words(written)} Start the client with: '
                    f'{command}')
        return EXIT_OK
    except ConfigError as ex:
        print(f'Configuration error in {target.file}, which setup just wrote: {ex}' if written
              else f'Configuration error: {ex}. {NOTHING_WRITTEN}', file=sys.stderr)
        return EXIT_CONFIG_ERROR
    return asyncio.run(run(config)) if start_now else EXIT_OK


def _backup_in_words(written: list) -> str:
    """Where the file that was in place went, for the sentence that says the new one is written."""
    return ''.join(f' The file that was there is now {backup}.' for backup in written if backup)


def is_interactive() -> bool:
    """Whether a person can be asked: questions are shown on and answered from a terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Events at INFO (see events), the details with -v
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, handlers=[console_handler])
    if not args.verbose:
        # httpx logs every request at INFO; one line per relayed chunk source is noise in a daemon log
        for noisy in NOISY_LOGGERS:
            logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.command == COMMAND_SETUP:
        if not is_interactive():
            print(f'"{PROGRAM} {COMMAND_SETUP}" asks questions, so it needs a terminal: run it in a terminal window, '
                  f'not through a pipe or a service. To configure without it, copy config.example.toml to '
                  f'{DEFAULT_CONFIG_PATH} and edit it (see README.md).', file=sys.stderr)
            return EXIT_NOT_A_TERMINAL
        show_banner(sys.stdout)
        return setup(args.config, Terminal())
    try:
        config = load_config(args.config)
    except ConfigError as ex:
        print(f'Configuration error: {ex}', file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if args.command == COMMAND_RUN:
        show_banner(sys.stdout)
        return asyncio.run(run(config))
    return asyncio.run(check(config))


if __name__ == '__main__':
    sys.exit(main())
