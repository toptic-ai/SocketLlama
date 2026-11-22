"""What the client shows to a person at a terminal: the banner, the questions of setup, and the format of
the lines the running client logs (their texts: events)."""
import getpass
import logging
import shutil
from typing import TextIO

from . import __version__
from .banner import BANNER, HYPHEN_LINES, LLAMA_COLUMN


def _width(art: str) -> int:
    return max(len(line) for line in art.splitlines())


def _split_banner() -> str:
    """BANNER on two rows, "Socket-" above "Llama", for a terminal too narrow for it on one."""
    lines = BANNER.splitlines()
    socket = [(line[:LLAMA_COLUMN].ljust(LLAMA_COLUMN) + hyphen).rstrip() for line, hyphen in zip(lines, HYPHEN_LINES)]
    llama = [line[LLAMA_COLUMN:].rstrip() for line in lines]
    return '\n'.join(socket + llama)


SPLIT_BANNER = _split_banner()
BANNER_WIDTH = _width(BANNER)
SPLIT_BANNER_WIDTH = _width(SPLIT_BANNER)
# Shown under the banner, and instead of it where not even the split banner fits
TITLE = f'SocketLlama {__version__}'
# The space between the banner and the title under it
BANNER_TITLE_GAP = '\n\n'


def terminal_columns() -> int:
    return shutil.get_terminal_size().columns


def banner_text(columns: int) -> str:
    """What a terminal of ``columns`` width shows: the banner on one row, else on two, else the title alone,
    so that it never wraps."""
    if columns >= BANNER_WIDTH:
        return f'{BANNER}{BANNER_TITLE_GAP}{TITLE}'
    if columns >= SPLIT_BANNER_WIDTH:
        return f'{SPLIT_BANNER}{BANNER_TITLE_GAP}{TITLE}'
    return TITLE


def show_banner(stream: TextIO) -> None:
    """Print the banner if ``stream`` is a terminal: it never goes into a pipe, a service's journal or a log file."""
    if stream.isatty():
        print(banner_text(terminal_columns()), file=stream)
        print(file=stream)


# Local time, to the second
LOG_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
# The client's own loggers (socketllama and those below it): their INFO lines are its events
CLIENT_LOGGER_NAME = 'socketllama'
LOGGER_NAME_SEPARATOR = '.'
# An event (INFO) is its time and its text; anything else names its level as well
EVENT_FORMAT = '%(asctime)s %(message)s'
LEVELLED_FORMAT = '%(asctime)s %(levelname)s %(message)s'


class LogFormatter(logging.Formatter):
    """Every line begins with the local time. Plain text without colours, so it reads the same in any
    terminal, a pipe, a service's journal or a log file."""

    def __init__(self) -> None:
        super().__init__(fmt=LEVELLED_FORMAT, datefmt=LOG_TIME_FORMAT)
        self._event_style = logging.PercentStyle(EVENT_FORMAT)

    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO and record.name.split(LOGGER_NAME_SEPARATOR)[0] == CLIENT_LOGGER_NAME:
            return self._event_style.format(record)
        return super().formatMessage(record)


class Terminal:
    """The Console of setup_wizard on the real terminal."""

    def say(self, text: str = '') -> None:
        print(text)

    def ask(self, prompt: str) -> str:
        return input(prompt)

    def ask_secret(self, prompt: str) -> str:
        return getpass.getpass(prompt)
