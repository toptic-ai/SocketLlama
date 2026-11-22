"""The banner is for terminals wide enough to show it, and never for pipes or log files."""
import io

from socketllama import __version__, console
from socketllama.banner import BANNER, HYPHEN_LINES, LLAMA_COLUMN
from socketllama.console import (BANNER_WIDTH, SPLIT_BANNER, SPLIT_BANNER_WIDTH, TITLE, Terminal, banner_text,
                                        show_banner)

OWNERS_BANNER_WIDTH = 107
TOO_NARROW_FOR_ONE_ROW = BANNER_WIDTH - 1
NARROW = SPLIT_BANNER_WIDTH - 1
ANSWER = 'an answer'


class FakeTerminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_the_banner_is_as_wide_as_its_longest_line():
    assert BANNER_WIDTH == OWNERS_BANNER_WIDTH


def test_a_terminal_wide_enough_gets_the_banner_exactly():
    assert banner_text(BANNER_WIDTH).startswith(BANNER)
    assert __version__ in banner_text(BANNER_WIDTH)


def test_an_empty_line_stands_between_the_banner_and_the_title():
    for columns in (BANNER_WIDTH, SPLIT_BANNER_WIDTH):
        assert banner_text(columns).endswith(f'\n\n{TITLE}')


def test_a_terminal_too_narrow_for_one_row_gets_socket_and_llama_on_two():
    text = banner_text(TOO_NARROW_FOR_ONE_ROW)
    assert text.startswith(SPLIT_BANNER) and __version__ in text
    assert all(len(line) <= TOO_NARROW_FOR_ONE_ROW for line in text.splitlines())


def test_the_two_rows_are_the_owners_art_cut_before_llama_with_a_hyphen_after_socket():
    art = BANNER.splitlines()
    socket, llama = SPLIT_BANNER.splitlines()[:len(art)], SPLIT_BANNER.splitlines()[len(art):]
    assert len(llama) == len(art) == len(HYPHEN_LINES)
    for art_line, socket_line, llama_line, hyphen in zip(art, socket, llama, HYPHEN_LINES):
        assert socket_line == (art_line[:LLAMA_COLUMN].ljust(LLAMA_COLUMN) + hyphen).rstrip()
        assert (art_line[:LLAMA_COLUMN] + llama_line).rstrip() == art_line.rstrip()


def test_the_cut_runs_where_the_l_begins():
    """On every line the "L" starts at the cut, with its stem or a blank before its top and foot."""
    assert {line[LLAMA_COLUMN] for line in BANNER.splitlines()} == {' ', '|'}


def test_a_narrower_terminal_gets_a_plain_title_line():
    title = banner_text(NARROW)
    assert '\n' not in title and '$' not in title
    assert 'SocketLlama' in title and __version__ in title
    assert len(title) < NARROW


def test_the_banner_goes_to_a_terminal(monkeypatch):
    monkeypatch.setattr(console, 'terminal_columns', lambda: BANNER_WIDTH)
    stream = FakeTerminal()
    show_banner(stream)
    assert BANNER in stream.getvalue()


def test_a_narrow_terminal_shows_the_title_instead(monkeypatch):
    monkeypatch.setattr(console, 'terminal_columns', lambda: NARROW)
    stream = FakeTerminal()
    show_banner(stream)
    assert BANNER not in stream.getvalue() and 'SocketLlama' in stream.getvalue()


def test_nothing_goes_to_a_pipe_or_a_file(monkeypatch):
    monkeypatch.setattr(console, 'terminal_columns', lambda: BANNER_WIDTH)
    stream = io.StringIO()
    show_banner(stream)
    assert stream.getvalue() == ''


def test_the_width_of_the_terminal_is_what_the_environment_says(monkeypatch):
    monkeypatch.setenv('COLUMNS', str(BANNER_WIDTH))
    assert console.terminal_columns() == BANNER_WIDTH


def test_the_terminal_shows_text_and_reads_answers(monkeypatch, capsys):
    monkeypatch.setattr('builtins.input', lambda prompt: f'{ANSWER} to {prompt}')
    monkeypatch.setattr(console.getpass, 'getpass', lambda prompt: f'secret {ANSWER} to {prompt}')
    terminal = Terminal()
    terminal.say('a line')
    terminal.say()
    assert terminal.ask('Name: ') == f'{ANSWER} to Name: '
    assert terminal.ask_secret('Token: ') == f'secret {ANSWER} to Token: '
    assert capsys.readouterr().out == 'a line\n\n'
