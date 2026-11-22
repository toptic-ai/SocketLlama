"""The prices and descriptions of config.toml: the client holds them, reports them, and writes the ones
toptic.ai sets.

Each is a table below a backend's [[backends]] entry with one line per model (a ModelTable): the prices in
USD per 1M tokens, and a description of what a model is good for, as toptic.ai shows it.

    [[backends]]
    name = "ollama"
    models = ["llama3.1:8b", "qwen3:14b"]

    # Prices in USD per 1M tokens. toptic.ai keeps these lines up to date.
    [backends.prices_per_1m_tokens]
    "llama3.1:8b" = { input = 0.8, output = 2.0 }

    # What each model is good for, as toptic.ai shows it. toptic.ai keeps these lines up to date.
    [backends.descriptions]
    "llama3.1:8b" = "Quick answers to short questions"

The file is read once, when the client starts. What its tables hold goes with every models report to a
toptic.ai that takes it (capabilities models.prices and models.descriptions), and toptic.ai decides what is in
force: the edit that reached it last. When it sends a value (an entry of a models.settings frame), the client
writes it into config.toml in place: only that model's line in that table changes, or is added (with the
table, if the backend has none). Every other byte of the file stays as it was, its comments and its layout
included.

The standard library reads TOML (tomllib) but does not write it, and a library that keeps a file's
layout would be a new dependency for one line. So the editor works on the lines of the file and checks
its own work before anything is written: the new text must read back as the data of the old one with
exactly this one value changed. A layout it does not follow (dotted keys, a table per model, an inline
table on the backend, a string over several lines) fails that check, and the file is left as it is. Both
tables go through the same editor; what differs between them is a ModelTable.
"""
import copy
import logging
import os
import re
import stat
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path

from . import events, protocol
from .config import KEY_DESCRIPTIONS, KEY_PRICE_INPUT, KEY_PRICE_OUTPUT, KEY_PRICES, Backend, Config, ConfigError, \
    ModelPrice, parse_description, parse_price
from .config_writer import FILE_ENCODING, KEY_MODELS, KEY_NAME, SECTION_BACKENDS, BackupFailed, inspect_target, \
    write_config_file
from .text import printable
from .toml_writer import KEY_VALUE_SEPARATOR, toml_inline_table, toml_key, toml_string

logger = logging.getLogger('socketllama')

LINE_FEED = '\n'
CARRIAGE_RETURN = '\r'
CRLF = CARRIAGE_RETURN + LINE_FEED
COMMENT_MARK = '#'
INLINE_TABLE_START = '{'
INLINE_TABLE_END = '}'
BASIC_STRING_QUOTE = '"'
LITERAL_STRING_QUOTE = "'"
STRING_QUOTES = (BASIC_STRING_QUOTE, LITERAL_STRING_QUOTE)
ESCAPE = '\\'
# What a dotted name of a table header is joined with
NAME_SEPARATOR = '.'
TABLE_BRACKETS = ('[', ']')
ARRAY_OF_TABLES_BRACKETS = ('[[', ']]')
# A table header, [name] or [[name]], with an optional comment after it
TABLE_HEADER = re.compile(r'\s*(?P<open>\[\[?)\s*(?P<name>[^\[\]#]+?)\s*(?P<close>\]\]?)\s*(?:#.*)?')
DOTTED_NAME_SEPARATOR = re.compile(r'\s*\.\s*')
# The key a line begins with and the "=" after it: a bare key, a "basic" or a 'literal' string
KEY_AT_LINE_START = re.compile(r'(?P<indent>[ \t]*)(?P<key>"(?:[^"\\]|\\.)*"|\'[^\']*\'|[A-Za-z0-9_-]+)[ \t]*=[ \t]*')
# What may follow a value on its line
REST_OF_LINE = re.compile(r'\s*(?:#.*)?')
# A key is read by letting tomllib read it, so that it is read exactly as TOML reads it
KEY_PROBE = '{key} = 0'

ENTRY_HEADER = (ARRAY_OF_TABLES_BRACKETS, SECTION_BACKENDS)


class LineEditRefused(Exception):
    """A value cannot be written into the text of a configuration file; the message says why."""


# ----- the tables -----

class ModelTable(ABC):
    """A table below a [[backends]] entry with one line per model, its key the model's name, that toptic.ai keeps
    up to date: where it is, how its values are read, written and carried, and how the owner is told of them."""
    # The table's key in its backend (config.KEY_*): its header is [backends.<key>]
    key: str
    # What one value is, and several: "price", "prices"
    noun: str
    plural: str
    # Written above the header of a table the client adds
    comment: str
    # How a value is written on its line, for a line the editor cannot change
    written_as: str
    # Layouts of the table the editor does not follow, for the message about one
    not_followed: str
    # The capability under which toptic.ai takes and sends the values, and the fields that carry them: in a
    # model of a models report, and in an entry of a models.settings frame
    capability: str
    report_field: str
    settings_field: str

    @property
    def header(self) -> str:
        """The dotted name of the table's header: backends.prices_per_1m_tokens."""
        return f'{SECTION_BACKENDS}{NAME_SEPARATOR}{self.key}'

    @property
    def header_line(self) -> str:
        return f'{TABLE_BRACKETS[0]}{self.header}{TABLE_BRACKETS[1]}'

    def line(self, model: str, value: object) -> str:
        """The line of ``model`` in the table: ``"llama3.1:8b" = { input = 0.8, output = 2.0 }``."""
        return f'{toml_key(model)}{KEY_VALUE_SEPARATOR}{self.toml(value)}'

    def same(self, value: object, other: object | None) -> bool:
        """Whether ``other`` (None: there is none) is ``value``, as toptic.ai compares them."""
        return value == other

    @abstractmethod
    def held(self, backend: Backend) -> Mapping[str, object]:
        """{model name: value} of the backend's table, as the file was loaded."""

    @abstractmethod
    def parse(self, value: object, where: str) -> object:
        """A value as config.toml holds it; ConfigError (naming ``where``) unless the file could load with it."""

    @abstractmethod
    def read_wire(self, value: object, where: str) -> object:
        """A value an entry of a models.settings frame carries; ConfigError (naming ``where``) unless config.toml
        could hold it."""

    @abstractmethod
    def wire(self, value: object) -> object:
        """A value as a models report carries it."""

    @abstractmethod
    def data(self, value: object) -> object:
        """A value as tomllib reads it from its line."""

    @abstractmethod
    def toml(self, value: object) -> str:
        """A value as TOML, on one line."""

    @abstractmethod
    def value_end(self, content: str, start: int) -> int | None:
        """Where the value written at ``content[start]`` ends; None unless a value of this table begins there and
        ends on this line."""

    @abstractmethod
    def in_words(self, value: object) -> str:
        """A value as a line of the terminal shows it."""


class PriceTable(ModelTable):
    """[backends.prices_per_1m_tokens]: what a model costs, ``{ input = 0.8, output = 2.0 }``."""
    key = KEY_PRICES
    noun = 'price'
    plural = 'prices'
    comment = f'{COMMENT_MARK} Prices in USD per 1M tokens. toptic.ai keeps these lines up to date.'
    written_as = f'{{ {KEY_PRICE_INPUT} = ..., {KEY_PRICE_OUTPUT} = ... }}'
    not_followed = 'dotted keys, a table per model, or the prices as an inline table on the backend'
    capability = protocol.CAPABILITY_MODELS_PRICES
    report_field = protocol.CONFIG_PRICE_FIELD
    settings_field = protocol.SETTINGS_PRICE_FIELD

    def same(self, value: ModelPrice, other: ModelPrice | None) -> bool:
        return value.same_as(other)

    def held(self, backend: Backend) -> Mapping[str, ModelPrice]:
        return backend.prices

    def parse(self, value: object, where: str) -> ModelPrice:
        return parse_price(value, where)

    def read_wire(self, value: object, where: str) -> ModelPrice:
        return read_wire_price(value, where)

    def wire(self, value: ModelPrice) -> dict:
        return {protocol.PRICE_INPUT_FIELD: value.input, protocol.PRICE_OUTPUT_FIELD: value.output}

    def data(self, value: ModelPrice) -> dict:
        return value.as_table()

    def toml(self, value: ModelPrice) -> str:
        return toml_inline_table(value.as_table())

    def value_end(self, content: str, start: int) -> int | None:
        return _inline_table_end(content, start)

    def in_words(self, value: ModelPrice) -> str:
        return events.price_in_words(value)


class DescriptionTable(ModelTable):
    """[backends.descriptions]: what a model is good for, one line of text, ``"Quick answers"``."""
    key = KEY_DESCRIPTIONS
    noun = 'description'
    plural = 'descriptions'
    comment = (f'{COMMENT_MARK} What each model is good for, as toptic.ai shows it. toptic.ai keeps these lines up '
               f'to date.')
    written_as = '"<text>", a string on one line'
    not_followed = 'a string over several lines, dotted keys, or the descriptions as an inline table on the backend'
    capability = protocol.CAPABILITY_MODELS_DESCRIPTIONS
    report_field = protocol.CONFIG_DESCRIPTION_FIELD
    settings_field = protocol.SETTINGS_DESCRIPTION_FIELD

    def held(self, backend: Backend) -> Mapping[str, str]:
        return backend.descriptions

    def parse(self, value: object, where: str) -> str:
        return parse_description(value, where)

    def read_wire(self, value: object, where: str) -> str:
        # The wire carries a description as config.toml holds it, and toptic.ai is held to the same limits
        return parse_description(value, where)

    def wire(self, value: str) -> str:
        return value

    def data(self, value: str) -> str:
        return value

    def toml(self, value: str) -> str:
        return toml_string(value)

    def value_end(self, content: str, start: int) -> int | None:
        return _string_end(content, start)

    def in_words(self, value: str) -> str:
        return events.description_in_words(value)


PRICES = PriceTable()
DESCRIPTIONS = DescriptionTable()
MODEL_TABLES = (PRICES, DESCRIPTIONS)


def read_wire_price(value: object, where: str) -> ModelPrice:
    """A price a models.settings frame carries; ConfigError (naming ``where``) unless config.toml could hold it.
    Fields besides the two numbers are left out: a toptic.ai that sends more still has its prices taken."""
    if not isinstance(value, dict):
        return parse_price(value, where)
    fields = {protocol.PRICE_INPUT_FIELD: KEY_PRICE_INPUT, protocol.PRICE_OUTPUT_FIELD: KEY_PRICE_OUTPUT}
    return parse_price({key: value[field] for field, key in fields.items() if field in value}, where)


def price_line(model: str, price: ModelPrice) -> str:
    """The line of a model's price in its backend's table: ``"llama3.1:8b" = { input = 0.8, output = 2.0 }``."""
    return PRICES.line(model, price)


def description_line(model: str, description: str) -> str:
    """The line of a model's description in its backend's table: ``"llama3.1:8b" = "Quick answers"``."""
    return DESCRIPTIONS.line(model, description)


def edit_price(text: str, backend: str, model: str, price: ModelPrice) -> str:
    """``text`` with the price of ``model`` of ``backend`` set to ``price``; see edit_line."""
    return edit_line(text, PRICES, backend, model, price)


def edit_description(text: str, backend: str, model: str, description: str) -> str:
    """``text`` with the description of ``model`` of ``backend`` set to ``description``; see edit_line."""
    return edit_line(text, DESCRIPTIONS, backend, model, description)


# ----- the editor -----

def edit_line(text: str, table: ModelTable, backend: str, model: str, value: object) -> str:
    """``text`` of a configuration file with the line of ``model`` in ``table`` of ``backend`` set to ``value``,
    every other byte unchanged. LineEditRefused if that cannot be done, or not verified to have been done."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as ex:
        raise LineEditRefused(f'it is not valid TOML at the moment ({ex})') from ex
    entries = data.get(SECTION_BACKENDS)
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise LineEditRefused(f'its backends are not [[{SECTION_BACKENDS}]] entries')
    positions = [position for position, entry in enumerate(entries) if entry.get(KEY_NAME) == backend]
    if len(positions) != 1:
        raise LineEditRefused(f'it has no backend {backend!r}' if not positions else
                              f'it has more than one backend {backend!r}')
    index = positions[0]
    shared = entries[index].get(KEY_MODELS)
    if isinstance(shared, list) and shared and model not in shared:
        raise LineEditRefused(f'backend {backend} does not share {model!r} (not in its models): the file would not '
                              f'load with a {table.noun} for it')
    values = entries[index].get(table.key)
    if values is not None and not isinstance(values, dict):
        raise LineEditRefused(_not_followed(table))

    lines = _split_lines(text)
    headers = [(number, header) for number, line in enumerate(lines) if (header := _header(line)) is not None]
    entry_starts = [number for number, header in headers if header == ENTRY_HEADER]
    if len(entry_starts) != len(entries):
        raise LineEditRefused(_not_followed(table))
    start = entry_starts[index]
    end = next((number for number, header in headers if number > start and not _of_an_entry(header)), len(lines))
    table_header = (TABLE_BRACKETS, table.header)
    tables = [number for number, header in headers if start < number < end and header == table_header]
    newline = CRLF if lines and lines[0].endswith(CRLF) else LINE_FEED
    if tables:
        table_end = next((number for number, _ in headers if number > tables[0]), len(lines))
        _set_in_table(lines, table, tables[0], table_end, model, value, newline)
    elif values is not None:
        raise LineEditRefused(_not_followed(table))
    else:
        _add_table(lines, table, start, end, model, value, newline)
    edited = ''.join(lines)
    _verify(edited, data, index, table, model, value)
    return edited


def _not_followed(table: ModelTable) -> str:
    return (f'it is written in a way this client cannot change safely (for example {table.not_followed}): write '
            f'the {table.noun} as a line of a [{table.header}] table')


def _split_lines(text: str) -> list[str]:
    """The lines of ``text`` with their endings. Only a line feed ends a line, as in TOML: splitlines would
    also split at characters a string or a comment may hold."""
    parts = text.split(LINE_FEED)
    return [part + LINE_FEED for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _content(line: str) -> str:
    return line.removesuffix(LINE_FEED).removesuffix(CARRIAGE_RETURN)


def _ending(line: str) -> str:
    return line[len(_content(line)):]


def _header(line: str) -> tuple | None:
    """((open, close) brackets, dotted name) of a table header line; None for any other line."""
    match = TABLE_HEADER.fullmatch(_content(line))
    if match is None:
        return None
    return (match['open'], match['close']), DOTTED_NAME_SEPARATOR.sub(NAME_SEPARATOR, match['name'])


def _of_an_entry(header: tuple) -> bool:
    """Whether a header after a [[backends]] header still belongs to that entry (a table below it)."""
    return header != ENTRY_HEADER and header[1].startswith(f'{SECTION_BACKENDS}{NAME_SEPARATOR}')


def _is_statement(line: str) -> bool:
    """A line that is neither empty nor only a comment."""
    content = _content(line).strip()
    return bool(content) and not content.startswith(COMMENT_MARK)


def _key(written: str) -> str | None:
    """The key as TOML reads what is written; None if it cannot be read."""
    try:
        return next(iter(tomllib.loads(KEY_PROBE.format(key=written))))
    except tomllib.TOMLDecodeError:
        return None


def _set_in_table(lines: list[str], table: ModelTable, header: int, table_end: int, model: str, value: object,
                  newline: str) -> None:
    """Replace the line of ``model`` in ``table``, whose header is at ``header``, or add one at the table's end."""
    key_lines = []
    for number in range(header + 1, table_end):
        match = KEY_AT_LINE_START.match(_content(lines[number]))
        if match is None:
            continue
        key_lines.append(match)
        if _key(match['key']) == model:
            content = _content(lines[number])
            value_end = table.value_end(content, match.end())
            if value_end is None or not REST_OF_LINE.fullmatch(content[value_end:]):
                raise LineEditRefused(f'the {table.noun} of {model!r} is not written on one line as '
                                      f'{table.written_as}')
            # Indentation, key and "=" as they were, and whatever follows the value (a comment)
            lines[number] = content[:match.end()] + table.toml(value) + content[value_end:] + _ending(lines[number])
            return
    indent = key_lines[-1]['indent'] if key_lines else ''
    after = max((number for number in range(header + 1, table_end) if _is_statement(lines[number])), default=header)
    _insert(lines, after + 1, [indent + table.line(model, value)], newline)


def _add_table(lines: list[str], table: ModelTable, start: int, end: int, model: str, value: object,
               newline: str) -> None:
    """Add ``table`` with the line of ``model`` at the end of the [[backends]] entry from ``start`` to ``end``:
    after its last value (the last line of its other table of models, if it has one) and the comments directly
    below it. Comments after an empty line are taken to be about what comes next (a commented-out backend), and
    stay below."""
    after = max(number for number in range(start, end) if _is_statement(lines[number])) + 1
    while after < end and _content(lines[after]).strip():
        after += 1
    block = ['', table.comment, table.header_line, table.line(model, value)]
    if after < len(lines) and _content(lines[after]).strip():
        block.append('')  # keeps the next header apart
    _insert(lines, after, block, newline)


def _insert(lines: list[str], position: int, new_lines: list[str], newline: str) -> None:
    if position > 0 and not _ending(lines[position - 1]):
        lines[position - 1] += newline  # the file ended without one
    lines[position:position] = [line + newline for line in new_lines]


def _string_end(content: str, start: int) -> int | None:
    """Where the "basic" or 'literal' string that opens at ``content[start]`` ends (just after its closing
    quote); None if no string opens there or it does not close on this line. A string over several lines
    (three quotes) reads here as an empty string with more after it, so its line is not taken for one value."""
    if start >= len(content) or content[start] not in STRING_QUOTES:
        return None
    quote = content[start]
    position = start + 1
    while position < len(content):
        character = content[position]
        if quote == BASIC_STRING_QUOTE and character == ESCAPE:
            position += len(ESCAPE) + 1  # the escape and the character it escapes
            continue
        if character == quote:
            return position + 1
        position += 1
    return None


def _inline_table_end(content: str, start: int) -> int | None:
    """Where the inline table that opens at ``content[start]`` ends (just after its closing brace); None if no
    inline table opens there or it does not close on this line."""
    if not content.startswith(INLINE_TABLE_START, start):
        return None
    depth = 0
    position = start
    while position < len(content):
        character = content[position]
        if character in STRING_QUOTES:
            string_end = _string_end(content, position)
            if string_end is None:
                return None
            position = string_end
            continue
        if character == INLINE_TABLE_START:
            depth += 1
        elif character == INLINE_TABLE_END:
            depth -= 1
            if depth == 0:
                return position + 1
        elif character == COMMENT_MARK:
            return None
        position += 1
    return None


def _verify(edited: str, data: dict, index: int, table: ModelTable, model: str, value: object) -> None:
    """The edited text must read as ``data`` with exactly this one value of ``table`` changed."""
    expected = copy.deepcopy(data)
    expected[SECTION_BACKENDS][index].setdefault(table.key, {})[model] = table.data(value)
    try:
        if tomllib.loads(edited) == expected:
            return
    except tomllib.TOMLDecodeError:
        pass
    raise LineEditRefused(_not_followed(table))


# ----- what the client holds, and writing what toptic.ai sets -----

class ConfigSettings:
    """The prices and descriptions of the configuration file as the client holds them: as the file was loaded,
    and as the client wrote them since. Reports are made from these. ``take_settings`` writes what toptic.ai
    sets."""

    def __init__(self, config: Config) -> None:
        self._path = config.path
        self._backends: Mapping[str, Backend] = config.backends
        # {table: {(backend, model): value}}. Replaced, never changed in place: a report may read it while
        # take_settings runs in a thread
        self._held: dict[ModelTable, dict[tuple[str, str], object]] = {
            table: {(backend.name, model): value
                    for backend in config.backends.values() for model, value in table.held(backend).items()}
            for table in MODEL_TABLES}
        # The file was copied to a backup before the first write of this run, of a price or of a description
        self._backed_up = False
        # What was warned about, so that a value offered again at every report is warned about once
        self._warned: set = set()

    def held(self, table: ModelTable) -> dict[tuple[str, str], object]:
        """{(backend, model): value} of every line of ``table`` the file holds."""
        return dict(self._held[table])

    @property
    def prices(self) -> dict[tuple[str, str], ModelPrice]:
        return self.held(PRICES)

    @property
    def descriptions(self) -> dict[tuple[str, str], str]:
        return self.held(DESCRIPTIONS)

    def report_fields(self, tables: Iterable[ModelTable]) -> dict[tuple[str, str], dict]:
        """{(backend, model): {field: value}}: what a model of a models report carries of the file, for the
        ``tables`` the gateway takes (a model without a line in one carries no field of it)."""
        held = self._held
        fields: dict[tuple[str, str], dict] = {}
        for table in tables:
            for key, value in held[table].items():
                fields.setdefault(key, {})[table.report_field] = table.wire(value)
        return fields

    def take_settings(self, frame: Mapping, acknowledged: Mapping[ModelTable, frozenset[tuple[str, str]]]) -> bool:
        """Write the values of a models.settings frame that are not those held into the file; whether the values
        held changed. Reads and writes the file: to be run in a thread, one frame at a time.

        ``acknowledged``: for each table the gateway takes, the (backend, model) pairs an acknowledged models
        report of this run listed with what the file holds of that table. toptic.ai has seen those lines only; a
        value for another model (its backend was down or could not be checked, the model was not offered, the
        report did not carry the table) is not taken: it could write over a line changed while the client was
        not running. The answer to a report that lists it brings the value again. A value of a table the gateway
        does not take is ignored.

        Each value of an entry is taken or refused for itself: a price that is none does not stop the description
        beside it."""
        changed = {(table, key): value for (table, key), value in self._settings(frame, acknowledged).items()
                   if not table.same(value, self._held[table].get(key))}
        if not changed:
            return False
        if self._path is None:
            self._not_written(changed, 'the client was not started with a configuration file', None)
            return False
        try:
            target = inspect_target(self._path)
            file = target.file
            with open(file, encoding=FILE_ENCODING, newline='') as handle:
                text = handle.read()
            status = file.stat()
        except (OSError, UnicodeDecodeError) as ex:
            self._not_written(changed, str(ex), self._path)
            return False
        reason = _unwritable(file, status)
        if reason is not None:
            self._not_written(changed, reason, file)
            return False
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as ex:
            self._not_written(changed, f'it is not valid TOML at the moment ({ex})', file)
            return False
        written, shown_already = {}, {}
        for (table, key), value in changed.items():
            has_line, in_file = _in_file(data, table, key)
            if in_file is not None and table.same(value, in_file):
                shown_already[(table, key)] = value  # changed by hand to just this value: nothing to write
                continue
            if has_line and not (in_file is not None and table.same(in_file, self._held[table].get(key))):
                # Changed by hand since the start: the owner's to report (at the next start), not to be written
                # over. A line that is gone is no value to keep: it is put back.
                self._warn_once(('changed', table.key, key, value),
                                events.setting_changed_meanwhile(table.noun, *key, table.in_words(value), file))
                continue
            try:
                text = edit_line(text, table, *key, value)
            except LineEditRefused as ex:
                self._not_written({(table, key): value}, str(ex), file)
                continue
            written[(table, key)] = value
        if written and not self._write(file, text, status, written):
            written = {}
        for (table, key), value in shown_already.items():
            logger.info(events.setting_shown_already(table.noun, *key, table.in_words(value), file))
        taken = {**shown_already, **written}
        self._held = {table: {**held, **{key: value for (of, key), value in taken.items() if of is table}}
                      for table, held in self._held.items()}
        return bool(taken)

    def _write(self, file: Path, text: str, status: os.stat_result, values: Mapping) -> bool:
        """Replace ``file`` by ``text`` with the mode and the group it has (``status``); whether it was."""
        try:
            backup = write_config_file(file, text, keep_backup=not self._backed_up, mode=stat.S_IMODE(status.st_mode),
                                       group=status.st_gid)
        except BackupFailed as ex:
            self._not_written(values, f'it could not be copied to a backup first: {ex}', file)
            return False
        except OSError as ex:
            self._not_written(values, str(ex), file)
            return False
        self._backed_up = True
        if backup is not None:
            logger.info(events.backup_kept(file, backup))
        for (table, key), value in values.items():
            logger.info(events.setting_written(table.noun, *key, table.in_words(value), file))
        return True

    def _settings(self, frame: Mapping, acknowledged: Mapping[ModelTable, frozenset[tuple[str, str]]]
                  ) -> dict[tuple[ModelTable, tuple[str, str]], object]:
        """{(table, (backend, model)): value} of a models.settings frame, for the tables and models of
        ``acknowledged`` (see take_settings); what this client does not take is warned about."""
        if not acknowledged:
            return {}
        entries = frame.get(protocol.SETTINGS_MODELS_FIELD)
        if not isinstance(entries, list):
            self._refused_entry(f'the frame has no list "{protocol.SETTINGS_MODELS_FIELD}"')
            return {}
        if len(entries) > protocol.MAX_SETTINGS_PER_FRAME:
            self._refused_entry(f'{len(entries)} models in one frame, more than the {protocol.MAX_SETTINGS_PER_FRAME} '
                                f'a machine can have: those after them are left out')
        settings = {}
        for entry in entries[:protocol.MAX_SETTINGS_PER_FRAME]:
            if not isinstance(entry, dict):
                self._refused_entry('an entry that is not an object')
                continue
            name, model = entry.get(protocol.SETTINGS_BACKEND_FIELD), entry.get(protocol.SETTINGS_MODEL_FIELD)
            backend = self._backends.get(name) if isinstance(name, str) else None
            if backend is None:
                self._refused_entry(f'the configuration has no backend {printable(str(name))!r}')
                continue
            if not isinstance(model, str) or not model or not backend.allows_model(model):
                self._refused_entry(f'backend {name} does not share {printable(str(model))!r}: no line is written '
                                    f'for it')
                continue
            carried = [table for table in acknowledged if table.settings_field in entry]
            if not carried:
                self._refused_entry(f'the entry of {name}/{printable(model)} carries no '
                                    f'{" or ".join(table.noun for table in acknowledged)}')
                continue
            for table in carried:
                if (name, model) not in acknowledged[table]:
                    logger.debug('Not taking the %s of %s/%s yet: no acknowledged report of this run listed it with '
                                 'its %s', table.noun, name, printable(model), table.noun)
                    continue
                try:
                    settings[(table, (name, model))] = table.read_wire(entry[table.settings_field],
                                                                       f'the {table.noun} of {name}/{printable(model)}')
                except ConfigError as ex:
                    self._refused(table, str(ex))
        return settings

    def _not_written(self, values: Mapping[tuple[ModelTable, tuple[str, str]], object], reason: str,
                     path: Path | None) -> None:
        for (table, (backend, model)), value in values.items():
            self._warn_once(('not_written', table.key, (backend, model), value),
                            events.setting_not_written(table.noun, table.header, backend, model,
                                                       table.in_words(value), path, reason, table.line(model, value)))

    def _refused(self, table: ModelTable, reason: str) -> None:
        self._warn_once(('refused', table.key, reason), events.setting_refused(table.noun, reason))

    def _refused_entry(self, reason: str) -> None:
        self._warn_once(('refused', reason), events.settings_entry_refused(reason))

    def _warn_once(self, key: tuple, line: str) -> None:
        if key in self._warned:
            logger.debug(line)
            return
        self._warned.add(key)
        logger.warning(line)


def _in_file(data: dict, table: ModelTable, key: tuple[str, str]) -> tuple[bool, object | None]:
    """Whether the file (``data``) has a line in ``table`` for the model of ``key``, and the value it holds (None
    if it holds none the table takes)."""
    backend, model = key
    entries = data.get(SECTION_BACKENDS)
    matching = [entry for entry in entries if isinstance(entry, dict) and entry.get(KEY_NAME) == backend] \
        if isinstance(entries, list) else []
    values = matching[0].get(table.key) if len(matching) == 1 else None
    if not isinstance(values, dict) or model not in values:
        return False, None
    try:
        return True, table.parse(values[model], model)
    except ConfigError:
        return True, None


def _unwritable(file: Path, status: os.stat_result) -> str | None:
    """Why the client may not replace ``file``; None if it may. A file without write permission is left as it
    is, although the folder would let the client put a new one in its place; so is one of another user, which
    the new file would take from them."""
    if status.st_uid != os.geteuid():
        return 'the file belongs to another user'
    if not os.access(file, os.W_OK):
        return 'the file is read-only'
    if not os.access(file.parent, os.W_OK | os.X_OK):
        return f'the folder {file.parent} does not let the client create the new file in it'
    return None
