"""The TOML written by setup reads back, through tomllib, as exactly the values that went in."""
import math
import tomllib

import pytest

from socketllama.toml_writer import toml_assignment, toml_inline_table, toml_key, toml_string, toml_value

HOSTILE_STRINGS = [
    '',
    'plain',
    'with "double" and \'single\' quotes',
    'back\\slash and \\" escaped quote',
    'ends with a backslash\\',
    'line\nbreak, carriage\rreturn, tab\there',
    '"""three quotes"""',
    "'''three single quotes'''",
    'umlauts äöü, 日本語, emoji 🦙',
    'control \x00 \x01 \x08 \x0b \x0c \x1f \x7f characters',
    'next line \x85, line separator  , paragraph separator  , byte order mark ﻿',
    'right-to-left override ‮ and zero width joiner ‍',
    'beyond the basic plane, unassigned: \U000e0001',
    '# not a comment',
    'key = "value"\n[table]',
    '\\u0041 stays six characters',
    '  leading and trailing spaces  ',
]
LONE_SURROGATE = '\ud800'


def read_back(value):
    return tomllib.loads(toml_assignment('key', value))['key']


@pytest.mark.parametrize('text', HOSTILE_STRINGS)
def test_a_string_reads_back_unchanged(text):
    assert read_back(text) == text


@pytest.mark.parametrize('text', HOSTILE_STRINGS)
def test_a_string_is_written_on_one_line_of_printable_characters(text):
    written = toml_string(text)
    assert written.isprintable()
    assert written.startswith('"') and written.endswith('"')


def test_every_single_character_of_the_basic_plane_reads_back():
    surrogates = range(0xD800, 0xE000)
    for code in range(0x10000):
        if code not in surrogates:
            assert read_back(f'a{chr(code)}b') == f'a{chr(code)}b', hex(code)


def test_text_that_is_not_unicode_is_refused():
    """What a terminal hands over for bytes that are not UTF-8: TOML cannot hold it."""
    with pytest.raises(ValueError, match='Unicode'):
        toml_string(f'token{LONE_SURROGATE}')


@pytest.mark.parametrize('value, written', [
    (True, 'true'),
    (False, 'false'),
    (0, '0'),
    (-7, '-7'),
    (16384, '16384'),
    (30.0, '30.0'),
    (0.5, '0.5'),
    ([], '[]'),
    (['llama3.1:8b', 'qwen3:14b'], '["llama3.1:8b", "qwen3:14b"]'),
])
def test_values_are_written_as_toml(value, written):
    assert toml_value(value) == written
    assert read_back(value) == value
    assert type(read_back(value)) is type(value)


@pytest.mark.parametrize('value', [1e-07, 1e+16, 1.5e300, -0.25, math.inf, -math.inf])
def test_unusual_floats_read_back(value):
    assert read_back(value) == value


def test_not_a_number_reads_back():
    assert math.isnan(read_back(math.nan))


def test_a_list_of_hostile_strings_reads_back():
    assert read_back(HOSTILE_STRINGS) == HOSTILE_STRINGS


def test_a_tuple_is_written_as_a_list():
    assert read_back(('a', 'b')) == ['a', 'b']


@pytest.mark.parametrize('value', [None, {'a': 1}, b'bytes', object()])
def test_other_types_are_refused(value):
    with pytest.raises(TypeError):
        toml_value(value)


@pytest.mark.parametrize('key', ['url', 'signing_secret', 'base-url', 'x1'])
def test_a_plain_key_is_written_bare(key):
    assert toml_key(key) == key


@pytest.mark.parametrize('key', ['', 'with space', 'dotted.key', 'quote"d', 'ümlaut', 'new\nline'])
def test_any_other_key_is_quoted_and_reads_back(key):
    assert tomllib.loads(f'{toml_key(key)} = 1') == {key: 1}


def test_an_inline_table_reads_back_as_its_values():
    table = {'input': 0.1 + 0.2, 'output': 2.0}
    assert toml_inline_table(table) == '{ input = 0.30000000000000004, output = 2.0 }'
    assert tomllib.loads(f'price = {toml_inline_table(table)}') == {'price': table}


def test_an_empty_inline_table_is_written_as_one():
    assert tomllib.loads(f'price = {toml_inline_table({})}') == {'price': {}}
