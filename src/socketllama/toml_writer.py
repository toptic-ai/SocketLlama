"""Writes the values a configuration file holds as TOML.

The standard library reads TOML (``tomllib``) but does not write it. This covers exactly the
value types of the configuration: strings, booleans, whole numbers, floats and lists of those, and
inline tables of them (a price: ``{ input = 0.8, output = 2.0 }``).
"""
import math
import re

BARE_KEY = re.compile(r'[A-Za-z0-9_-]+')
STRING_DELIMITER = '"'
# The escapes TOML names; every other character that cannot be shown is written as its code point
SHORT_ESCAPES = {
    '\\': '\\\\',
    '"': '\\"',
    '\b': '\\b',
    '\t': '\\t',
    '\n': '\\n',
    '\f': '\\f',
    '\r': '\\r',
}
# Code points up to this one are escaped with four hexadecimal digits (\uXXXX), higher ones with eight
LAST_BASIC_PLANE_CODE_POINT = 0xFFFF
SHORT_CODE_POINT_ESCAPE = '\\u{:04X}'
LONG_CODE_POINT_ESCAPE = '\\U{:08X}'
# Reserved for UTF-16 pairs: not characters, and not allowed in a TOML file
FIRST_SURROGATE = 0xD800
LAST_SURROGATE = 0xDFFF
TRUE = 'true'
FALSE = 'false'
INFINITY = 'inf'
NOT_A_NUMBER = 'nan'
LIST_ITEM_SEPARATOR = ', '
KEY_VALUE_SEPARATOR = ' = '
INLINE_TABLE_OPEN = '{ '
INLINE_TABLE_CLOSE = ' }'
EMPTY_INLINE_TABLE = '{}'


def toml_assignment(key: str, value: object) -> str:
    """The line ``key = value``."""
    return f'{toml_key(key)}{KEY_VALUE_SEPARATOR}{toml_value(value)}'


def toml_inline_table(table: dict) -> str:
    """``table`` on one line, ``{ key = value, other = value }``."""
    if not table:
        return EMPTY_INLINE_TABLE
    items = LIST_ITEM_SEPARATOR.join(toml_assignment(key, value) for key, value in table.items())
    return f'{INLINE_TABLE_OPEN}{items}{INLINE_TABLE_CLOSE}'


def toml_key(key: str) -> str:
    return key if BARE_KEY.fullmatch(key) else toml_string(key)


def toml_value(value: object) -> str:
    if isinstance(value, str):
        return toml_string(value)
    if isinstance(value, bool):  # before int: a bool is one
        return TRUE if value else FALSE
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _toml_float(value)
    if isinstance(value, (list, tuple)):
        return f'[{LIST_ITEM_SEPARATOR.join(toml_value(item) for item in value)}]'
    raise TypeError(f'A configuration file cannot hold a value of type {type(value).__name__}')


def toml_string(text: str) -> str:
    """``text`` as a basic string on one line: whatever cannot be shown as itself is escaped."""
    return f'{STRING_DELIMITER}{"".join(_escaped(character) for character in text)}{STRING_DELIMITER}'


def _escaped(character: str) -> str:
    if character in SHORT_ESCAPES:
        return SHORT_ESCAPES[character]
    if character.isprintable():
        return character
    code_point = ord(character)
    if FIRST_SURROGATE <= code_point <= LAST_SURROGATE:
        raise ValueError(f'The text holds something that is not Unicode (code {code_point:#x}) and cannot be written')
    escape = SHORT_CODE_POINT_ESCAPE if code_point <= LAST_BASIC_PLANE_CODE_POINT else LONG_CODE_POINT_ESCAPE
    return escape.format(code_point)


def _toml_float(value: float) -> str:
    if math.isnan(value):
        return NOT_A_NUMBER
    if math.isinf(value):
        return INFINITY if value > 0 else f'-{INFINITY}'
    return repr(value)
