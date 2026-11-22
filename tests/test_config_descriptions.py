"""The descriptions of config.toml: read with the configuration, written in place by the editor the prices use,
every other byte kept, and only what it verified."""
import logging
import tomllib
from pathlib import Path

import pytest

from socketllama import protocol
from socketllama.config import ConfigError, ModelPrice, load_config, parse_config
from socketllama.config_settings import DESCRIPTIONS, PRICES, ConfigSettings, LineEditRefused, \
    description_line, edit_description, edit_price

MODEL = 'llama3.1:8b'
OTHER_MODEL = 'qwen3:14b'
NOT_SHARED = 'mistral:7b'
PROXY_MODEL = 'gpt-proxy'
TOKEN_VARIABLE = 'DESCRIPTIONS_TEST_TOKEN'
SECRET_VARIABLE = 'DESCRIPTIONS_TEST_SECRET'
# MAX_MODEL_DESCRIPTION_CHARS of the gateway, and what it refuses in a description: by category, and the
# bidirectional controls U+200E, U+200F, U+061C, U+202A to U+202E, U+2066 to U+2069
GATEWAY_MAX_MODEL_DESCRIPTION_CHARS = 500
GATEWAY_REFUSED_CATEGORIES = {'Cc', 'Zl', 'Zp', 'Cs'}
GATEWAY_REFUSED_CHARACTERS = '\u200e\u200f\u061c\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069'
OLD_DESCRIPTION = 'Quick answers'
NEW_DESCRIPTION = 'Quick answers to short questions, in English and German'
LATER_DESCRIPTION = 'Quick answers; slow on long documents'
HAND_EDITED_DESCRIPTION = 'Written by hand meanwhile'
TOO_LONG = 'x' * (GATEWAY_MAX_MODEL_DESCRIPTION_CHARS + 1)
FILE_PRICE = ModelPrice(input=0.8, output=2.0)
NEW_PRICE = ModelPrice(input=0.4, output=1.4)
OLD_PRICE_VALUE = '{ input = 0.80, output = 2.00 }'
NEW_PRICE_VALUE = '{ input = 0.4, output = 1.4 }'
# A file as somebody writes it by hand: comments everywhere, indented lines, a comment after a description,
# and the price table before the descriptions
HAND_WRITTEN = f'''# My machine at home
[gateway]
url = "wss://toptic.example/ws/socketllama/"   # from the machine's page
instance = "tester-home"
token_env = "{TOKEN_VARIABLE}"
signing_secret_env = "{SECRET_VARIABLE}"

[[backends]]
name = "ollama"   # the big box
kind = "ollama"
base_url = "http://ollama.invalid/v1"
models = ["{MODEL}", "{OTHER_MODEL}"]

[backends.prices_per_1m_tokens]
"{MODEL}" = {OLD_PRICE_VALUE}

    # For the people at work
[backends.descriptions]
  "{MODEL}" = "{OLD_DESCRIPTION}"   # since November
  # "{OTHER_MODEL}" = "Careful reasoning"

# The proxy at work
[[backends]]
name = "litellm"
kind = "litellm"
base_url = "http://llm.invalid/v1"
models = []
'''
OLD_LINE_VALUE = f'"{OLD_DESCRIPTION}"'
NEW_LINE_VALUE = f'"{NEW_DESCRIPTION}"'
DESCRIPTION_TABLE = f'''
    # For the people at work
[backends.descriptions]
  "{MODEL}" = "{OLD_DESCRIPTION}"   # since November
  # "{OTHER_MODEL}" = "Careful reasoning"
'''
PRICE_TABLE = f'''
[backends.prices_per_1m_tokens]
"{MODEL}" = {OLD_PRICE_VALUE}
'''
EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / 'config.example.toml'
OWNER_ONLY = 0o600
READ_ONLY = 0o400
# Characters a description may hold that TOML has to escape, or that are not ASCII
AWKWARD_DESCRIPTIONS = [
    'Says "hello" and quotes \'you\'',
    'Reads C:\\models\\llama and \\n is not a line break',
    'Grüezi, ça va? 日本語も 🦙',
    'A no-break\u00a0space and a family 👩\u200d👩\u200d👧',
    'Zusammen\u00adfassung with a soft hyphen, and a zero\u200cwidth non-joiner',
    '# not a comment = not a key',
]


def environ() -> dict:
    return {TOKEN_VARIABLE: 'tt_live_x', SECRET_VARIABLE: 'secret'}


def parsed(text: str):
    return parse_config(tomllib.loads(text), environ=environ())


def with_descriptions(table: str, models: str = f'["{MODEL}"]') -> str:
    """A configuration whose ollama backend shares ``models`` and has ``table`` as its descriptions."""
    return HAND_WRITTEN.replace(DESCRIPTION_TABLE, f'\n[backends.descriptions]\n{table}\n').replace(
        f'models = ["{MODEL}", "{OTHER_MODEL}"]', f'models = {models}')


def expected_data(text: str, index: int, model: str, description: str) -> dict:
    data = tomllib.loads(text)
    data['backends'][index].setdefault('descriptions', {})[model] = description
    return data


# ----- the configuration -----

def test_the_limit_matches_the_gateway():
    assert protocol.MAX_MODEL_DESCRIPTION_CHARS == GATEWAY_MAX_MODEL_DESCRIPTION_CHARS


def test_a_backend_holds_the_descriptions_of_its_table_without_the_blanks_around_them():
    backend = parsed(with_descriptions(f'"{MODEL}" = "  {OLD_DESCRIPTION} "')).backends['ollama']
    assert backend.descriptions == {MODEL: OLD_DESCRIPTION}


def test_without_a_table_a_backend_has_no_descriptions():
    assert parsed(HAND_WRITTEN.replace(DESCRIPTION_TABLE, '')).backends['ollama'].descriptions == {}


def test_both_tables_are_read_in_either_order():
    descriptions_first = HAND_WRITTEN.replace(PRICE_TABLE, '').replace(DESCRIPTION_TABLE,
                                                                       DESCRIPTION_TABLE + PRICE_TABLE)
    assert descriptions_first.index('[backends.descriptions]') < descriptions_first.index('[backends.prices')
    for text in (HAND_WRITTEN, descriptions_first):
        backend = parsed(text).backends['ollama']
        assert backend.descriptions == {MODEL: OLD_DESCRIPTION} and backend.prices == {MODEL: FILE_PRICE}


def test_a_backend_that_shares_every_model_may_describe_any():
    text = HAND_WRITTEN.replace('models = []', 'models = []\n\n[backends.descriptions]\nanything = "Whatever"')
    assert parsed(text).backends['litellm'].descriptions == {'anything': 'Whatever'}


def test_a_description_of_the_longest_length_is_one():
    longest = 'x' * protocol.MAX_MODEL_DESCRIPTION_CHARS
    assert parsed(with_descriptions(f'"{MODEL}" = "{longest}"')).backends['ollama'].descriptions[MODEL] == longest


@pytest.mark.parametrize('table, message', [
    (f'"{MODEL}" = 7', 'must be "<one line of text'),
    (f'"{MODEL}" = ["Quick"]', 'must be "<one line of text'),
    (f'"{MODEL}" = ""', 'not empty; to have none, leave the line out'),
    (f'"{MODEL}" = "   "', 'not empty'),
    (f'"{MODEL}" = "{TOO_LONG}"', f'has {len(TOO_LONG)} characters; it can have at most 500'),
    (f'"{MODEL}" = "one\\ntwo"', r'holds U\+000A \(a control character\), which a description may not'),
    (f'"{MODEL}" = """one\ntwo"""', r'holds U\+000A'),
    (f'"{MODEL}" = "a\\ttab"', r'holds U\+0009 \(a control character\)'),
    (f'"{MODEL}" = "a line\\u2028separator"', r'holds U\+2028 \(LINE SEPARATOR\)'),
    (f'"{MODEL}" = "a bell\\u0007"', r'holds U\+0007'),
    (f'"{MODEL}" = "turned \\u202eround"', r'holds U\+202E \(RIGHT-TO-LEFT OVERRIDE\)'),
], ids=['number', 'list', 'empty', 'blank', 'too long', 'line feed', 'over two lines', 'tab', 'line separator',
        'control character', 'right-to-left override'])
def test_a_description_that_cannot_be_one_is_refused(table, message):
    with pytest.raises(ConfigError, match=message):
        parsed(with_descriptions(table))


def test_a_refused_character_is_named_with_the_rule_in_full():
    with pytest.raises(ConfigError) as refused:
        parsed(with_descriptions(f'"{MODEL}" = "a\\u2066b\\u2069"'))
    assert str(refused.value) == (
        "Backend ollama: the description of 'llama3.1:8b' holds U+2066 (LEFT-TO-RIGHT ISOLATE), which a description "
        'may not: it must be one line of 1 to 500 characters, without the blanks around it, and without control '
        'characters (a tab, a line break), separators of lines or paragraphs, or the controls of writing direction '
        '(U+200E, U+200F, U+061C, U+202A to U+202E, U+2066 to U+2069)')


# The same table pins the rule in the gateway's tests: change both together
ACCEPTED_BY_BOTH = [
    'Rapide\u00a0: bon',                       # a no-break space
    'Rapide\u202f: bon',                       # a narrow no-break space
    'Coding \U0001F9D1\u200d\U0001F4BB',      # an emoji joined by a zero-width joiner
    'Zusammen\u00adfassung',                   # a soft hyphen
    '高速\u3000モデル',                         # an ideographic space
    ' ' + 'a' * GATEWAY_MAX_MODEL_DESCRIPTION_CHARS + ' ',  # the longest, once the blanks around it are off
]
REFUSED_BY_BOTH = [
    'a\tb',
    'a\nb',
    'a\u2028b',                                # line separator
    'a\u2029b',                                # paragraph separator
    'abc\u202edef',                            # right-to-left override
    'a\u2066b\u2069',                          # left-to-right isolate, pop directional isolate
    'a\u200fb',                                # right-to-left mark
    '',
    '   ',
    'a' * (GATEWAY_MAX_MODEL_DESCRIPTION_CHARS + 1),
]


@pytest.mark.parametrize('description', ACCEPTED_BY_BOTH)
def test_what_the_gateway_accepts_the_configuration_and_the_wire_accept(description):
    assert parsed(with_descriptions(description_line(MODEL, description))).backends['ollama'].descriptions == {
        MODEL: description.strip()}
    assert DESCRIPTIONS.read_wire(description, 'the description') == description.strip()


@pytest.mark.parametrize('description', REFUSED_BY_BOTH)
def test_what_the_gateway_refuses_the_configuration_and_the_wire_refuse(description):
    with pytest.raises(ConfigError):
        parsed(with_descriptions(description_line(MODEL, description)))
    with pytest.raises(ConfigError):
        DESCRIPTIONS.read_wire(description, 'the description')


def test_the_refused_characters_are_those_of_the_gateway():
    assert protocol.DESCRIPTION_REFUSED_CODE_POINTS == {ord(character) for character in GATEWAY_REFUSED_CHARACTERS}
    assert protocol.DESCRIPTION_REFUSED_CATEGORIES == GATEWAY_REFUSED_CATEGORIES


def test_descriptions_that_are_no_table_are_refused():
    text = HAND_WRITTEN.replace(DESCRIPTION_TABLE, '').replace('kind = "ollama"\n',
                                                               'kind = "ollama"\ndescriptions = "all quick"\n')
    with pytest.raises(ConfigError, match=r'descriptions must be a table, \[backends.descriptions\]'):
        parsed(text)


def test_a_description_of_a_model_not_shared_says_it_can_simply_be_deleted():
    with pytest.raises(ConfigError) as refused:
        parsed(with_descriptions(f'"{OTHER_MODEL}" = "Careful"'))
    assert str(refused.value) == (
        'Backend ollama: the line "qwen3:14b" = ... under [backends.descriptions] describes a model that is not '
        'among its models. If you no longer share that model, simply delete the line; if you do, check the name '
        'or add it to models')


def test_the_description_table_as_the_example_configuration_shows_it_is_valid_with_the_prices(monkeypatch):
    text = EXAMPLE_CONFIG.read_text()
    for commented in ('# [backends.prices_per_1m_tokens]', '# "llama3.1:8b" = { input = 0.80, output = 2.00 }',
                      '# [backends.descriptions]', '# "llama3.1:8b" = "Quick answers to short questions, in English '
                                                    'and German"'):
        assert text.count(commented) == 1
        text = text.replace(commented, commented.removeprefix('# '))
    for name in ('SOCKETLLAMA_TOKEN', 'SOCKETLLAMA_SIGNING_SECRET', 'LITELLM_API_KEY'):
        monkeypatch.setenv(name, 'example')
    ollama = parse_config(tomllib.loads(text)).backends['ollama']
    assert ollama.descriptions == {MODEL: 'Quick answers to short questions, in English and German'}
    assert ollama.prices == {MODEL: FILE_PRICE}


# ----- the editor -----

def test_a_description_is_replaced_in_its_line_and_nothing_else_changes():
    edited = edit_description(HAND_WRITTEN, 'ollama', MODEL, NEW_DESCRIPTION)
    # indentation, key, the comment after it, every other line: as they were
    assert edited == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)


def test_a_model_without_a_line_gets_one_at_the_end_of_its_table():
    edited = edit_description(HAND_WRITTEN, 'ollama', OTHER_MODEL, NEW_DESCRIPTION)
    described = f'  "{MODEL}" = {OLD_LINE_VALUE}   # since November\n'
    # after the last description, indented like it, before the commented-out line
    assert edited == HAND_WRITTEN.replace(described, f'{described}  "{OTHER_MODEL}" = {NEW_LINE_VALUE}\n')


def test_a_backend_without_a_table_gets_one_at_the_end_of_its_entry():
    edited = edit_description(HAND_WRITTEN, 'litellm', PROXY_MODEL, NEW_DESCRIPTION)
    assert edited == HAND_WRITTEN + (f'\n{DESCRIPTIONS.comment}\n[backends.descriptions]\n'
                                     f'{PROXY_MODEL} = {NEW_LINE_VALUE}\n')
    assert tomllib.loads(edited) == expected_data(HAND_WRITTEN, 1, PROXY_MODEL, NEW_DESCRIPTION)


def test_a_backend_with_a_price_table_gets_its_description_table_after_it():
    text = HAND_WRITTEN.replace(DESCRIPTION_TABLE, '')
    edited = edit_description(text, 'ollama', MODEL, NEW_DESCRIPTION)
    assert edited == text.replace(PRICE_TABLE, f'{PRICE_TABLE}\n{DESCRIPTIONS.comment}\n[backends.descriptions]\n'
                                               f'"{MODEL}" = {NEW_LINE_VALUE}\n')
    assert tomllib.loads(edited) == expected_data(text, 0, MODEL, NEW_DESCRIPTION)


def test_a_backend_with_a_description_table_gets_its_price_table_after_it():
    text = HAND_WRITTEN.replace(PRICE_TABLE, '')
    edited = edit_price(text, 'ollama', MODEL, NEW_PRICE)
    lines = edited.splitlines()
    assert lines.index('[backends.descriptions]') < lines.index('[backends.prices_per_1m_tokens]') < lines.index(
        'name = "litellm"')
    expected = tomllib.loads(text)
    expected['backends'][0]['prices_per_1m_tokens'] = {MODEL: NEW_PRICE.as_table()}
    assert tomllib.loads(edited) == expected
    # and the description in that file is changed in its line as ever
    again = edit_description(edited, 'ollama', MODEL, NEW_DESCRIPTION)
    assert again == edited.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)


def test_a_price_and_a_description_in_one_file_are_each_changed_in_their_own_line():
    edited = edit_description(edit_price(HAND_WRITTEN, 'ollama', MODEL, NEW_PRICE), 'ollama', MODEL, NEW_DESCRIPTION)
    assert edited == HAND_WRITTEN.replace(OLD_PRICE_VALUE, NEW_PRICE_VALUE).replace(OLD_LINE_VALUE, NEW_LINE_VALUE)


@pytest.mark.parametrize('description', AWKWARD_DESCRIPTIONS,
                         ids=['quotes', 'backslashes', 'not ascii', 'invisible characters', 'soft hyphen', 'hash'])
def test_any_description_is_written_so_that_it_reads_back_as_exactly_itself(description):
    edited = edit_description(HAND_WRITTEN, 'ollama', MODEL, description)
    assert tomllib.loads(edited) == expected_data(HAND_WRITTEN, 0, MODEL, description)
    # and its line is found again for the next change, the comment after it kept
    again = edit_description(edited, 'ollama', MODEL, LATER_DESCRIPTION)
    assert again == HAND_WRITTEN.replace(OLD_LINE_VALUE, f'"{LATER_DESCRIPTION}"')


def test_quotes_and_backslashes_are_escaped_and_other_characters_written_as_themselves():
    assert description_line(MODEL, 'say "hi" \\ Grüezi 🦙') == f'"{MODEL}" = "say \\"hi\\" \\\\ Grüezi 🦙"'
    # a character that cannot be shown as itself is written as its escape, never as itself
    assert description_line(MODEL, 'no-break\u00a0space') == f'"{MODEL}" = "no-break\\u00A0space"'


@pytest.mark.parametrize('written', [f"'{OLD_DESCRIPTION}'", f'"{OLD_DESCRIPTION[:5]}\\u0020{OLD_DESCRIPTION[6:]}"',
                                     '"Quick \\"answers\\""'],
                         ids=['literal string', 'escape', 'escaped quotes'])
def test_a_line_is_replaced_however_its_string_is_written(written):
    text = HAND_WRITTEN.replace(OLD_LINE_VALUE, written)
    assert edit_description(text, 'ollama', MODEL, NEW_DESCRIPTION) == HAND_WRITTEN.replace(OLD_LINE_VALUE,
                                                                                             NEW_LINE_VALUE)


@pytest.mark.parametrize('written', [f'"""{OLD_DESCRIPTION}"""', f"'''{OLD_DESCRIPTION}'''"],
                         ids=['basic', 'literal'])
def test_a_string_of_three_quotes_is_refused(written):
    text = HAND_WRITTEN.replace(OLD_LINE_VALUE, written)
    with pytest.raises(LineEditRefused, match='not written on one line as "<text>"'):
        edit_description(text, 'ollama', MODEL, NEW_DESCRIPTION)


def test_descriptions_as_an_inline_table_on_the_backend_are_refused():
    text = HAND_WRITTEN.replace(DESCRIPTION_TABLE, '').replace(
        'kind = "ollama"\n', f'kind = "ollama"\ndescriptions = {{ "{MODEL}" = {OLD_LINE_VALUE} }}\n')
    for model in (MODEL, OTHER_MODEL):
        with pytest.raises(LineEditRefused, match=r'write the description as a line of a \[backends.descriptions\]'):
            edit_description(text, 'ollama', model, NEW_DESCRIPTION)


def test_in_the_example_configuration_the_table_stays_with_its_backend():
    """Below the commented-out settings and tables of the backend, before the next one."""
    text = EXAMPLE_CONFIG.read_text()
    for index, (backend, model) in enumerate([('ollama', MODEL), ('litellm', 'gpt-4o-mini')]):
        edited = edit_description(text, backend, model, NEW_DESCRIPTION)
        assert tomllib.loads(edited) == expected_data(text, index, model, NEW_DESCRIPTION)
        lines = edited.splitlines()
        added = lines.index(description_line(model, NEW_DESCRIPTION))
        assert lines[added + 1] == ''
    lines = edit_description(text, 'ollama', MODEL, NEW_DESCRIPTION).splitlines()
    assert lines.index('# [backends.descriptions]') < lines.index('[backends.descriptions]') < lines.index(
        'name = "litellm"')


def test_windows_line_endings_are_kept():
    text = HAND_WRITTEN.replace('\n', '\r\n')
    assert edit_description(text, 'ollama', MODEL, NEW_DESCRIPTION) == text.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)
    added = edit_description(text, 'litellm', PROXY_MODEL, NEW_DESCRIPTION)
    assert '\n' not in added.replace('\r\n', '')


@pytest.mark.parametrize('text, backend, model, reason', [
    (HAND_WRITTEN.replace('[gateway]', '[gateway'), 'ollama', MODEL, 'not valid TOML'),
    (HAND_WRITTEN, 'vllm', MODEL, "no backend 'vllm'"),
    (HAND_WRITTEN, 'ollama', NOT_SHARED, 'would not load with a description for it'),
])
def test_what_cannot_be_written_says_why(text, backend, model, reason):
    with pytest.raises(LineEditRefused, match=reason):
        edit_description(text, backend, model, NEW_DESCRIPTION)


# ----- the descriptions the client holds, and writing them -----

@pytest.fixture(autouse=True)
def secrets_from_the_environment(monkeypatch):
    for name, value in environ().items():
        monkeypatch.setenv(name, value)


def prepared(tmp_path, text: str = HAND_WRITTEN, mode: int = OWNER_ONLY) -> tuple[Path, ConfigSettings]:
    path = tmp_path / 'config.toml'
    path.write_text(text)
    path.chmod(mode)
    return path, ConfigSettings(load_config(path))


def entry(backend: str, model: str, description: object = None, price: ModelPrice | None = None) -> dict:
    """An entry of a models.settings frame, with a description, a price, both or neither."""
    return {'backend': backend, 'model': model,
            **({'description': description} if description is not None else {}),
            **({'price': {'input': price.input, 'output': price.output}} if price is not None else {})}


def settings(*entries) -> dict:
    return {'type': protocol.FRAME_MODELS_SETTINGS, 'models': list(entries)}


def take(held: ConfigSettings, frame: dict, acknowledged: frozenset | None = None,
         tables=(PRICES, DESCRIPTIONS)) -> bool:
    """``held.take_settings`` for a gateway that takes ``tables``, with every model the frame names taken to be
    listed by an acknowledged report unless ``acknowledged`` says which were."""
    if acknowledged is None:
        acknowledged = frozenset((item.get('backend'), item.get('model')) for item in frame['models']
                                 if isinstance(item, dict) and isinstance(item.get('model'), str))
    return held.take_settings(frame, {table: acknowledged for table in tables})


def backups(directory: Path) -> list[Path]:
    return sorted(directory.glob('config.toml.backup-*'))


def warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]


def test_the_descriptions_of_the_file_are_held_as_loaded(tmp_path):
    _, held = prepared(tmp_path)
    assert held.descriptions == {('ollama', MODEL): OLD_DESCRIPTION}
    assert held.prices == {('ollama', MODEL): FILE_PRICE}


def test_a_report_carries_what_the_file_holds_for_the_tables_the_gateway_takes(tmp_path):
    _, held = prepared(tmp_path, HAND_WRITTEN.replace('models = []', 'models = []\n\n[backends.descriptions]\n'
                                                                        f'{PROXY_MODEL} = "Through the proxy"'))
    wire_price = {protocol.PRICE_INPUT_FIELD: FILE_PRICE.input, protocol.PRICE_OUTPUT_FIELD: FILE_PRICE.output}
    assert held.report_fields((PRICES, DESCRIPTIONS)) == {
        ('ollama', MODEL): {protocol.CONFIG_PRICE_FIELD: wire_price,
                            protocol.CONFIG_DESCRIPTION_FIELD: OLD_DESCRIPTION},
        ('litellm', PROXY_MODEL): {protocol.CONFIG_DESCRIPTION_FIELD: 'Through the proxy'}}
    assert held.report_fields((PRICES,)) == {('ollama', MODEL): {protocol.CONFIG_PRICE_FIELD: wire_price}}
    assert held.report_fields(()) == {}


def test_a_description_toptic_sets_is_written_and_held_from_then_on(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path, held = prepared(tmp_path)
    assert take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION))) is True
    assert path.read_text() == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)
    assert held.descriptions[('ollama', MODEL)] == NEW_DESCRIPTION
    assert f'toptic.ai set the description of ollama/{MODEL} to "{NEW_DESCRIPTION}": written to {path}' in caplog.text


def test_a_description_toptic_sets_is_taken_without_the_blanks_around_it(tmp_path):
    path, held = prepared(tmp_path)
    assert take(held, settings(entry('ollama', MODEL, f' {NEW_DESCRIPTION}  ')))
    assert load_config(path).backends['ollama'].descriptions[MODEL] == NEW_DESCRIPTION


def test_a_description_the_client_holds_already_writes_nothing(tmp_path):
    path, held = prepared(tmp_path)
    before = path.stat()
    assert take(held, settings(entry('ollama', MODEL, OLD_DESCRIPTION))) is False
    assert path.stat().st_ino == before.st_ino and backups(tmp_path) == []


def test_a_price_and_a_description_of_one_entry_are_one_write(tmp_path):
    path, held = prepared(tmp_path)
    assert take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION, NEW_PRICE)))
    assert path.read_text() == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE).replace(OLD_PRICE_VALUE,
                                                                                            NEW_PRICE_VALUE)
    assert len(backups(tmp_path)) == 1
    assert held.prices[('ollama', MODEL)] == NEW_PRICE and held.descriptions[('ollama', MODEL)] == NEW_DESCRIPTION


def test_a_price_and_a_later_description_share_the_one_backup_of_the_run(tmp_path):
    path, held = prepared(tmp_path)
    take(held, settings(entry('ollama', MODEL, price=NEW_PRICE)))
    [backup] = backups(tmp_path)
    take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION)))
    assert backups(tmp_path) == [backup] and backup.read_text() == HAND_WRITTEN
    assert load_config(path).backends['ollama'].descriptions[MODEL] == NEW_DESCRIPTION


@pytest.mark.parametrize('bad_description', [TOO_LONG, '', 'one\ntwo', 7, ['Quick']],
                         ids=['too long', 'empty', 'two lines', 'number', 'list'])
def test_a_description_that_is_none_does_not_stop_the_price_beside_it(tmp_path, caplog, bad_description):
    path, held = prepared(tmp_path)
    assert take(held, settings(entry('ollama', MODEL, bad_description, NEW_PRICE)))
    assert path.read_text() == HAND_WRITTEN.replace(OLD_PRICE_VALUE, NEW_PRICE_VALUE)
    assert held.descriptions[('ollama', MODEL)] == OLD_DESCRIPTION
    [warning] = warnings(caplog)
    assert warning.startswith('Ignored a description toptic.ai sent: the description of ollama/')


@pytest.mark.parametrize('bad_price', [{'input': -1.0, 'output': 1.0}, {'input': 1.0}, None],
                         ids=['negative', 'no output', 'null'])
def test_a_price_that_is_none_does_not_stop_the_description_beside_it(tmp_path, caplog, bad_price):
    path, held = prepared(tmp_path)
    assert take(held, settings({**entry('ollama', MODEL, NEW_DESCRIPTION), 'price': bad_price}))
    assert path.read_text() == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)
    assert held.prices[('ollama', MODEL)] == FILE_PRICE
    [warning] = warnings(caplog)
    assert warning.startswith('Ignored a price toptic.ai sent')


def test_an_entry_that_carries_neither_is_warned_about(tmp_path, caplog):
    path, held = prepared(tmp_path)
    assert take(held, settings(entry('ollama', MODEL))) is False
    assert path.read_text() == HAND_WRITTEN
    assert warnings(caplog) == [f'Ignored what toptic.ai sent for a model: the entry of ollama/{MODEL} carries no '
                                f'price or description']


def test_a_gateway_that_takes_prices_only_has_its_description_ignored(tmp_path):
    path, held = prepared(tmp_path)
    assert take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION, NEW_PRICE)), tables=(PRICES,))
    assert path.read_text() == HAND_WRITTEN.replace(OLD_PRICE_VALUE, NEW_PRICE_VALUE)
    assert held.descriptions[('ollama', MODEL)] == OLD_DESCRIPTION


def test_a_gateway_that_takes_descriptions_only_has_its_price_ignored(tmp_path):
    path, held = prepared(tmp_path)
    assert take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION, NEW_PRICE)), tables=(DESCRIPTIONS,))
    assert path.read_text() == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)


def test_a_description_for_a_model_no_acknowledged_report_listed_is_not_taken(tmp_path):
    """Its backend was down when the client started, say: toptic.ai has not seen the line yet."""
    path, held = prepared(tmp_path)
    frame = settings(entry('ollama', MODEL, NEW_DESCRIPTION), entry('ollama', OTHER_MODEL, LATER_DESCRIPTION))
    assert take(held, frame, acknowledged=frozenset({('ollama', OTHER_MODEL)}))
    assert load_config(path).backends['ollama'].descriptions == {MODEL: OLD_DESCRIPTION, OTHER_MODEL: LATER_DESCRIPTION}


def test_a_description_for_a_model_only_a_report_without_descriptions_listed_is_not_taken(tmp_path):
    """The report acknowledged on an earlier connection carried prices only (that toptic.ai did not take
    descriptions): toptic.ai has seen the price line of the model, not its description line."""
    path, held = prepared(tmp_path)
    listed = frozenset({('ollama', MODEL)})
    frame = settings(entry('ollama', MODEL, NEW_DESCRIPTION, NEW_PRICE))
    assert held.take_settings(frame, {PRICES: listed, DESCRIPTIONS: frozenset()})
    assert path.read_text() == HAND_WRITTEN.replace(OLD_PRICE_VALUE, NEW_PRICE_VALUE)
    assert held.descriptions[('ollama', MODEL)] == OLD_DESCRIPTION


def test_a_read_only_file_is_left_alone_and_the_line_to_write_is_shown_once(tmp_path, caplog):
    path, held = prepared(tmp_path, mode=READ_ONLY)
    awkward = AWKWARD_DESCRIPTIONS[0]
    for _ in range(2):  # toptic.ai offers it again at every report
        assert take(held, settings(entry('ollama', MODEL, awkward))) is False
    assert path.read_text() == HAND_WRITTEN
    assert held.descriptions[('ollama', MODEL)] == OLD_DESCRIPTION  # still reported
    [warning] = warnings(caplog)
    assert 'read-only' in warning and '[backends.descriptions] of backend ollama' in warning
    # the line as it is to be written, escaped, so that it can be put in as it stands
    assert warning.endswith(description_line(MODEL, awkward))
    assert tomllib.loads(description_line(MODEL, awkward)) == {MODEL: awkward}


def test_a_description_changed_by_hand_since_the_start_is_not_written_over(tmp_path, caplog):
    path, held = prepared(tmp_path)
    hand_edited = HAND_WRITTEN.replace(OLD_LINE_VALUE, f'"{HAND_EDITED_DESCRIPTION}"')
    path.write_text(hand_edited)
    assert take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION))) is False
    assert path.read_text() == hand_edited
    assert 'Restart the client to report the description in the file' in caplog.text


def test_a_line_changed_by_hand_to_the_description_toptic_sets_is_held_without_writing(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path, held = prepared(tmp_path)
    hand_edited = HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)
    path.write_text(hand_edited)
    assert take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION)))
    assert path.read_text() == hand_edited and backups(tmp_path) == []
    assert held.descriptions[('ollama', MODEL)] == NEW_DESCRIPTION
    assert 'shows already' in caplog.text


def test_a_line_removed_by_hand_is_put_back(tmp_path):
    path, held = prepared(tmp_path)
    path.write_text(HAND_WRITTEN.replace(f'  "{MODEL}" = {OLD_LINE_VALUE}   # since November\n', ''))
    assert take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION)))
    assert tomllib.loads(path.read_text()) == expected_data(HAND_WRITTEN, 0, MODEL, NEW_DESCRIPTION)


def test_a_description_that_cannot_be_written_in_its_line_leaves_the_price_beside_it_written(tmp_path, caplog):
    text = HAND_WRITTEN.replace(OLD_LINE_VALUE, f'"""{OLD_DESCRIPTION}"""')
    path, held = prepared(tmp_path, text)
    assert take(held, settings(entry('ollama', MODEL, NEW_DESCRIPTION, NEW_PRICE)))
    assert path.read_text() == text.replace(OLD_PRICE_VALUE, NEW_PRICE_VALUE)
    [warning] = warnings(caplog)
    assert 'not written on one line' in warning and warning.endswith(description_line(MODEL, NEW_DESCRIPTION))
