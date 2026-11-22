"""The prices of config.toml: written in place by the editor, every other byte kept, and only what it verified."""
import logging
import math
import os
import stat
import tomllib
from pathlib import Path

import pytest

from socketllama import config_settings, config_writer, protocol
from socketllama.config import ModelPrice, load_config, parse_config
from socketllama.config_settings import MODEL_TABLES, ConfigSettings, LineEditRefused, edit_price, price_line

MODEL = 'llama3.1:8b'
OTHER_MODEL = 'qwen3:14b'
NOT_SHARED = 'mistral:7b'
TOKEN_VARIABLE = 'PRICES_TEST_TOKEN'
SECRET_VARIABLE = 'PRICES_TEST_SECRET'
# A file as somebody writes it by hand: comments everywhere, indented lines, a comment after a price
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

    # What the electricity costs
[backends.prices_per_1m_tokens]
  "{MODEL}" = {{ input = 0.80, output = 2.00 }}   # since November
  # "{OTHER_MODEL}" = {{ input = 1, output = 3 }}

# The proxy at work
[[backends]]
name = "litellm"
kind = "litellm"
base_url = "http://llm.invalid/v1"
models = []
'''
OLD_LINE_VALUE = '{ input = 0.80, output = 2.00 }'
NEW_PRICE = ModelPrice(input=0.4, output=1.4)
NEW_LINE_VALUE = '{ input = 0.4, output = 1.4 }'
LATER_PRICE = ModelPrice(input=0.6, output=1.6)
HAND_EDITED_PRICE = ModelPrice(input=0.9, output=2.1)
# Not the shortest decimal of anything: must still read back as exactly this float
AWKWARD_PRICE = ModelPrice(input=0.1 + 0.2, output=1 / 3)
PROXY_MODEL = 'gpt-proxy'
EXAMPLE_CONFIG = Path(__file__).resolve().parent.parent / 'config.example.toml'
OWNER_ONLY = 0o600
GROUP_READABLE = 0o640
WORLD_READABLE = 0o644
READ_ONLY = 0o400
NO_NEW_FILES = 0o500
EVERYTHING = 0o700


def expected_data(text: str, index: int, model: str, price: ModelPrice) -> dict:
    data = tomllib.loads(text)
    data['backends'][index].setdefault('prices_per_1m_tokens', {})[model] = price.as_table()
    return data


# ----- the editor -----

def test_a_price_is_replaced_in_its_line_and_nothing_else_changes():
    edited = edit_price(HAND_WRITTEN, 'ollama', MODEL, NEW_PRICE)
    # indentation, key, the comment after it, every other line: as they were
    assert edited == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)


def test_a_model_without_a_line_gets_one_at_the_end_of_its_table():
    edited = edit_price(HAND_WRITTEN, 'ollama', OTHER_MODEL, NEW_PRICE)
    priced = f'  "{MODEL}" = {OLD_LINE_VALUE}   # since November\n'
    # after the last price, indented like it, before the commented-out line
    assert edited == HAND_WRITTEN.replace(priced, f'{priced}  "{OTHER_MODEL}" = {NEW_LINE_VALUE}\n')


def test_a_backend_without_a_table_gets_one_at_the_end_of_its_entry():
    edited = edit_price(HAND_WRITTEN, 'litellm', PROXY_MODEL, NEW_PRICE)
    assert edited == HAND_WRITTEN + (f'\n{config_settings.PRICES.comment}\n[backends.prices_per_1m_tokens]\n'
                                     f'{PROXY_MODEL} = {NEW_LINE_VALUE}\n')
    assert tomllib.loads(edited) == expected_data(HAND_WRITTEN, 1, PROXY_MODEL, NEW_PRICE)


def test_the_first_of_two_backends_gets_its_table_before_the_second():
    text = HAND_WRITTEN.replace(f'''
    # What the electricity costs
[backends.prices_per_1m_tokens]
  "{MODEL}" = {OLD_LINE_VALUE}   # since November
  # "{OTHER_MODEL}" = {{ input = 1, output = 3 }}
''', '')
    edited = edit_price(text, 'ollama', MODEL, NEW_PRICE)
    assert tomllib.loads(edited) == expected_data(text, 0, MODEL, NEW_PRICE)
    lines = edited.splitlines()
    assert lines.index('[backends.prices_per_1m_tokens]') == lines.index(f'models = ["{MODEL}", "{OTHER_MODEL}"]') + 3
    assert lines.index('[backends.prices_per_1m_tokens]') < lines.index('name = "litellm"')


def test_the_same_model_of_the_second_backend_is_its_own():
    text = HAND_WRITTEN.replace('models = []', f'models = []\n\n[backends.prices_per_1m_tokens]\n"{MODEL}" = '
                                               f'{{ input = 5.0, output = 6.0 }}')
    edited = edit_price(text, 'litellm', MODEL, NEW_PRICE)
    assert tomllib.loads(edited) == expected_data(text, 1, MODEL, NEW_PRICE)
    assert f'"{MODEL}" = {OLD_LINE_VALUE}' in edited  # the first backend's line is untouched


def test_in_the_example_configuration_the_table_stays_with_its_backend():
    """Commented-out settings directly below a backend stay above its new table, so that taking one in
    later does not put it into the price table; comments after an empty line belong to what follows."""
    text = EXAMPLE_CONFIG.read_text()
    for index, (backend, model) in enumerate([('ollama', MODEL), ('litellm', 'gpt-4o-mini')]):
        edited = edit_price(text, backend, model, NEW_PRICE)
        assert tomllib.loads(edited) == expected_data(text, index, model, NEW_PRICE)
        added = edited.splitlines().index(price_line(model, NEW_PRICE))
        # what follows the new line is an empty line, then the next backend or the comments about one
        assert edited.splitlines()[added + 1] == ''
    edited = edit_price(text, 'ollama', MODEL, NEW_PRICE).splitlines()
    assert edited.index('# load_at_start = true') < edited.index('[backends.prices_per_1m_tokens]')


@pytest.mark.parametrize('written_key', [f"'{MODEL}'", f'"{MODEL}"', '"llama3\\u002e1:8b"'])
def test_a_key_is_found_however_it_is_quoted(written_key):
    text = HAND_WRITTEN.replace(f'"{MODEL}" = {OLD_LINE_VALUE}', f'{written_key} = {OLD_LINE_VALUE}')
    edited = edit_price(text, 'ollama', MODEL, NEW_PRICE)
    assert edited == text.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)


@pytest.mark.parametrize('model', ['mistral', 'with "quotes" and \\ backslash', 'ünïcödé 🦙', 'dotted.name'])
def test_any_model_name_is_written_so_that_it_reads_back(model):
    edited = edit_price(HAND_WRITTEN, 'litellm', model, NEW_PRICE)
    assert tomllib.loads(edited) == expected_data(HAND_WRITTEN, 1, model, NEW_PRICE)
    # and found again for the next change
    again = edit_price(edited, 'litellm', model, LATER_PRICE)
    assert tomllib.loads(again) == expected_data(HAND_WRITTEN, 1, model, LATER_PRICE)
    assert again.count(price_line(model, LATER_PRICE)) == 1


def test_a_price_reads_back_as_exactly_the_number_sent():
    edited = edit_price(HAND_WRITTEN, 'ollama', MODEL, AWKWARD_PRICE)
    assert tomllib.loads(edited)['backends'][0]['prices_per_1m_tokens'][MODEL] == AWKWARD_PRICE.as_table()


def test_windows_line_endings_are_kept():
    text = HAND_WRITTEN.replace('\n', '\r\n')
    assert edit_price(text, 'ollama', MODEL, NEW_PRICE) == text.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)
    added = edit_price(text, 'litellm', PROXY_MODEL, NEW_PRICE)
    assert '\n' not in added.replace('\r\n', '')
    assert tomllib.loads(added) == expected_data(text, 1, PROXY_MODEL, NEW_PRICE)


def test_a_file_without_a_final_line_break_gets_its_table_on_a_line_of_its_own():
    text = HAND_WRITTEN.removesuffix('\n')
    edited = edit_price(text, 'litellm', PROXY_MODEL, NEW_PRICE)
    assert edited.startswith(text + '\n\n')
    assert tomllib.loads(edited) == expected_data(text, 1, PROXY_MODEL, NEW_PRICE)


PRICE_TABLE = f'''[backends.prices_per_1m_tokens]
  "{MODEL}" = {OLD_LINE_VALUE}   # since November
'''


@pytest.mark.parametrize('model, replacement', [
    # dotted keys
    (MODEL, f'[backends.prices_per_1m_tokens]\n"{MODEL}".input = 0.8\n"{MODEL}".output = 2.0\n'),
    # a table per model, for that model and for another one
    (MODEL, f'[backends.prices_per_1m_tokens."{MODEL}"]\ninput = 0.8\noutput = 2.0\n'),
    (OTHER_MODEL, f'[backends.prices_per_1m_tokens."{MODEL}"]\ninput = 0.8\noutput = 2.0\n'),
    # the header written another way
    (MODEL, f'[backends."prices_per_1m_tokens"]\n"{MODEL}" = {OLD_LINE_VALUE}\n'),
    # a price that is no inline table on its line
    (MODEL, f'[backends.prices_per_1m_tokens]\n"{MODEL}" = 2.0\n'),
], ids=['dotted keys', 'table per model', 'table per other model', 'quoted header', 'not an inline table'])
def test_a_layout_the_editor_does_not_follow_is_refused(model, replacement):
    text = HAND_WRITTEN.replace(PRICE_TABLE, replacement)
    assert text != HAND_WRITTEN
    with pytest.raises(LineEditRefused):
        edit_price(text, 'ollama', model, NEW_PRICE)


def test_prices_as_an_inline_table_on_the_backend_are_refused():
    text = HAND_WRITTEN.replace(PRICE_TABLE, '').replace(
        'kind = "ollama"\n', f'kind = "ollama"\nprices_per_1m_tokens = {{ "{MODEL}" = {OLD_LINE_VALUE} }}\n')
    for model in (MODEL, OTHER_MODEL):
        with pytest.raises(LineEditRefused):
            edit_price(text, 'ollama', model, NEW_PRICE)


def test_backends_written_as_an_inline_array_are_refused():
    text = ('[gateway]\nurl = "wss://x.example/ws/"\n'
            f'backends = [{{ name = "ollama", kind = "ollama", base_url = "http://o.invalid/v1", models = [] }}]\n')
    with pytest.raises(LineEditRefused, match='not \\[\\[backends\\]\\] entries'):
        edit_price(text, 'ollama', MODEL, NEW_PRICE)


def test_a_header_inside_a_multi_line_string_is_not_taken_for_one():
    text = HAND_WRITTEN.replace('instance = "tester-home"\n',
                                'instance = "tester-home"\nnote = """\n[[backends]]\n"""\n')
    with pytest.raises(LineEditRefused):
        edit_price(text, 'litellm', PROXY_MODEL, NEW_PRICE)


@pytest.mark.parametrize('text, backend, model, reason', [
    (HAND_WRITTEN.replace('[gateway]', '[gateway'), 'ollama', MODEL, 'not valid TOML'),
    (HAND_WRITTEN, 'vllm', MODEL, "no backend 'vllm'"),
    (HAND_WRITTEN, 'ollama', NOT_SHARED, 'does not share'),
])
def test_what_cannot_be_written_says_why(text, backend, model, reason):
    with pytest.raises(LineEditRefused, match=reason):
        edit_price(text, backend, model, NEW_PRICE)


# ----- the prices the client holds, and writing them -----

@pytest.fixture(autouse=True)
def secrets_from_the_environment(monkeypatch):
    monkeypatch.setenv(TOKEN_VARIABLE, 'tt_live_x')
    monkeypatch.setenv(SECRET_VARIABLE, 'secret')


def prepared(tmp_path, text: str = HAND_WRITTEN, mode: int = OWNER_ONLY) -> tuple[Path, ConfigSettings]:
    path = tmp_path / 'config.toml'
    path.write_text(text)
    path.chmod(mode)
    return path, ConfigSettings(load_config(path))


def settings(*entries) -> dict:
    return {'type': protocol.FRAME_MODELS_SETTINGS, 'models': [
        {'backend': backend, 'model': model, 'price': {'input': price.input, 'output': price.output}}
        for backend, model, price in entries]}


def take(prices: ConfigSettings, frame: dict, acknowledged: frozenset | None = None) -> bool:
    """``prices.take_settings``, with every model the frame names taken to be listed by an acknowledged report
    (of every table) unless ``acknowledged`` says which were."""
    if acknowledged is None:
        entries = frame.get('models') if isinstance(frame.get('models'), list) else []
        acknowledged = frozenset((entry.get('backend'), entry.get('model')) for entry in entries
                                 if isinstance(entry, dict) and isinstance(entry.get('model'), str))
    return prices.take_settings(frame, {table: acknowledged for table in MODEL_TABLES})


def backups(directory: Path) -> list[Path]:
    return sorted(directory.glob('config.toml.backup-*'))


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_prices_of_the_file_are_held_as_loaded(tmp_path):
    _, prices = prepared(tmp_path)
    assert prices.prices == {('ollama', MODEL): ModelPrice(input=0.8, output=2.0)}


def test_a_price_toptic_sets_is_written_and_held_from_then_on(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path, prices = prepared(tmp_path)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is True
    assert path.read_text() == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)
    assert prices.prices[('ollama', MODEL)] == NEW_PRICE
    assert f'toptic.ai set the price of ollama/{MODEL} to $0.40 in, $1.40 out per 1M tokens' in caplog.text


def test_the_first_write_of_a_run_keeps_a_backup_for_the_owner_only(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path, prices = prepared(tmp_path, mode=GROUP_READABLE)
    take(prices, settings(('ollama', MODEL, NEW_PRICE)))
    [backup] = backups(tmp_path)
    assert backup.read_text() == HAND_WRITTEN
    assert mode_of(backup) == OWNER_ONLY
    assert str(backup) in caplog.text
    take(prices, settings(('ollama', MODEL, LATER_PRICE)))
    assert backups(tmp_path) == [backup]  # not one per price


@pytest.mark.parametrize('mode', [OWNER_ONLY, GROUP_READABLE, WORLD_READABLE])
def test_the_file_keeps_its_permissions(tmp_path, mode):
    path, prices = prepared(tmp_path, mode=mode)
    take(prices, settings(('ollama', MODEL, NEW_PRICE)))
    assert mode_of(path) == mode


def test_a_symbolic_link_stays_one_and_the_file_it_leads_to_is_written(tmp_path):
    real = tmp_path / 'real'
    real.mkdir()
    (real / 'config.toml').write_text(HAND_WRITTEN)
    link = tmp_path / 'config.toml'
    link.symlink_to(real / 'config.toml')
    prices = ConfigSettings(load_config(link))
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE)))
    assert link.is_symlink()
    assert (real / 'config.toml').read_text() == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)


def test_a_price_the_client_holds_already_writes_nothing(tmp_path):
    path, prices = prepared(tmp_path)
    before = path.stat()
    # as toptic.ai compares prices: a float that differs in its last place is the same price
    same = ModelPrice(input=math.nextafter(0.8, 1.0), output=2.0)
    assert take(prices, settings(('ollama', MODEL, same))) is False
    assert path.stat().st_ino == before.st_ino and path.read_text() == HAND_WRITTEN
    assert backups(tmp_path) == []


def test_a_read_only_file_is_left_alone_and_the_line_to_write_is_shown_once(tmp_path, caplog):
    path, prices = prepared(tmp_path, mode=READ_ONLY)
    for _ in range(2):  # toptic.ai offers it again at every report
        assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    assert path.read_text() == HAND_WRITTEN
    assert prices.prices[('ollama', MODEL)] == ModelPrice(input=0.8, output=2.0)  # still reported
    [warning] = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert 'read-only' in warning
    assert price_line(MODEL, NEW_PRICE) in warning
    # a new price is a new warning
    take(prices, settings(('ollama', MODEL, LATER_PRICE)))
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 2


def test_a_folder_that_takes_no_new_file_leaves_the_file_alone(tmp_path):
    directory = tmp_path / 'locked'
    directory.mkdir()
    path, prices = prepared(directory)
    directory.chmod(NO_NEW_FILES)
    try:
        assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    finally:
        directory.chmod(EVERYTHING)
    assert path.read_text() == HAND_WRITTEN
    assert sorted(entry.name for entry in directory.iterdir()) == ['config.toml']


def test_a_price_changed_by_hand_since_the_start_is_not_written_over(tmp_path, caplog):
    path, prices = prepared(tmp_path)
    hand_edited = HAND_WRITTEN.replace(OLD_LINE_VALUE, '{ input = 0.9, output = 2.1 }')
    path.write_text(hand_edited)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    assert path.read_text() == hand_edited
    assert 'Restart the client' in caplog.text


def test_a_line_added_by_hand_since_the_start_is_not_written_over(tmp_path):
    path, prices = prepared(tmp_path)
    hand_edited = HAND_WRITTEN.replace('  # "qwen3', f'  "{OTHER_MODEL}" = {{ input = 1, output = 3 }}\n  # "qwen3')
    path.write_text(hand_edited)
    assert take(prices, settings(('ollama', OTHER_MODEL, NEW_PRICE))) is False
    assert path.read_text() == hand_edited


def test_other_changes_made_by_hand_since_the_start_are_kept(tmp_path):
    path, prices = prepared(tmp_path)
    hand_edited = HAND_WRITTEN.replace('# The proxy at work', '# The proxy at work, moved').replace(
        'http://llm.invalid/v1', 'http://other.invalid/v1')
    path.write_text(hand_edited)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE)))
    assert path.read_text() == hand_edited.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)


def test_a_line_removed_by_hand_is_put_back(tmp_path):
    path, prices = prepared(tmp_path)
    path.write_text(HAND_WRITTEN.replace(f'  "{MODEL}" = {OLD_LINE_VALUE}   # since November\n', ''))
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE)))
    assert tomllib.loads(path.read_text()) == expected_data(HAND_WRITTEN, 0, MODEL, NEW_PRICE)


def test_a_file_that_is_being_edited_is_left_alone(tmp_path, caplog):
    path, prices = prepared(tmp_path)
    half_written = HAND_WRITTEN.replace('kind = "litellm"', 'kind = "lite')
    path.write_text(half_written)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    assert path.read_text() == half_written
    assert 'not valid TOML' in caplog.text
    assert prices.prices[('ollama', MODEL)] == ModelPrice(input=0.8, output=2.0)


def test_a_backup_that_cannot_be_made_writes_nothing_and_is_tried_again(tmp_path, monkeypatch, caplog):
    path, prices = prepared(tmp_path)
    real_copy = config_writer._copy_for_owner_only

    def failing_copy(source, candidates):
        raise PermissionError('no room for a copy')
    monkeypatch.setattr(config_writer, '_copy_for_owner_only', failing_copy)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    assert path.read_text() == HAND_WRITTEN
    assert 'backup' in caplog.text and 'no room for a copy' in caplog.text
    monkeypatch.setattr(config_writer, '_copy_for_owner_only', real_copy)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE)))
    assert len(backups(tmp_path)) == 1


def test_a_model_the_backend_does_not_share_gets_no_line(tmp_path, caplog):
    path, prices = prepared(tmp_path)
    assert take(prices, settings(('ollama', NOT_SHARED, NEW_PRICE))) is False
    assert path.read_text() == HAND_WRITTEN
    assert 'does not share' in caplog.text


def test_a_backend_that_shares_every_model_takes_a_price_for_any(tmp_path):
    path, prices = prepared(tmp_path)
    assert take(prices, settings(('litellm', PROXY_MODEL, NEW_PRICE)))
    assert load_config(path).backends['litellm'].prices == {PROXY_MODEL: NEW_PRICE}


@pytest.mark.parametrize('bad_entry', [
    'not an object',
    {'backend': 'vllm', 'model': MODEL, 'price': {'input': 1.0, 'output': 1.0}},
    {'backend': 'ollama', 'model': 7, 'price': {'input': 1.0, 'output': 1.0}},
    {'backend': 'ollama', 'model': OTHER_MODEL, 'price': {'input': True, 'output': 1.0}},
    {'backend': 'ollama', 'model': OTHER_MODEL, 'price': {'input': -1.0, 'output': 1.0}},
    {'backend': 'ollama', 'model': OTHER_MODEL, 'price': {'input': 1.0, 'output': protocol.MAX_PRICE_PER_UNIT * 2}},
    {'backend': 'ollama', 'model': OTHER_MODEL, 'price': {'input': math.nan, 'output': 1.0}},
    {'backend': 'ollama', 'model': OTHER_MODEL, 'price': {'input': math.inf, 'output': 1.0}},
    {'backend': 'ollama', 'model': OTHER_MODEL, 'price': {'input': '1.0', 'output': 1.0}},
    {'backend': 'ollama', 'model': OTHER_MODEL, 'price': {'input': 1.0}},
    {'backend': 'ollama', 'model': OTHER_MODEL, 'price': None},
], ids=['not an object', 'unknown backend', 'model not a name', 'true', 'negative', 'too high', 'nan', 'infinite',
        'text', 'no output', 'no price'])
def test_a_price_this_client_does_not_take_is_ignored_and_the_others_are_written(tmp_path, caplog, bad_entry):
    path, prices = prepared(tmp_path)
    frame = settings(('ollama', MODEL, NEW_PRICE))
    frame['models'].insert(0, bad_entry)
    assert take(prices, frame)
    assert path.read_text() == HAND_WRITTEN.replace(OLD_LINE_VALUE, NEW_LINE_VALUE)
    # the price, or the entry as a whole
    assert any(said in caplog.text
               for said in ('Ignored a price toptic.ai sent', 'Ignored what toptic.ai sent for a model'))
    assert ('ollama', OTHER_MODEL) not in prices.prices


def test_a_frame_without_a_list_of_prices_changes_nothing(tmp_path, caplog):
    path, prices = prepared(tmp_path)
    assert take(prices, {'type': protocol.FRAME_MODELS_SETTINGS, 'models': 'everything'}) is False
    assert path.read_text() == HAND_WRITTEN


def test_fields_besides_the_two_numbers_do_not_stop_a_price(tmp_path):
    path, prices = prepared(tmp_path)
    frame = settings(('ollama', MODEL, NEW_PRICE))
    frame['models'][0]['price']['cached_input'] = 0.1
    assert take(prices, frame)
    assert load_config(path).backends['ollama'].prices[MODEL] == NEW_PRICE


def test_no_more_prices_are_taken_from_a_frame_than_a_machine_can_have(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, 'MAX_SETTINGS_PER_FRAME', 1)
    path, prices = prepared(tmp_path)
    take(prices, settings(('ollama', MODEL, NEW_PRICE), ('ollama', OTHER_MODEL, NEW_PRICE)))
    assert load_config(path).backends['ollama'].prices == {MODEL: NEW_PRICE}


def test_several_prices_of_one_frame_are_one_write(tmp_path):
    path, prices = prepared(tmp_path)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE), ('ollama', OTHER_MODEL, LATER_PRICE),
                                         ('litellm', PROXY_MODEL, AWKWARD_PRICE)))
    loaded = load_config(path).backends
    assert loaded['ollama'].prices == {MODEL: NEW_PRICE, OTHER_MODEL: LATER_PRICE}
    assert loaded['litellm'].prices == {PROXY_MODEL: AWKWARD_PRICE}
    assert len(backups(tmp_path)) == 1


def test_without_a_configuration_file_nothing_can_be_written(caplog):
    config = parse_config(tomllib.loads(HAND_WRITTEN), environ={TOKEN_VARIABLE: 'x', SECRET_VARIABLE: 'y'})
    prices = ConfigSettings(config)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    assert 'not started with a configuration file' in caplog.text


def test_the_temporary_file_is_never_left_behind(tmp_path, monkeypatch):
    path, prices = prepared(tmp_path)

    def failing_replace(source, target):
        raise OSError('disk full')
    monkeypatch.setattr(os, 'replace', failing_replace)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    assert path.read_text() == HAND_WRITTEN
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']


def test_a_price_for_a_model_no_acknowledged_report_listed_is_not_taken(tmp_path):
    """Its backend was down when the client started, say: toptic.ai has not seen the line yet."""
    path, prices = prepared(tmp_path)
    frame = settings(('ollama', MODEL, NEW_PRICE), ('ollama', OTHER_MODEL, NEW_PRICE))
    assert take(prices, frame, acknowledged=frozenset({('ollama', OTHER_MODEL)}))
    assert load_config(path).backends['ollama'].prices == {MODEL: ModelPrice(input=0.8, output=2.0),
                                                           OTHER_MODEL: NEW_PRICE}


def test_a_line_changed_by_hand_to_the_price_toptic_sets_is_held_without_writing(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path, prices = prepared(tmp_path)
    hand_edited = HAND_WRITTEN.replace(OLD_LINE_VALUE, '{ input = 0.40, output = 1.40 }')
    path.write_text(hand_edited)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE)))
    assert path.read_text() == hand_edited and backups(tmp_path) == []
    assert prices.prices[('ollama', MODEL)] == NEW_PRICE
    assert 'Restart the client' not in caplog.text and 'shows already' in caplog.text


def test_the_file_keeps_its_group(tmp_path):
    path, prices = prepared(tmp_path, mode=GROUP_READABLE)
    group = path.stat().st_gid
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE)))
    assert path.stat().st_gid == group


def test_a_group_this_user_cannot_give_the_file_leaves_it_alone(tmp_path, monkeypatch, caplog):
    path, prices = prepared(tmp_path, mode=GROUP_READABLE)
    real_stat = Path.stat
    foreign_group = path.stat().st_gid + 1

    def stat_of_another_group(self, *args, **kwargs):
        status = real_stat(self, *args, **kwargs)
        if self == path:
            fields = list(status)
            fields[stat.ST_GID] = foreign_group
            return os.stat_result(fields)
        return status
    monkeypatch.setattr(Path, 'stat', stat_of_another_group)

    def refused_chown(descriptor, owner, group):
        raise PermissionError('Operation not permitted')
    monkeypatch.setattr(os, 'fchown', refused_chown)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    assert path.read_text() == HAND_WRITTEN
    assert 'Operation not permitted' in caplog.text
    assert [entry.name for entry in tmp_path.iterdir()] == ['config.toml']


def test_a_file_of_another_user_is_left_alone(tmp_path, monkeypatch, caplog):
    path, prices = prepared(tmp_path)
    monkeypatch.setattr(os, 'geteuid', lambda: path.stat().st_uid + 1)
    assert take(prices, settings(('ollama', MODEL, NEW_PRICE))) is False
    assert path.read_text() == HAND_WRITTEN
    assert 'belongs to another user' in caplog.text


def test_a_negative_zero_is_written_as_zero(tmp_path):
    path, prices = prepared(tmp_path)
    assert take(prices, settings(('ollama', MODEL, ModelPrice(input=-0.0, output=1.0))))
    written = load_config(path).backends['ollama'].prices[MODEL].input
    assert written == 0 and math.copysign(1.0, written) > 0
    assert '-0.0' not in path.read_text()
