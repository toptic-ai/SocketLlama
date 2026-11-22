"""The conversation of ``socketllama setup``, driven by scripted answers and scripted backends."""
import tomllib
from pathlib import Path

import pytest

from socketllama import protocol, setup_wizard
from socketllama.backends import (
    NOTE_BACKEND_NOT_REPORTED, NOTE_DETAILS_UNREADABLE, NOTE_DOWN, NOTE_ENTRY_IGNORED, NOTE_FAILED, NOTE_MESSAGE,
    NOTE_NAMES_REFUSED, NOTE_REPORT_TOO_LARGE, NOTE_TIMED_OUT, NOTE_TOO_MANY_MODELS, BackendUnavailable,
    DiscoveryNote,
)
from socketllama.config import KIND_LITELLM, KIND_OLLAMA, KIND_OPENAI, ConfigError, parse_config
from socketllama.config_writer import ConfigTarget
from socketllama.setup_wizard import (
    BACKUP_RETRY, BACKUP_SKIP, BACKUP_STOP, DEFAULT_BACKEND_URLS, OPENAI_EXAMPLE_URL, SIGNING_SECRET_PLACEHOLDER,
    TOKEN_PLACEHOLDER, AnswerError, Discovered, Selection, SetupDeclined, SetupWizard, parse_gateway_block,
    parse_selection, summary_lines,
)

from .fakes import BACKEND_API_KEY, INSTANCE, SIGNING_SECRET, TOKEN, ScriptedConsole, ScriptedDiscovery

GATEWAY_URL = 'wss://toptic.example/ws/socketllama/'
INSECURE_GATEWAY_URL = 'ws://127.0.0.1:8888/ws/socketllama/'
# Never contacted: the wizard asks backends through the scripted discovery only
OLLAMA_URL = 'http://ollama.invalid/v1'
OTHER_OLLAMA_URL = 'http://gpu-box.invalid/v1'
LITELLM_URL = 'http://litellm.invalid/v1'
VLLM_URL = 'http://vllm.invalid/v1'
BACKEND_URLS = {KIND_OLLAMA: OLLAMA_URL, KIND_LITELLM: LITELLM_URL}
MODELS = ['llama3.1:8b', 'qwen3:14b', 'gemma3:4b', 'mistral:7b']
MODEL_COUNT = len(MODELS)
CONFIG_PATH = Path('config.toml')
DOWN = BackendUnavailable('ConnectError: All connection attempts failed', is_down=True)
TIMED_OUT = BackendUnavailable('no answer within 15 s', timed_out=True)
HTTP_UNAUTHORIZED = 401
UNAUTHORIZED = BackendUnavailable("HTTPStatusError: Client error '401 Unauthorized' for url\nFor more information "
                                  'check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/401',
                                  answered=True, status=HTTP_UNAUTHORIZED)
NOT_THE_API = BackendUnavailable('JSONDecodeError: Expecting value: line 1 column 1 (char 0)', answered=True)
# More digits than int() converts (Python refuses beyond 4300), and than any list has entries
ENDLESS_NUMBER = '9' * 5000
# Digits Python's \d and int() take and Ollama does not (Go reads 0-9 only): 3600 and 45m in fullwidth digits
FULLWIDTH_3600 = '\N{FULLWIDTH DIGIT THREE}\N{FULLWIDTH DIGIT SIX}\N{FULLWIDTH DIGIT ZERO}\N{FULLWIDTH DIGIT ZERO}'
FULLWIDTH_45M = '\N{FULLWIDTH DIGIT FOUR}\N{FULLWIDTH DIGIT FIVE}m'
# A terminal control sequence and a right-to-left override, as a backend might send them
HOSTILE = 'evil\x9b31m\u202etxet'
# Words of the configuration file that somebody answering questions has never seen
FILE_WORDS = ('gateway.url', 'base_url', 'allow_insecure', 'api_key')
UNREACHABLE = BackendUnavailable('ConnectError: [Errno 113] No route to host')

# Answers to the menus, by what they pick
PASTE_BLOCK = '1'
ANSWER_QUESTIONS = '2'
END_OF_BLOCK = 'END'
TRY_AGAIN = '1'
OTHER_ADDRESS = '2'
SKIP_OLLAMA = '3'
KEY_AGAIN = '3'
LOOK_AGAIN = '1'
SHARE_LATER_MODELS = '2'
SKIP_EMPTY = '3'
ADD_LITELLM = '1'
ADD_OPENAI = '2'
ADD_OLLAMA = '3'
NO_MORE_BACKENDS = '4'
DEFAULT = ''
YES = 'y'
NO = 'n'

PAGE_BLOCK = f'''[gateway]
url = "{GATEWAY_URL}"
instance = "{INSTANCE}"
token = "{TOKEN}"
signing_secret = "{SIGNING_SECRET}"

[[backends]]
name = "ollama"
kind = "ollama"
base_url = "http://127.0.0.1:11434/v1"
# The models to share, as "ollama list" names them
models = ["NAME-OF-A-MODEL-FROM-OLLAMA-LIST"]'''
GATEWAY = {'url': GATEWAY_URL, 'instance': INSTANCE, 'token': TOKEN, 'signing_secret': SIGNING_SECRET}
PASTED_GATEWAY = [PASTE_BLOCK, *PAGE_BLOCK.splitlines(), END_OF_BLOCK]
# Answers to the questions about an Ollama backend's memory that leave it to Ollama: how long models
# stay, and whether they are loaded at start (the second is not asked of a backend that shares everything)
LEAVE_MEMORY = [DEFAULT, DEFAULT]
LEAVE_MEMORY_OF_ALL = [DEFAULT]
FIRST_MODEL_ONLY = ['1', *LEAVE_MEMORY, NO_MORE_BACKENDS]
OLLAMA_WITH_FIRST_MODEL = {'name': 'ollama', 'kind': KIND_OLLAMA, 'base_url': OLLAMA_URL, 'models': [MODELS[0]]}


def target(exists: bool = False, **details) -> ConfigTarget:
    return ConfigTarget(**{'named': CONFIG_PATH, 'file': CONFIG_PATH, 'is_link': False, 'exists': exists,
                           'readable_by_others': False, **details})


def run_wizard(answers, discovery=None, target_exists=False):
    """(plan, console, discovery) of a wizard run; every scripted answer must have been asked for."""
    console = ScriptedConsole(answers)
    discovery = discovery or ScriptedDiscovery({OLLAMA_URL: [MODELS]})
    plan = SetupWizard(console, discovery, BACKEND_URLS).run(target(exists=target_exists))
    assert console.unused_answers == []
    return plan, console, discovery


# ----- the selection syntax -----

@pytest.mark.parametrize('answer, positions', [
    ('1', (0,)),
    ('1,3-4', (0, 2, 3)),
    ('3-4, 1', (0, 2, 3)),
    ('2 4', (1, 3)),
    (' 1 , 2 ', (0, 1)),
    ('1 - 3', (0, 1, 2)),
    ('2,2,1-2', (0, 1)),
    ('4-4', (3,)),
    ('1,', (0,)),
    ('all', (0, 1, 2, 3)),
    ('ALL', (0, 1, 2, 3)),
])
def test_a_selection_names_positions_in_the_list(answer, positions):
    assert parse_selection(answer, MODEL_COUNT) == Selection(positions=positions)


def test_future_shares_everything_including_what_comes_later():
    selection = parse_selection(' Future ', MODEL_COUNT)
    assert selection.includes_future and selection.positions == ()


def test_none_selects_nothing():
    selection = parse_selection('none', MODEL_COUNT)
    assert selection.is_empty and not selection.includes_future


@pytest.mark.parametrize('answer, reason', [
    ('', 'numbers'),
    (',', 'numbers'),
    ('0', 'from 1 to 4'),
    ('5', 'from 1 to 4'),
    ('2-5', 'from 1 to 4'),
    ('3-1', '1-3'),
    ('one', '"one"'),
    ('1,all', '"all"'),
    ('1.5', '"1.5"'),
    ('-1', '"-1"'),
    ('1-2-3', '"1-2-3"'),
    (ENDLESS_NUMBER, 'from 1 to 4'),
    (f'1-{ENDLESS_NUMBER}', 'from 1 to 4'),
    (f'{ENDLESS_NUMBER}-2', 'from 1 to 4'),
    ('²', '"²"'),
])
def test_a_selection_that_cannot_be_used_says_why(answer, reason):
    with pytest.raises(AnswerError, match=reason):
        parse_selection(answer, MODEL_COUNT)


# ----- the pasted config block -----

def test_only_the_gateway_values_of_a_pasted_block_are_taken():
    assert parse_gateway_block(PAGE_BLOCK) == GATEWAY


def test_a_development_site_block_permits_plain_ws():
    block = PAGE_BLOCK.replace(GATEWAY_URL, INSECURE_GATEWAY_URL).replace(
        '\n\n', '\nallow_insecure = true          # this site runs without https (a development server)\n\n', 1)
    assert parse_gateway_block(block) == {**GATEWAY, 'url': INSECURE_GATEWAY_URL, 'allow_insecure': True}


def test_the_block_of_a_page_without_token_and_with_a_secret_to_renew_gives_neither():
    """The block as toptic.ai's page writes it before a token exists and when the secret has to be renewed."""
    found = parse_gateway_block(f'''[gateway]
url = "{GATEWAY_URL}"
instance = "{INSTANCE}"
token = "PASTE-YOUR-CONNECTION-TOKEN-HERE"
signing_secret = "CREATE-A-NEW-SIGNING-SECRET-BELOW"
''')
    assert found == {'url': GATEWAY_URL, 'instance': INSTANCE}


@pytest.mark.parametrize('key', ['token', 'signing_secret'])
@pytest.mark.parametrize('placeholder', [TOKEN_PLACEHOLDER, SIGNING_SECRET_PLACEHOLDER])
def test_a_placeholder_of_the_page_is_a_missing_value_wherever_it_stands(key, placeholder):
    found = parse_gateway_block(PAGE_BLOCK.replace(GATEWAY[key], placeholder))
    assert found == {name: value for name, value in GATEWAY.items() if name != key}


def test_empty_values_and_values_of_another_type_are_missing():
    found = parse_gateway_block('[gateway]\nurl = "  "\ninstance = 7\ntoken = " padded "\nallow_insecure = "yes"\n'
                                'token_env = "SOCKETLLAMA_TOKEN"')
    assert found == {'token': 'padded'}


@pytest.mark.parametrize('block, reason', [
    ('', r'\[gateway\]'),
    ('url = "wss://toptic.example/ws/socketllama/"', r'\[gateway\]'),
    ('gateway = "x"', r'\[gateway\]'),
    ('[gateway]\nurl = "unfinished', 'whole block'),
    ('[gateway]\ntoken = "a"\ntoken = "b"', 'whole block'),
])
def test_a_block_that_cannot_be_read_says_what_to_paste(block, reason):
    with pytest.raises(AnswerError, match=reason):
        parse_gateway_block(block)


# ----- step 1: the connection to toptic.ai -----

def test_a_pasted_block_and_a_selection_give_a_complete_configuration():
    plan, console, discovery = run_wizard([*PASTED_GATEWAY, '1,3-4', *LEAVE_MEMORY, NO_MORE_BACKENDS, YES])
    assert plan.data == {'gateway': GATEWAY, 'backends': [
        {'name': 'ollama', 'kind': KIND_OLLAMA, 'base_url': OLLAMA_URL, 'models': ['llama3.1:8b', 'gemma3:4b',
                                                                                 'mistral:7b']}]}
    assert parse_config(plan.data, environ={}).instance == INSTANCE
    assert [backend.base_url for backend in discovery.asked] == [OLLAMA_URL]
    assert console.secret_prompts == []


def test_answers_one_by_one_read_the_secrets_without_echo():
    plan, console, _ = run_wizard([ANSWER_QUESTIONS, GATEWAY_URL, INSTANCE, TOKEN, SIGNING_SECRET,
                                   *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY
    assert len(console.secret_prompts) == 2
    assert 'token' in console.secret_prompts[0].lower() and 'secret' in console.secret_prompts[1].lower()


def test_the_conversation_says_where_each_value_is_found():
    _, console, _ = run_wizard([ANSWER_QUESTIONS, GATEWAY_URL, INSTANCE, TOKEN, SIGNING_SECRET,
                                *FIRST_MODEL_ONLY, YES])
    for words in ('account menu', 'My Machines', 'Create connection token', 'shown only once', 'instance'):
        assert words in console.output
    assert 'open SocketLlama' not in console.output


@pytest.mark.parametrize('wrong, reason', [
    ('https://toptic.example/ws/socketllama/', 'wss://'),
    ('toptic.example', 'wss://'),
    ('wss://[::1', 'wss://'),
    ('', 'needed'),
])
def test_a_gateway_address_that_breaks_the_rules_is_asked_again_with_the_reason(wrong, reason):
    plan, console, _ = run_wizard([ANSWER_QUESTIONS, wrong, GATEWAY_URL, INSTANCE, TOKEN, SIGNING_SECRET,
                                   *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway']['url'] == GATEWAY_URL
    assert reason in console.output


def test_an_empty_or_unprintable_answer_is_asked_again():
    plan, console, _ = run_wizard([ANSWER_QUESTIONS, GATEWAY_URL, '', INSTANCE, 'tok\x1b[200~en', TOKEN, '',
                                   SIGNING_SECRET, *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY
    assert len(console.secret_prompts) == 4


def test_the_placeholder_typed_as_a_token_is_asked_again():
    plan, console, _ = run_wizard([ANSWER_QUESTIONS, GATEWAY_URL, INSTANCE, TOKEN_PLACEHOLDER, TOKEN,
                                   SIGNING_SECRET, *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway']['token'] == TOKEN
    assert 'placeholder' in console.output


def test_a_block_without_a_token_asks_for_the_token_only():
    block = PAGE_BLOCK.replace(TOKEN, TOKEN_PLACEHOLDER)
    plan, console, _ = run_wizard([PASTE_BLOCK, *block.splitlines(), END_OF_BLOCK, TOKEN, *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY
    assert len(console.secret_prompts) == 1 and 'token' in console.secret_prompts[0].lower()


def test_a_block_whose_signing_secret_has_to_be_renewed_asks_for_it():
    block = PAGE_BLOCK.replace(SIGNING_SECRET, SIGNING_SECRET_PLACEHOLDER)
    plan, console, _ = run_wizard([PASTE_BLOCK, *block.splitlines(), END_OF_BLOCK, SIGNING_SECRET,
                                   *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY
    assert 'new signing secret' in console.output


def test_a_block_with_a_value_that_breaks_the_rules_asks_for_that_value():
    block = PAGE_BLOCK.replace(GATEWAY_URL, 'https://toptic.example/')
    plan, console, _ = run_wizard([PASTE_BLOCK, *block.splitlines(), END_OF_BLOCK, GATEWAY_URL,
                                   *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY
    assert 'wss://' in console.output


def test_a_broken_block_can_be_pasted_again():
    plan, console, _ = run_wizard([PASTE_BLOCK, 'not the block', END_OF_BLOCK, YES, *PAGE_BLOCK.splitlines(),
                                   END_OF_BLOCK.lower(), *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY


def test_after_a_broken_block_the_values_can_be_answered_one_by_one():
    plan, _, _ = run_wizard([PASTE_BLOCK, 'not the block', END_OF_BLOCK, NO, GATEWAY_URL, INSTANCE, TOKEN,
                             SIGNING_SECRET, *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY


def test_a_plain_ws_address_needs_a_clear_yes():
    plan, console, _ = run_wizard([ANSWER_QUESTIONS, INSECURE_GATEWAY_URL, DEFAULT, GATEWAY_URL, INSTANCE, TOKEN,
                                   SIGNING_SECRET, *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY
    assert 'not encrypted' in console.output
    plan, _, _ = run_wizard([ANSWER_QUESTIONS, INSECURE_GATEWAY_URL, YES, INSTANCE, TOKEN, SIGNING_SECRET,
                             *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == {'url': INSECURE_GATEWAY_URL, 'instance': INSTANCE, 'token': TOKEN,
                                    'signing_secret': SIGNING_SECRET, 'allow_insecure': True}
    assert parse_config(plan.data, environ={}).gateway_url == INSECURE_GATEWAY_URL


def test_a_development_site_block_is_not_asked_about_plain_ws():
    block = PAGE_BLOCK.replace(GATEWAY_URL, INSECURE_GATEWAY_URL).replace('\n\n', '\nallow_insecure = true\n\n', 1)
    plan, _, _ = run_wizard([PASTE_BLOCK, *block.splitlines(), END_OF_BLOCK, *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway']['allow_insecure'] is True


# ----- step 2: backends and their models -----

def test_the_models_ollama_reports_are_listed_with_numbers():
    _, console, _ = run_wizard([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES])
    for number, model in enumerate(MODELS, start=1):
        assert f'{number}) {model}' in console.output
    assert OLLAMA_URL in console.output


def test_all_shares_the_models_listed_now():
    plan, _, _ = run_wizard([*PASTED_GATEWAY, 'all', *LEAVE_MEMORY, NO_MORE_BACKENDS, YES])
    assert plan.data['backends'][0]['models'] == MODELS


def test_sharing_models_pulled_later_is_a_separate_confirmed_choice():
    plan, console, _ = run_wizard([*PASTED_GATEWAY, 'future', YES, *LEAVE_MEMORY_OF_ALL, NO_MORE_BACKENDS, YES])
    assert plan.data['backends'][0]['models'] == []
    assert parse_config(plan.data, environ={}).backends['ollama'].shares_all_models
    assert 'pull later' in console.output


def test_declining_to_share_later_models_asks_for_the_selection_again():
    plan, _, _ = run_wizard([*PASTED_GATEWAY, 'future', DEFAULT, '2', *LEAVE_MEMORY, NO_MORE_BACKENDS, YES])
    assert plan.data['backends'][0]['models'] == [MODELS[1]]


def test_a_selection_that_cannot_be_used_is_asked_again_with_the_reason():
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '9', 'llama', '2', *LEAVE_MEMORY, NO_MORE_BACKENDS, YES])
    assert plan.data['backends'][0]['models'] == [MODELS[1]]
    assert 'from 1 to 4' in console.output and '"llama"' in console.output


def test_an_ollama_that_does_not_answer_is_explained_and_can_be_tried_again():
    discovery = ScriptedDiscovery({OLLAMA_URL: [DOWN, MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, TRY_AGAIN, *FIRST_MODEL_ONLY, YES], discovery)
    assert plan.data['backends'] == [OLLAMA_WITH_FIRST_MODEL]
    assert f'Nothing answers at {OLLAMA_URL}' in console.output
    assert 'ollama serve' in console.output
    assert len(discovery.asked) == 2


def test_an_ollama_at_another_address_can_be_entered():
    discovery = ScriptedDiscovery({OLLAMA_URL: [DOWN], OTHER_OLLAMA_URL: [MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, OTHER_ADDRESS, 'gpu-box:11434', f'{OTHER_OLLAMA_URL}/',
                                   *FIRST_MODEL_ONLY, YES], discovery)
    assert plan.data['backends'] == [{**OLLAMA_WITH_FIRST_MODEL, 'base_url': OTHER_OLLAMA_URL}]
    assert 'http://' in console.output


def test_a_backend_that_answers_late_or_wrongly_is_explained_with_the_reason():
    discovery = ScriptedDiscovery({OLLAMA_URL: [TIMED_OUT, UNAUTHORIZED, MODELS]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, TRY_AGAIN, TRY_AGAIN, *FIRST_MODEL_ONLY, YES], discovery)
    assert f'{OLLAMA_URL} did not answer within 15 s.' in console.output
    assert 'no answer within' not in console.output and 'in time' not in console.output
    assert '401 Unauthorized' in console.output


def test_a_backend_that_cannot_be_reached_is_not_said_to_have_answered():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [UNREACHABLE, UNAUTHORIZED, MODELS]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                ADD_OPENAI, 'vllm', VLLM_URL, DEFAULT, TRY_AGAIN, TRY_AGAIN,
                                '1', NO_MORE_BACKENDS, YES], discovery)
    assert f'{VLLM_URL} could not be reached (ConnectError: [Errno 113] No route to host)' in console.output
    assert 'answered' not in console.output.split('could not be reached')[1].split('What do you want')[0]
    assert console.output.count('refused the request') == 1


def test_only_ollama_is_known_to_list_chat_models_only():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [MODELS]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                ADD_OPENAI, 'vllm', VLLM_URL, DEFAULT, '1', NO_MORE_BACKENDS,
                                YES], discovery)
    assert f'It has {MODEL_COUNT} chat model(s)' in console.output
    assert f'It lists {MODEL_COUNT} model(s)' in console.output


def test_a_skipped_ollama_leaves_no_way_to_finish_without_a_backend():
    discovery = ScriptedDiscovery({OLLAMA_URL: [DOWN], VLLM_URL: [MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, SKIP_OLLAMA, NO_MORE_BACKENDS, ADD_OPENAI, DEFAULT, VLLM_URL,
                                   DEFAULT, '2', NO_MORE_BACKENDS, YES], discovery)
    assert 'from 1 to 3' in console.output  # continuing is not among the choices yet
    assert plan.data['backends'] == [{'name': 'openai', 'kind': KIND_OPENAI, 'base_url': VLLM_URL,
                                      'models': [MODELS[1]]}]
    assert 'at least one' in console.output.lower()


def test_without_any_backend_the_input_ending_ends_setup():
    discovery = ScriptedDiscovery({OLLAMA_URL: [DOWN]})
    with pytest.raises(EOFError):
        run_wizard([*PASTED_GATEWAY, SKIP_OLLAMA], discovery)


def test_none_skips_a_backend():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [MODELS]})
    plan, _, _ = run_wizard([*PASTED_GATEWAY, 'none', ADD_OPENAI, 'vllm', VLLM_URL, DEFAULT, 'all',
                             NO_MORE_BACKENDS, YES], discovery)
    assert [backend['name'] for backend in plan.data['backends']] == ['vllm']


def test_an_ollama_without_models_can_be_asked_again_after_pulling_one():
    discovery = ScriptedDiscovery({OLLAMA_URL: [[], MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, LOOK_AGAIN, *FIRST_MODEL_ONLY, YES], discovery)
    assert plan.data['backends'] == [OLLAMA_WITH_FIRST_MODEL]
    assert 'ollama pull' in console.output


def test_an_ollama_without_models_can_share_what_is_pulled_later():
    discovery = ScriptedDiscovery({OLLAMA_URL: [[]]})
    plan, _, _ = run_wizard([*PASTED_GATEWAY, SHARE_LATER_MODELS, YES, *LEAVE_MEMORY_OF_ALL, NO_MORE_BACKENDS, YES],
                            discovery)
    assert plan.data['backends'] == [{**OLLAMA_WITH_FIRST_MODEL, 'models': []}]


def test_an_ollama_without_models_can_be_skipped():
    discovery = ScriptedDiscovery({OLLAMA_URL: [[]], LITELLM_URL: [['gpt-proxy']]})
    plan, _, _ = run_wizard([*PASTED_GATEWAY, SKIP_EMPTY, ADD_LITELLM, DEFAULT, DEFAULT, DEFAULT, 'all',
                             NO_MORE_BACKENDS, YES], discovery)
    assert plan.data['backends'] == [{'name': 'litellm', 'kind': KIND_LITELLM, 'base_url': LITELLM_URL,
                                      'models': ['gpt-proxy']}]


def test_a_litellm_proxy_is_added_with_its_key_read_without_echo():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], LITELLM_URL: [['gpt-proxy', 'claude-proxy']]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                   ADD_LITELLM, 'proxy', LITELLM_URL, BACKEND_API_KEY, '2',
                                   NO_MORE_BACKENDS, YES], discovery)
    assert plan.data['backends'] == [
        OLLAMA_WITH_FIRST_MODEL,
        {'name': 'proxy', 'kind': KIND_LITELLM, 'base_url': LITELLM_URL, 'api_key': BACKEND_API_KEY,
         'models': ['claude-proxy']}]
    assert discovery.asked[-1].api_key == BACKEND_API_KEY
    assert len(console.secret_prompts) == 1 and 'API key' in console.secret_prompts[0]
    assert parse_config(plan.data, environ={}).backends['proxy'].api_key == BACKEND_API_KEY


@pytest.mark.parametrize('wrong_name, reason', [
    ('ollama', 'already'),
    ('a/b', 'cannot contain "/"'),
    ('n' * (protocol.BACKEND_NAME_MAX_LENGTH + 1), 'characters'),
])
def test_a_backend_name_that_breaks_the_rules_is_asked_again(wrong_name, reason):
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                   ADD_OPENAI, wrong_name, 'vllm', VLLM_URL, DEFAULT, '1',
                                   NO_MORE_BACKENDS, YES], discovery)
    assert plan.data['backends'][1]['name'] == 'vllm'
    assert reason in console.output


@pytest.mark.parametrize('wrong_address, reason', [
    ('vllm.invalid:8000', 'http'),
    ('http://[fe80::1/v1', 'http'),
    ('http://vllm.invalid:80x/v1', 'http'),
    ('http://user:password@vllm.invalid/v1', 'user name and password'),
    ('', 'needed'),
])
def test_a_backend_address_that_breaks_the_rules_is_asked_again(wrong_address, reason):
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                   ADD_OPENAI, 'vllm', wrong_address, VLLM_URL, DEFAULT, '1',
                                   NO_MORE_BACKENDS, YES], discovery)
    assert plan.data['backends'][1]['base_url'] == VLLM_URL
    assert reason in console.output


def test_a_second_backend_of_a_kind_gets_another_default_name():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], OTHER_OLLAMA_URL: [MODELS]})
    plan, _, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY, ADD_OLLAMA, DEFAULT, OTHER_OLLAMA_URL, DEFAULT, '1',
                             *LEAVE_MEMORY, NO_MORE_BACKENDS, YES], discovery)
    assert [backend['name'] for backend in plan.data['backends']] == ['ollama', 'ollama-2']


def test_a_refused_key_can_be_entered_again():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], LITELLM_URL: [UNAUTHORIZED, ['gpt-proxy']]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                   ADD_LITELLM, DEFAULT, DEFAULT, 'sk-wrong', KEY_AGAIN,
                                   BACKEND_API_KEY, '1', NO_MORE_BACKENDS, YES], discovery)
    assert plan.data['backends'][1]['api_key'] == BACKEND_API_KEY
    assert [backend.api_key for backend in discovery.asked[1:]] == ['sk-wrong', BACKEND_API_KEY]
    assert 'sk-wrong' not in console.output


def test_no_more_backends_are_offered_than_toptic_accepts(monkeypatch):
    monkeypatch.setattr(protocol, 'MAX_BACKENDS_PER_REPORT', 2)
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                   ADD_OPENAI, 'vllm', VLLM_URL, DEFAULT, '1', YES], discovery)
    assert len(plan.data['backends']) == 2
    assert 'the most toptic.ai accepts' in console.output


def test_a_menu_answer_that_is_not_one_of_the_choices_is_asked_again():
    plan, console, _ = run_wizard(['7', 'paste', *PASTED_GATEWAY, *FIRST_MODEL_ONLY, 'maybe', YES])
    assert plan.data['gateway'] == GATEWAY
    assert 'from 1 to 2' in console.output and 'y or n' in console.output


# ----- step 3: summary and confirmation -----

def test_the_summary_shows_what_will_be_written_but_no_secret():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], LITELLM_URL: [['gpt-proxy']]})
    _, console, _ = run_wizard([ANSWER_QUESTIONS, GATEWAY_URL, INSTANCE, TOKEN, SIGNING_SECRET, '1,2', *LEAVE_MEMORY,
                                ADD_LITELLM,
                                DEFAULT, DEFAULT, BACKEND_API_KEY, 'future', YES, NO_MORE_BACKENDS, YES], discovery)
    for shown in (str(CONFIG_PATH), GATEWAY_URL, INSTANCE, OLLAMA_URL, 'llama3.1:8b, qwen3:14b', LITELLM_URL):
        assert shown in console.output
    for secret in (TOKEN, SIGNING_SECRET, BACKEND_API_KEY):
        assert secret not in console.output


def test_summary_lines_mask_every_secret_and_say_what_each_backend_shares():
    data = {'gateway': {**GATEWAY, 'allow_insecure': True},
            'backends': [OLLAMA_WITH_FIRST_MODEL,
                         {'name': 'proxy', 'kind': KIND_LITELLM, 'base_url': LITELLM_URL, 'api_key': BACKEND_API_KEY,
                          'models': []}]}
    text = '\n'.join(summary_lines(data))
    for secret in (TOKEN, SIGNING_SECRET, BACKEND_API_KEY):
        assert secret not in text
    assert f'{len(TOKEN)} characters' in text
    assert 'not encrypted' in text
    assert 'every model' in text and 'later' in text
    assert MODELS[0] in text


def test_declining_the_summary_writes_nothing():
    with pytest.raises(SetupDeclined):
        run_wizard([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, NO])


def test_an_existing_file_is_announced_and_only_replaced_after_a_clear_yes():
    console = ScriptedConsole([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, DEFAULT])
    with pytest.raises(SetupDeclined):
        SetupWizard(console, ScriptedDiscovery({OLLAMA_URL: [MODELS]}), BACKEND_URLS).run(target(exists=True))
    assert console.output.count('config.toml already exists') == 2  # at the start and before the question


@pytest.mark.parametrize('answer, keep_backup', [(DEFAULT, True), (YES, True), (NO, False)])
def test_the_old_file_is_kept_as_a_backup_unless_declined(answer, keep_backup):
    plan, console, _ = run_wizard([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES, answer], target_exists=True)
    assert plan.keep_backup is keep_backup
    assert 'backup' in console.output


def test_a_new_file_needs_no_backup_question():
    plan, console, _ = run_wizard([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, DEFAULT])
    assert 'backup' not in console.output
    assert 'already exists' not in console.output


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, EOFError])
def test_an_interruption_ends_the_conversation(interruption):
    with pytest.raises(interruption):
        run_wizard([*PASTED_GATEWAY, interruption()])


# ----- what the wizard assumes about the rest -----

def test_the_default_addresses_are_the_ones_of_the_example_configuration():
    example = tomllib.loads(Path('config.example.toml').read_text())
    by_kind = {backend['kind']: backend['base_url'] for backend in example['backends']}
    assert DEFAULT_BACKEND_URLS == {KIND_OLLAMA: by_kind[KIND_OLLAMA], KIND_LITELLM: by_kind[KIND_LITELLM]}


def test_the_end_marker_typed_right_behind_a_paste_without_a_final_line_break_still_ends_the_block():
    lines = PAGE_BLOCK.splitlines()
    plan, _, _ = run_wizard([PASTE_BLOCK, *lines[:-1], f'{lines[-1]}{END_OF_BLOCK}', *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY


# ----- answers and backends that are out of the ordinary -----

def test_a_number_too_long_to_be_one_is_an_ordinary_wrong_answer():
    plan, console, _ = run_wizard([ENDLESS_NUMBER, *PASTED_GATEWAY, ENDLESS_NUMBER, f'1-{ENDLESS_NUMBER}', '2',
                                   *LEAVE_MEMORY, ENDLESS_NUMBER, NO_MORE_BACKENDS, YES])
    assert plan.data['backends'][0]['models'] == [MODELS[1]]
    assert 'from 1 to 2' in console.output and 'from 1 to 4' in console.output
    assert ENDLESS_NUMBER not in console.output.replace(f'Your choice [1]: {ENDLESS_NUMBER}', '')


def test_the_end_of_input_ends_a_pasted_block_and_setup_goes_on():
    """Ctrl-D, the habit for "that was all"; the terminal goes on answering afterwards."""
    plan, console, _ = run_wizard([PASTE_BLOCK, *PAGE_BLOCK.splitlines(), EOFError(), *FIRST_MODEL_ONLY, YES])
    assert plan.data['gateway'] == GATEWAY
    assert 'Ctrl-D' in console.output


def test_answers_that_break_a_rule_are_explained_without_the_words_of_the_file():
    discovery = ScriptedDiscovery({OLLAMA_URL: [DOWN], LITELLM_URL: [MODELS], VLLM_URL: [MODELS]})
    _, console, _ = run_wizard([
        ANSWER_QUESTIONS, 'https://toptic.example/', 'toptic.example', 'wss://[::1', INSECURE_GATEWAY_URL, NO,
        GATEWAY_URL, INSTANCE, TOKEN, SIGNING_SECRET,
        OTHER_ADDRESS, 'ollama.invalid:11434', 'http://me:pw@ollama.invalid/v1', OLLAMA_URL, SKIP_OLLAMA,
        ADD_LITELLM, 'a/b', 'n' * (protocol.BACKEND_NAME_MAX_LENGTH + 1), DEFAULT, 'litellm.invalid', DEFAULT,
        DEFAULT, '1',
        ADD_OPENAI, DEFAULT, 'ftp://vllm.invalid/v1', VLLM_URL, DEFAULT, '1', NO_MORE_BACKENDS, YES], discovery)
    for word in FILE_WORDS:
        assert word not in console.output
    assert 'A gateway address starts with wss://' in console.output
    assert 'That is not a gateway address' in console.output
    assert f'at most {protocol.BACKEND_NAME_MAX_LENGTH} characters' in console.output


def test_the_example_address_is_the_one_of_the_kind_asked_for():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], LITELLM_URL: [MODELS], VLLM_URL: [MODELS]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                ADD_LITELLM, DEFAULT, 'nonsense', DEFAULT, DEFAULT, '1',
                                ADD_OPENAI, DEFAULT, 'nonsense', VLLM_URL, DEFAULT, '1', NO_MORE_BACKENDS, YES],
                               discovery)
    assert f'like {DEFAULT_BACKEND_URLS[KIND_LITELLM]}.' in console.output
    assert f'like {OPENAI_EXAMPLE_URL}.' in console.output
    assert f'like {DEFAULT_BACKEND_URLS[KIND_OLLAMA]}' not in console.output


def test_the_name_prompt_states_the_rule_that_is_applied():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                   ADD_OPENAI, 'My GPU box (vLLM)', VLLM_URL, DEFAULT, '1',
                                   NO_MORE_BACKENDS, YES], discovery)
    assert plan.data['backends'][1]['name'] == 'My GPU box (vLLM)'
    assert f'up to {protocol.BACKEND_NAME_MAX_LENGTH} characters' in console.output and 'without "/"' in console.output
    assert 'letters, digits' not in console.output


def test_the_number_of_values_asked_for_is_the_number_taken_from_the_block():
    _, console, _ = run_wizard([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES])
    assert f'needs {len(GATEWAY)} values' in console.output


@pytest.mark.parametrize('status, sentence', [
    (401, 'refused the request (HTTP 401 Unauthorized): the API key is missing or wrong'),
    (403, 'refused the request (HTTP 403 Forbidden): the API key is missing, wrong, or may not list the models'),
    (404, 'does not have what was asked for (HTTP 404 Not Found)'),
    (302, 'sends requests on to another address (HTTP 302 Found)'),
    (503, 'reported a problem of its own (HTTP 503 Service Unavailable)'),
    (418, "refused the request (HTTP 418 I'm a Teapot)"),
    (429, 'is limiting requests (HTTP 429 Too Many Requests): try again in a moment'),
    (599, 'reported a problem of its own (HTTP 599)'),
])
def test_a_refusal_is_told_by_its_status_in_a_plain_sentence(status, sentence):
    refused = BackendUnavailable(f"HTTPStatusError: Client error '{status} X' for url 'http://x'\nFor more "
                                 f'information check: https://developer.mozilla.org/', answered=True, status=status)
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [refused, MODELS]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY,
                                ADD_OPENAI, DEFAULT, VLLM_URL, 'sk-wrong', TRY_AGAIN, '1',
                                NO_MORE_BACKENDS, YES], discovery)
    assert f'{VLLM_URL} {sentence}' in console.output
    assert 'developer.mozilla.org' not in console.output and 'HTTPStatusError' not in console.output


def test_an_answer_that_is_not_the_api_keeps_its_reason():
    discovery = ScriptedDiscovery({OLLAMA_URL: [NOT_THE_API, MODELS]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, TRY_AGAIN, *FIRST_MODEL_ONLY, YES], discovery)
    assert f'{OLLAMA_URL} answered, but not with its list of models (JSONDecodeError' in console.output


def test_an_ollama_that_wants_a_key_can_be_given_one_on_the_spot():
    discovery = ScriptedDiscovery({OLLAMA_URL: [UNAUTHORIZED, MODELS]})
    plan, console, _ = run_wizard([*PASTED_GATEWAY, KEY_AGAIN, BACKEND_API_KEY, *FIRST_MODEL_ONLY, YES], discovery)
    assert plan.data['backends'] == [{'name': 'ollama', 'kind': KIND_OLLAMA, 'base_url': OLLAMA_URL,
                                      'api_key': BACKEND_API_KEY, 'models': [MODELS[0]]}]
    assert discovery.asked[-1].api_key == BACKEND_API_KEY
    assert '3) Enter an API key' in console.output
    assert 'Add an Ollama at another address' not in console.output.split('What do you')[0]
    assert len(console.secret_prompts) == 1


def test_an_ollama_that_merely_does_not_answer_is_not_offered_a_key():
    discovery = ScriptedDiscovery({OLLAMA_URL: [DOWN, MODELS]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, TRY_AGAIN, *FIRST_MODEL_ONLY, YES], discovery)
    assert 'API key' not in console.output


NOTES_AND_SENTENCES = [
    (DiscoveryNote(NOTE_NAMES_REFUSED, 'ollama', names=('tab\there', HOSTILE)),
     '2 model(s) of ollama are left out: toptic.ai does not accept their names'),
    (DiscoveryNote(NOTE_TOO_MANY_MODELS, 'ollama', names=('m201',)),
     f'ollama has more models than the {protocol.MAX_MODELS_PER_BACKEND} toptic.ai accepts from one backend'),
    (DiscoveryNote(NOTE_REPORT_TOO_LARGE, 'ollama', names=('m9',)), 'larger than toptic.ai accepts'),
    (DiscoveryNote(NOTE_BACKEND_NOT_REPORTED, 'ninth', names=('m1',)),
     f'ninth is left out: toptic.ai accepts {protocol.MAX_BACKENDS_PER_REPORT} backends'),
    (DiscoveryNote(NOTE_DETAILS_UNREADABLE, 'ollama', status=403),
     'The API key may not read the model details of ollama (HTTP 403 Forbidden)'),
    (DiscoveryNote(NOTE_DETAILS_UNREADABLE, 'ollama', status=401),
     'The API key may not read the model details of ollama (HTTP 401 Unauthorized)'),
    (DiscoveryNote(NOTE_DETAILS_UNREADABLE, 'ollama', status=404),
     'The model details of ollama could not be read (HTTP 404 Not Found)'),
    (DiscoveryNote('a kind of a later version', 'ollama', text='Backend ollama: something new happened'),
     'Backend ollama: something new happened'),
    (DiscoveryNote(NOTE_ENTRY_IGNORED, 'ollama', names=(HOSTILE,)), 'ollama lists a model in a form that cannot'),
    (DiscoveryNote(NOTE_DOWN, 'ollama', problem=DOWN),
     'Backend ollama does not answer: nothing listens at its address.'),
    (DiscoveryNote(NOTE_TIMED_OUT, 'ollama', problem=TIMED_OUT), 'Backend ollama did not answer within 15 s.'),
    (DiscoveryNote(NOTE_FAILED, 'ollama', problem=UNAUTHORIZED), 'backend ollama refused the request (HTTP 401'),
    (DiscoveryNote(NOTE_MESSAGE, '', text='Something nobody foresaw'), 'Something nobody foresaw'),
]


@pytest.mark.parametrize('note, sentence', NOTES_AND_SENTENCES)
def test_what_discovery_notes_is_said_in_the_words_of_setup(note, sentence):
    discovery = ScriptedDiscovery({OLLAMA_URL: [Discovered(names=MODELS, notes=[note])]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES], discovery)
    assert sentence[0].upper() + sentence[1:] in console.output or sentence in console.output
    assert all(line.isprintable() for line in console.shown)
    assert 'WARNING' not in console.output and 'INFO' not in console.output


def test_a_long_list_of_names_left_out_is_shown_in_part_and_counted():
    names = tuple(f'model-{number}' for number in range(25))
    discovery = ScriptedDiscovery({OLLAMA_URL: [Discovered(names=MODELS, notes=[
        DiscoveryNote(NOTE_TOO_MANY_MODELS, 'ollama', names=names)])]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES], discovery)
    assert 'model-9 and 15 more' in console.output and 'model-10' not in console.output


def test_notes_can_be_said_after_the_conversation_too():
    console = ScriptedConsole([])
    SetupWizard(console, ScriptedDiscovery({}), BACKEND_URLS).say_notes(
        [note for note, _ in NOTES_AND_SENTENCES])
    assert len([line for line in console.shown if line]) >= len(NOTES_AND_SENTENCES)


def test_text_of_a_backend_is_escaped_wherever_it_is_shown():
    hostile_problem = BackendUnavailable(f'RemoteProtocolError: {HOSTILE}')
    discovery = ScriptedDiscovery({OLLAMA_URL: [hostile_problem, [HOSTILE, *MODELS]]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, TRY_AGAIN, '2', *LEAVE_MEMORY, NO_MORE_BACKENDS, YES], discovery)
    assert all(line.isprintable() for line in console.shown)
    assert 'evil\\x9b31m\\u202etxet' in console.output


# ----- the place the file goes to -----

def test_a_link_is_named_with_the_file_it_leads_to_in_the_summary_and_the_question():
    console = ScriptedConsole([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES, YES])
    linked = target(exists=True, is_link=True, file=Path('/elsewhere/real.toml'))
    plan = SetupWizard(console, ScriptedDiscovery({OLLAMA_URL: [MODELS]}), BACKEND_URLS).run(linked)
    assert plan.keep_backup
    assert console.output.count('config.toml is a link to /elsewhere/real.toml; /elsewhere/real.toml will be '
                                'replaced') == 2


def test_a_link_to_a_file_that_is_not_there_yet_says_what_will_be_written():
    console = ScriptedConsole([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES])
    linked = target(is_link=True, file=Path('/elsewhere/real.toml'))
    SetupWizard(console, ScriptedDiscovery({OLLAMA_URL: [MODELS]}), BACKEND_URLS).run(linked)
    assert 'config.toml is a link to /elsewhere/real.toml; /elsewhere/real.toml will be written' in console.output
    assert 'already exists' not in console.output


def test_an_old_file_others_could_read_is_a_reason_to_renew_the_secrets():
    console = ScriptedConsole([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES, YES])
    SetupWizard(console, ScriptedDiscovery({OLLAMA_URL: [MODELS]}), BACKEND_URLS).run(
        target(exists=True, readable_by_others=True))
    before_the_questions, the_end = console.output.split('Step 1 of 3')[0], console.output.split('Step 3 of 3')[1]
    for words in ('Other users of this computer can read', 'may have been read', 'before you go on',
                  'Profile → API tokens', 'new signing secret', "machine's page"):
        assert words in before_the_questions
    assert 'may have been read' not in the_end and 'API tokens' not in the_end
    assert 'only you can read the backup' in the_end.lower()


def test_an_old_file_only_its_owner_could_read_needs_no_such_warning():
    _, console, _ = run_wizard([*PASTED_GATEWAY, *FIRST_MODEL_ONLY, YES, YES], target_exists=True)
    assert 'may have been read' not in console.output


@pytest.mark.parametrize('answer, action', [('1', BACKUP_RETRY), ('2', BACKUP_SKIP), ('3', BACKUP_STOP)])
def test_a_failed_backup_is_named_and_leaves_the_choice(answer, action):
    console = ScriptedConsole([answer])
    wizard = SetupWizard(console, ScriptedDiscovery({}), BACKEND_URLS)
    assert wizard.backup_failed(target(exists=True), '[Errno 13] Permission denied') == action
    assert 'backup of config.toml could not be made ([Errno 13] Permission denied)' in console.output
    assert 'Nothing was changed' in console.output and 'answers' in console.output


@pytest.mark.parametrize('wrong, sentence', [
    ('http://vllm.invalid:80x/v1', 'The port of an address is a whole number from 0 to 65535'),
    ('http://vllm.invalid:65536/v1', 'The port of an address is a whole number from 0 to 65535'),
    ('http://[fe80::1/v1', 'An IPv6 address in an address stands in brackets, like [::1]'),
])
def test_an_address_that_starts_right_is_told_what_is_wrong_with_it(wrong, sentence):
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], VLLM_URL: [MODELS]})
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', *LEAVE_MEMORY, ADD_OPENAI, 'vllm', wrong, VLLM_URL, DEFAULT, '1',
                                NO_MORE_BACKENDS, YES], discovery)
    assert sentence in console.output
    assert 'starts with http' not in console.output


def test_a_gateway_address_with_a_bad_port_is_told_so():
    _, console, _ = run_wizard([ANSWER_QUESTIONS, 'wss://toptic.example:44x/ws/', 'wss://[::1', GATEWAY_URL, INSTANCE,
                                TOKEN, SIGNING_SECRET, *FIRST_MODEL_ONLY, YES])
    assert 'The port of an address is a whole number from 0 to 65535' in console.output
    assert 'stands in brackets, like [::1]' in console.output
    assert 'That is not a gateway address' not in console.output


def test_every_rule_has_its_sentence_whoever_asks():
    """No rule needs something only some callers pass."""
    for reason in setup_wizard.RULES_IN_WORDS:
        sentence = setup_wizard.rule_in_words(ConfigError('for the file', reason=reason), example='an-example')
        assert sentence and '{' not in sentence
    assert setup_wizard.rule_in_words(ConfigError('a rule of a later version'), example='x') == (
        'a rule of a later version.')


def test_a_note_is_said_once_however_often_discovery_makes_it():
    note = DiscoveryNote(NOTE_DETAILS_UNREADABLE, 'litellm', status=403, text='Backend litellm: not readable')
    console = ScriptedConsole([])
    wizard = SetupWizard(console, ScriptedDiscovery({}), BACKEND_URLS)
    wizard.say_notes([note])
    wizard.say_notes([DiscoveryNote(NOTE_DETAILS_UNREADABLE, 'litellm', status=403,
                                    text='Backend litellm: not readable'),
                      DiscoveryNote(NOTE_DETAILS_UNREADABLE, 'other', status=403)])
    assert console.output.count('model details of litellm') == 1
    assert console.output.count('model details of other') == 1


# ----- the memory of an Ollama backend -----

KEEP_QUESTION = "Keep models in memory longer than Ollama's 5 minutes?"
LOAD_QUESTION = 'Load the chosen models when the client starts?'
THIRTY_MINUTES = '1'
TWO_HOURS = '2'
UNTIL_OLLAMA_STOPS = '3'
ANOTHER_TIME = '4'


def ollama_entry(plan) -> dict:
    return plan.data['backends'][0]


def test_both_questions_are_asked_of_an_ollama_backend_and_no_is_the_default():
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', DEFAULT, DEFAULT, NO_MORE_BACKENDS, YES])
    assert ollama_entry(plan) == OLLAMA_WITH_FIRST_MODEL  # nothing about memory: two noes leave it to Ollama
    assert f'{KEEP_QUESTION} [y/N]: ' in console.shown and f'{LOAD_QUESTION} [y/N]: ' in console.shown


def test_the_questions_say_what_is_decided():
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', DEFAULT, DEFAULT, NO_MORE_BACKENDS, YES])
    keeping, loading = console.output.split(KEEP_QUESTION)[0].split('Models to share: ')[1], (
        console.output.split(KEEP_QUESTION)[1].split(LOAD_QUESTION)[0])
    for words in ('5 minutes', 'waits', 'RAM', 'VRAM', 'after requests through toptic.ai',
                  'one copy of each model', 'a model you use yourself', 'how much memory stays occupied'):
        assert words in keeping
    assert 'not affected' not in keeping
    for words in ('first prompt', 'right away', 'runs in the background', 'once the client is connected',
                  'a prompt that arrives meanwhile waits for its model'):
        assert words in loading
    assert 'takes as long as' not in loading


@pytest.mark.parametrize('choice, keep_loaded', [(THIRTY_MINUTES, '30m'), (TWO_HOURS, '2h'), (UNTIL_OLLAMA_STOPS, -1)])
def test_a_time_can_be_picked_from_named_choices(choice, keep_loaded):
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', YES, choice, DEFAULT, NO_MORE_BACKENDS, YES])
    assert ollama_entry(plan) == {**OLLAMA_WITH_FIRST_MODEL, 'keep_loaded': keep_loaded}
    assert parse_config(plan.data, environ={}).backends['ollama'].keep_loaded == keep_loaded
    for choice_text in ('1) 30 minutes', '2) 2 hours', '3) Until Ollama stops', '4) Another time'):
        assert choice_text in console.output


@pytest.mark.parametrize('typed, keep_loaded', [
    ('45m', '45m'), ('1h30m', '1h30m'), ('3600', 3600), ('900.5', 900.5), ('-1', -1), ('  8h ', '8h'),
    ('301', 301), ('5m1s', '5m1s'), ('9223372036', 9223372036), ('2562047h', '2562047h'), ('-1m', '-1m'),
])
def test_another_time_is_typed_the_way_the_configuration_takes_it(typed, keep_loaded):
    plan, _, _ = run_wizard([*PASTED_GATEWAY, '1', YES, ANOTHER_TIME, typed, DEFAULT, NO_MORE_BACKENDS, YES])
    kept = ollama_entry(plan)['keep_loaded']
    assert kept == keep_loaded and type(kept) is type(keep_loaded)
    assert parse_config(plan.data, environ={}).backends['ollama'].keep_loaded == keep_loaded


@pytest.mark.parametrize('typed', ['tomorrow', '30 minutes', '', '1d', ENDLESS_NUMBER + 'x', FULLWIDTH_3600,
                                   FULLWIDTH_45M])
def test_a_time_ollama_would_not_take_is_asked_again_in_plain_words(typed):
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', YES, ANOTHER_TIME, typed, '45m', DEFAULT, NO_MORE_BACKENDS,
                                   YES])
    assert ollama_entry(plan)['keep_loaded'] == '45m'
    assert 'keep_loaded' not in console.output
    assert typed == '' or 'That is not a time Ollama takes' in console.output


def test_a_number_beyond_all_numbers_is_no_time():
    plan, _, _ = run_wizard([*PASTED_GATEWAY, '1', YES, ANOTHER_TIME, ENDLESS_NUMBER, '2h', DEFAULT,
                             NO_MORE_BACKENDS, YES])
    assert ollama_entry(plan)['keep_loaded'] == '2h'


def test_the_chosen_models_can_be_loaded_at_start():
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1,2', YES, TWO_HOURS, YES, NO_MORE_BACKENDS, YES])
    assert ollama_entry(plan) == {**OLLAMA_WITH_FIRST_MODEL, 'models': MODELS[:2], 'keep_loaded': '2h',
                                  'load_at_start': True}
    loaded = parse_config(plan.data, environ={}).backends['ollama']
    assert loaded.load_at_start and loaded.keep_loaded == '2h'
    assert 'OLLAMA_MAX_LOADED_MODELS' not in console.output


def test_loading_at_start_without_a_longer_time_says_how_long_the_models_then_stay():
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', DEFAULT, YES, NO_MORE_BACKENDS, YES])
    assert ollama_entry(plan) == {**OLLAMA_WITH_FIRST_MODEL, 'load_at_start': True}
    assert 'takes them out again after its 5 minutes' in console.output
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', YES, TWO_HOURS, YES, NO_MORE_BACKENDS, YES])
    assert 'takes them out again' not in console.output


def test_more_models_than_ollama_keeps_by_default_are_met_with_what_can_be_known():
    _, console, _ = run_wizard([*PASTED_GATEWAY, 'all', DEFAULT, DEFAULT, NO_MORE_BACKENDS, YES])
    assert f'You chose {MODEL_COUNT} models' in console.output
    assert 'OLLAMA_MAX_LOADED_MODELS' in console.output and 'at most 3' in console.output


def test_a_backend_that_shares_whatever_it_has_is_not_offered_loading_at_start():
    plan, console, _ = run_wizard([*PASTED_GATEWAY, 'future', YES, YES, THIRTY_MINUTES, NO_MORE_BACKENDS, YES])
    assert ollama_entry(plan) == {**OLLAMA_WITH_FIRST_MODEL, 'models': [], 'keep_loaded': '30m'}
    assert LOAD_QUESTION not in console.output
    assert 'not offered' in console.output and 'more memory than there is' in console.output
    assert parse_config(plan.data, environ={}).backends['ollama'].keep_loaded == '30m'


@pytest.mark.parametrize('typed', ['0', '3m', '5m', '300', '299.9', '90s', '0s'])
def test_after_a_yes_to_longer_a_time_that_is_not_longer_is_asked_again(typed):
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', YES, ANOTHER_TIME, typed, '6m', DEFAULT, NO_MORE_BACKENDS,
                                   YES])
    assert ollama_entry(plan)['keep_loaded'] == '6m'
    assert "That is not longer than Ollama's 5 minutes" in console.output
    assert 'by hand' in console.output  # where a shorter time can still be set


@pytest.mark.parametrize('typed', ['3000000h', '9223372037', '1e300', ENDLESS_NUMBER, '9007199254740993.5h'])
def test_a_time_longer_than_ollama_can_count_is_asked_again(typed):
    plan, console, _ = run_wizard([*PASTED_GATEWAY, '1', YES, ANOTHER_TIME, typed, '6m', DEFAULT, NO_MORE_BACKENDS,
                                   YES])
    assert ollama_entry(plan)['keep_loaded'] == '6m'
    assert 'That is not a time Ollama takes' in console.output


def test_every_ollama_backend_is_asked_and_no_other_kind():
    discovery = ScriptedDiscovery({OLLAMA_URL: [MODELS], OTHER_OLLAMA_URL: [MODELS], LITELLM_URL: [MODELS],
                                   VLLM_URL: [MODELS]})
    plan, console, _ = run_wizard([
        *PASTED_GATEWAY, '1', DEFAULT, DEFAULT,
        ADD_LITELLM, DEFAULT, DEFAULT, DEFAULT, '1',
        ADD_OPENAI, DEFAULT, VLLM_URL, DEFAULT, '1',
        ADD_OLLAMA, DEFAULT, OTHER_OLLAMA_URL, DEFAULT, '2', YES, UNTIL_OLLAMA_STOPS, YES,
        NO_MORE_BACKENDS, YES], discovery)
    assert console.output.count(KEEP_QUESTION) == 2 and console.output.count(LOAD_QUESTION) == 2
    assert [(backend['kind'], backend.get('keep_loaded'), backend.get('load_at_start'))
            for backend in plan.data['backends']] == [
        (KIND_OLLAMA, None, None), (KIND_LITELLM, None, None), (KIND_OPENAI, None, None), (KIND_OLLAMA, -1, True)]
    parse_config(plan.data, environ={})


def test_the_summary_says_what_was_decided_about_memory():
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', YES, TWO_HOURS, YES, NO_MORE_BACKENDS, YES])
    summary = console.output.split('Step 3 of 3')[1]
    assert 'stays in memory: for 2h after a request through toptic.ai' in summary
    assert 'loaded when the client starts: yes' in summary
    _, console, _ = run_wizard([*PASTED_GATEWAY, '1', DEFAULT, DEFAULT, NO_MORE_BACKENDS, YES])
    summary = console.output.split('Step 3 of 3')[1]
    assert 'stays in memory: as long as Ollama itself keeps' in summary
    assert 'loaded when the client starts: no' in summary


def test_the_summary_of_another_kind_says_nothing_about_memory():
    data = {'gateway': GATEWAY, 'backends': [{'name': 'proxy', 'kind': KIND_LITELLM, 'base_url': LITELLM_URL,
                                              'models': []}]}
    assert 'memory' not in '\n'.join(summary_lines(data))
