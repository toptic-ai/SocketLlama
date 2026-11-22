"""The lines the running client shows its owner: one per event, calm, never a prompt's text or a secret."""
import logging
import time

import pytest

from socketllama import events, protocol
from socketllama.answer_watch import TokenUsage
from socketllama.config import ModelPrice
from socketllama.console import LOG_TIME_FORMAT, LogFormatter
from socketllama.events import PromptSize, prompt_size
from socketllama.text import ELLIPSIS

USER_NAME = 'Alice Example'
BACKEND = 'ollama'
MODEL = 'deepseek-r1:32b'
NUMBER = 3
SECONDS = 21.34
USAGE = TokenUsage(prompt_tokens=1812, completion_tokens=304)
PROMPT_TEXT = 'What should we offer in the merger talks?'
IMAGE_PART = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGgo='}}
# A terminal control sequence (CSI) a peer could slip into a name
CONTROL_SEQUENCE = '\x1b[2J'
GATEWAY_URL = 'wss://toptic.example/ws/socketllama/'
INSTANCE = 'home-gpu'
FIXED_TIME = 1_700_000_000.0


# ----- the prompt -----

def test_the_prompt_line_names_the_number_the_person_the_model_and_the_size():
    line = events.prompt_received(NUMBER, events.requester(USER_NAME), BACKEND, MODEL,
                                  PromptSize(messages=4, characters=1234, attachments=0))
    assert line == '#3 prompt from Alice Example for ollama/deepseek-r1:32b: 4 messages, 1,234 characters'


def test_without_a_name_the_prompt_is_from_an_unknown_user():
    for absent in (None, '', '   ', 42, ['Alice']):
        line = events.prompt_received(NUMBER, events.requester(absent), BACKEND, MODEL,
                                      PromptSize(messages=1, characters=1, attachments=0))
        assert line == '#3 prompt from unknown user for ollama/deepseek-r1:32b: 1 message, 1 character'


def test_attachments_are_counted_not_measured():
    line = events.prompt_received(NUMBER, USER_NAME, BACKEND, MODEL, PromptSize(messages=2, characters=10,
                                                                                attachments=1))
    assert line.endswith(': 2 messages, 10 characters, 1 attachment')


def test_a_name_from_the_gateway_is_made_printable_and_kept_short():
    assert events.requester(f'Mallory{CONTROL_SEQUENCE}') == 'Mallory\\x1b[2J'
    long_name = 'n' * (protocol.USER_NAME_MAX_LENGTH * 2)
    assert len(events.requester(long_name)) <= protocol.USER_NAME_MAX_LENGTH


def test_the_size_of_a_prompt_counts_messages_and_the_characters_of_their_text():
    payload = {'model': MODEL, 'messages': [
        {'role': 'system', 'content': 'Be brief.'},
        {'role': 'user', 'content': [{'type': 'text', 'text': PROMPT_TEXT}, IMAGE_PART]},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1'}]},
        {'role': 'tool', 'content': 'result'},
    ]}
    assert prompt_size(payload) == PromptSize(messages=4, characters=len('Be brief.') + len(PROMPT_TEXT)
                                              + len('result'), attachments=1)


def test_the_size_of_a_malformed_prompt_is_what_can_be_counted():
    assert prompt_size({'messages': 'not a list'}) == PromptSize(messages=0, characters=0, attachments=0)
    assert prompt_size({'messages': ['not a message', {'content': [7, {'type': 'text', 'text': 9}]}]}) == \
        PromptSize(messages=2, characters=0, attachments=0)


# ----- the end of the answer -----

def test_a_finished_answer_says_how_long_it_took_and_the_token_counts_the_backend_gave():
    assert events.answer_ended(NUMBER, events.OUTCOME_DONE, SECONDS, USAGE) == \
        '#3 done after 21.3 s, 1,812 prompt tokens, 304 answer tokens'


def test_without_token_counts_the_end_line_has_none():
    assert events.answer_ended(NUMBER, events.OUTCOME_DONE, SECONDS) == '#3 done after 21.3 s'


def test_a_cancelled_answer_says_why():
    assert events.answer_ended(NUMBER, events.OUTCOME_CANCELLED, SECONDS, detail=events.STOPPED_BY_GATEWAY) == \
        '#3 cancelled after 21.3 s: stopped by toptic.ai'


def test_a_failed_answer_says_what_failed():
    assert events.answer_ended(NUMBER, events.OUTCOME_ERROR, SECONDS, detail='HTTP 404 from ollama: not found') == \
        '#3 error after 21.3 s: HTTP 404 from ollama: not found'


def test_pause_resume_and_a_model_to_load_are_one_line_each():
    assert events.paused(NUMBER) == '#3 paused: toptic.ai takes the answer more slowly than it comes'
    assert events.resumed(NUMBER) == '#3 resumed'
    assert events.model_not_in_memory(NUMBER, BACKEND, MODEL) == \
        '#3 ollama/deepseek-r1:32b is not in memory: Ollama loads it first'


def test_a_load_the_client_asks_for_is_told_from_the_request_that_ended_to_its_own_end():
    """Only what the client knows: no answer had begun, and the model was not in memory when the prompt came."""
    assert events.load_goes_on(NUMBER, BACKEND, MODEL) == (
        '#3 ended before ollama/deepseek-r1:32b began its answer, and the model was not in memory when the prompt '
        'came: the client has Ollama load it, so that the next prompt finds it in memory')
    assert events.load_already_going_on(NUMBER, BACKEND, MODEL) == (
        '#3 ended before ollama/deepseek-r1:32b began its answer, and the model was not in memory when the prompt '
        'came: the client has Ollama load it already')
    assert events.load_finished(NUMBER, BACKEND, MODEL, SECONDS) == (
        'ollama/deepseek-r1:32b is in memory after 21.3 s: the client had Ollama load it after #3 ended without '
        'an answer')
    assert events.load_failed(NUMBER, BACKEND, MODEL, 'out of memory') == (
        'ollama/deepseek-r1:32b could not be loaded after #3 ended without an answer: out of memory. A prompt to '
        'it makes Ollama try again')


def test_how_long_the_gateway_waits_for_a_model_is_one_line_and_so_is_a_wait_that_is_ignored():
    assert events.gateway_waits_for_loading(900) == (
        'toptic.ai waits up to 900 s for a model to load: this client waits at least as long for a model that was '
        'not in memory when its prompt came, and for a load it asks Ollama for')
    assert events.loading_wait_shortened(7200, 3600) == (
        'toptic.ai says it waits 7,200 s for a model to load: this client takes that as 3,600 s, the longest it '
        "waits on the word of toptic.ai. A backend's read_timeout can be set longer")
    assert events.loading_wait_not_said() == (
        'This toptic.ai does not say how long it waits for a model to load: the client waits as long as each '
        "backend's read_timeout says again")
    ignored = events.loading_wait_ignored(CONTROL_SEQUENCE + 'is not a whole number of seconds')
    assert ignored.isprintable() and ignored.startswith('Ignored how long toptic.ai says it waits for a model to load')


def test_a_backend_that_is_still_loading_a_model_is_one_line():
    assert events.backend_loading_model(NUMBER, BACKEND) == (
        '#3 ollama is still loading a model and answers no prompt until it is in memory: toptic.ai is told so '
        "instead of being passed Ollama's error")


def test_what_ollama_says_about_a_failed_load_is_made_printable_and_kept_short():
    line = events.load_failed(NUMBER, BACKEND, MODEL + CONTROL_SEQUENCE, CONTROL_SEQUENCE + 'x' * 1000)
    assert line.isprintable() and len(line) < 400


def test_a_refused_request_says_why_and_takes_no_number():
    assert events.request_refused(protocol.ERROR_BAD_SIGNATURE, 'Request signature does not match') == \
        'Refused a request: bad_signature, Request signature does not match'


# ----- the connection -----

def test_connected_names_the_machine():
    assert events.connected(GATEWAY_URL, INSTANCE) == f'Connected to {GATEWAY_URL} as machine {INSTANCE}'


@pytest.mark.parametrize('reason, explained', [
    (protocol.AUTH_FAILED_TOKEN, 'wrong, revoked or expired'),
    (protocol.AUTH_FAILED_SCOPE, 'connection token'),
    (protocol.AUTH_FAILED_INSTANCE, 'config block'),
    (protocol.AUTH_FAILED_PROTOCOL, 'update the client'),
    (protocol.AUTH_FAILED_SECRET_UNREADABLE, 'new signing secret'),
])
def test_a_refused_authentication_says_the_reason_and_what_it_means(reason, explained):
    line = events.auth_refused(reason)
    assert line.startswith(f'toptic.ai refused this machine ({reason}): ')
    assert explained in line


def test_an_unknown_reason_is_shown_as_given_and_printable():
    assert events.auth_refused(f'new{CONTROL_SEQUENCE}') == \
        "toptic.ai refused this machine (new\\x1b[2J): the machine's page on toptic.ai says why"
    assert events.auth_refused(None) == "toptic.ai refused this machine (no reason given): the machine's page on " \
                                        "toptic.ai says why"


# ----- models -----

REPORT = [{'name': 'ollama', 'kind': 'ollama', 'models': [{'name': 'a'}, {'name': 'b'}]},
          {'name': 'proxy', 'kind': 'litellm', 'models': [{'name': 'c'}]},
          {'name': 'busy', 'kind': 'ollama', protocol.BACKEND_CHECKED_FIELD: False, 'models': []}]


def test_the_models_report_line_counts_the_models_of_each_backend():
    assert events.models_reported(REPORT) == \
        'Reported to toptic.ai: ollama 2 models, proxy 1 model, busy not checked'


def test_the_acknowledgement_counts_what_changed():
    ack = {'type': protocol.FRAME_MODELS_ACK, 'created': ['x/ollama/a'], 'updated': [], 'removed': ['x/ollama/z'],
           'skipped': [], 'unchecked': ['busy']}
    assert events.models_registered(ack, REPORT) == \
        'toptic.ai registered the 3 models reported: 1 new, 1 removed, 1 backend kept as not checked'


def test_an_acknowledgement_without_changes_says_so():
    assert events.models_registered({'type': protocol.FRAME_MODELS_ACK}, REPORT) == \
        'toptic.ai registered the 3 models reported: no changes'


def test_the_models_in_memory_are_listed_per_backend():
    snapshot = [{'backend': 'ollama', 'models': ['a', 'b']}, {'backend': 'other', 'models': []}]
    assert events.models_in_memory(snapshot) == 'In memory now: ollama a, b; other none'


# ----- the log format -----

def _formatted(level: int, message: str) -> str:
    record = logging.LogRecord('socketllama', level, __file__, 0, message, None, None)
    record.created = FIXED_TIME
    return LogFormatter().format(record)


def test_every_line_starts_with_the_local_time():
    local = time.strftime(LOG_TIME_FORMAT, time.localtime(FIXED_TIME))
    assert _formatted(logging.INFO, 'an event') == f'{local} an event'


def test_the_level_is_named_only_where_it_is_not_info():
    local = time.strftime(LOG_TIME_FORMAT, time.localtime(FIXED_TIME))
    assert _formatted(logging.WARNING, 'a problem') == f'{local} WARNING a problem'
    assert _formatted(logging.DEBUG, 'a detail') == f'{local} DEBUG a detail'


def test_a_refusal_message_from_the_gateways_frame_is_printable_and_short():
    line = events.request_refused(protocol.ERROR_PATH_NOT_ALLOWED, f'POST /x{CONTROL_SEQUENCE}' + 'p' * 1000)
    assert CONTROL_SEQUENCE not in line and '\\x1b[2J' in line
    assert len(line) < len('Refused a request: ') + len(protocol.ERROR_PATH_NOT_ALLOWED) + 1000


def test_an_unknown_reason_is_kept_short():
    assert len(events.auth_refused('r' * 1000)) < 1000


def test_another_programs_info_line_names_its_level():
    local = time.strftime(LOG_TIME_FORMAT, time.localtime(FIXED_TIME))
    record = logging.LogRecord('httpx', logging.INFO, __file__, 0, 'HTTP Request: GET /', None, None)
    record.created = FIXED_TIME
    assert LogFormatter().format(record) == f'{local} INFO HTTP Request: GET /'


# ----- prices -----

@pytest.mark.parametrize('price, words', [
    (ModelPrice(input=0.8, output=2.0), '$0.80 in, $2.00 out per 1M tokens'),
    (ModelPrice(input=0.0005, output=12.5), '$0.0005 in, $12.50 out per 1M tokens'),
    (ModelPrice(input=0.0, output=1.0), '$0.00 in, $1.00 out per 1M tokens'),
    (ModelPrice(input=0.0, output=0.0), 'free'),
])
def test_a_price_is_shown_in_dollars_with_every_place_it_has(price, words):
    assert events.price_in_words(price) == words


def test_a_price_that_was_not_written_names_the_line_to_put_in_by_hand():
    price = ModelPrice(input=0.4, output=1.4)
    line = events.setting_not_written('price', 'backends.prices_per_1m_tokens', 'ollama', 'llama3.1:8b',
                                      events.price_in_words(price), '/home/me/config.toml', 'the file is read-only',
                                      '"llama3.1:8b" = { input = 0.4, output = 1.4 }')
    assert 'read-only' in line and '[backends.prices_per_1m_tokens] of backend ollama' in line
    assert line.endswith('"llama3.1:8b" = { input = 0.4, output = 1.4 }')


def test_model_names_in_price_lines_are_made_printable():
    line = events.setting_written('price', 'ollama', 'evil\x1b[2J', events.price_in_words(ModelPrice(input=1.0,
                                                                                                     output=1.0)),
                                  'config.toml')
    assert '\x1b' not in line


def test_a_long_description_is_shown_shortened_but_put_in_by_hand_whole():
    description = 'x' * protocol.MAX_MODEL_DESCRIPTION_CHARS
    line_to_add = f'"llama3.1:8b" = "{description}"'
    shown = events.description_in_words(description)
    line = events.setting_not_written('description', 'backends.descriptions', 'ollama', 'llama3.1:8b', shown,
                                      'config.toml', 'the file is read-only', line_to_add)
    assert len(shown) < len(description) and shown.endswith(f'{ELLIPSIS}"')
    assert line.endswith(line_to_add)


def test_a_description_is_shown_printable():
    # a character a terminal does not show as such (a no-break space) is shown as its escape
    assert events.description_in_words('A\u00a0B') == '"A\\xa0B"'
