"""What the client learns from an answer while relaying it: the token counts and the backend's error."""
import json

from socketllama import answer_watch
from socketllama.answer_watch import AnswerWatcher, TokenUsage

SSE = 'text/event-stream'
JSON_TYPE = 'application/json'
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500
ANSWER_TEXT = 'The "usage" of the word "error" is'
USAGE = {'prompt_tokens': 812, 'completion_tokens': 304, 'total_tokens': 1116}
SMALL_LIMIT_BYTES = 64


def sse_event(payload: dict) -> bytes:
    return f'data: {json.dumps(payload)}\n\n'.encode()


def content_event(text: str) -> bytes:
    return sse_event({'choices': [{'delta': {'content': text}}]})


DONE = b'data: [DONE]\n\n'


def watched(content_type: str, status: int, *chunks: bytes) -> answer_watch.AnswerFacts:
    watcher = AnswerWatcher(status, content_type)
    for chunk in chunks:
        watcher.feed(chunk)
    return watcher.result()


def test_the_usage_of_the_last_event_of_a_stream_gives_the_token_counts():
    facts = watched(SSE, HTTP_OK, content_event(ANSWER_TEXT), sse_event({'choices': [], 'usage': USAGE}), DONE)
    assert facts.usage == TokenUsage(prompt_tokens=812, completion_tokens=304)
    assert facts.error is None and facts.succeeded


def test_an_event_split_over_chunks_is_read_whole():
    stream = content_event(ANSWER_TEXT) + sse_event({'choices': [], 'usage': USAGE}) + DONE
    chunks = [stream[index:index + 1] for index in range(len(stream))]
    assert watched(SSE, HTTP_OK, *chunks).usage == TokenUsage(prompt_tokens=812, completion_tokens=304)


def test_events_separated_by_carriage_returns_are_read_too():
    event = sse_event({'choices': [], 'usage': USAGE}).replace(b'\n', b'\r\n')
    assert watched(SSE, HTTP_OK, event, DONE).usage == TokenUsage(prompt_tokens=812, completion_tokens=304)


def test_a_stream_without_usage_has_no_token_counts():
    facts = watched(SSE, HTTP_OK, content_event(ANSWER_TEXT), DONE)
    assert facts.usage is None and facts.succeeded


def test_usage_that_is_not_a_count_is_ignored():
    for usage in (None, {'prompt_tokens': '812', 'completion_tokens': 304}, {'prompt_tokens': True,
                                                                            'completion_tokens': 1},
                  {'prompt_tokens': -1, 'completion_tokens': 1}, ['812']):
        assert watched(SSE, HTTP_OK, sse_event({'choices': [], 'usage': usage}), DONE).usage is None


def test_the_usage_of_an_answer_that_is_not_streamed():
    body = json.dumps({'choices': [{'message': {'content': ANSWER_TEXT}}], 'usage': USAGE}, indent=2).encode()
    middle = len(body) // 2
    assert watched(JSON_TYPE, HTTP_OK, body[:middle], body[middle:]).usage == \
        TokenUsage(prompt_tokens=812, completion_tokens=304)


def test_an_error_answer_gives_the_backends_message():
    body = json.dumps({'error': {'message': "model 'x' not found", 'type': 'api_error'}}).encode()
    facts = watched(JSON_TYPE, HTTP_NOT_FOUND, body)
    assert facts.error == "model 'x' not found" and not facts.succeeded


def test_an_error_given_as_text_is_read_too():
    facts = watched(JSON_TYPE, HTTP_SERVER_ERROR, json.dumps({'error': 'llama runner process has terminated'}).encode())
    assert facts.error == 'llama runner process has terminated'


def test_an_error_answer_that_is_not_json_has_only_its_status():
    facts = watched('text/plain', HTTP_SERVER_ERROR, b'Internal Server Error')
    assert facts.error is None and not facts.succeeded and facts.status == HTTP_SERVER_ERROR


def test_an_error_event_in_the_middle_of_a_stream_fails_the_answer():
    facts = watched(SSE, HTTP_OK, content_event(ANSWER_TEXT),
                    sse_event({'error': {'message': 'model runner crashed\nretry'}}))
    assert facts.error == 'model runner crashed\\nretry' and not facts.succeeded


def test_the_answer_text_is_never_kept():
    facts = watched(SSE, HTTP_OK, content_event(ANSWER_TEXT), DONE)
    assert ANSWER_TEXT not in repr(facts)


def test_an_overlong_event_is_passed_over_and_the_next_one_still_read(monkeypatch):
    monkeypatch.setattr(answer_watch, 'MAX_WATCHED_LINE_BYTES', SMALL_LIMIT_BYTES)
    overlong = sse_event({'choices': [], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                          'padding': 'x' * SMALL_LIMIT_BYTES})
    facts = watched(SSE, HTTP_OK, overlong, sse_event({'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}))
    assert facts.usage == TokenUsage(prompt_tokens=2, completion_tokens=3)


def test_an_overlong_event_never_counts_even_in_pieces(monkeypatch):
    monkeypatch.setattr(answer_watch, 'MAX_WATCHED_LINE_BYTES', SMALL_LIMIT_BYTES)
    overlong = sse_event({'padding': 'x' * SMALL_LIMIT_BYTES, 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
    assert watched(SSE, HTTP_OK, *[overlong[index:index + 1] for index in range(len(overlong))]).usage is None


def test_a_body_beyond_the_limit_is_not_read(monkeypatch):
    monkeypatch.setattr(answer_watch, 'MAX_WATCHED_BODY_BYTES', SMALL_LIMIT_BYTES)
    body = json.dumps({'choices': [{'message': {'content': 'x' * SMALL_LIMIT_BYTES}}], 'usage': USAGE}).encode()
    facts = watched(JSON_TYPE, HTTP_OK, body)
    assert facts.usage is None and facts.succeeded


def test_a_long_error_message_is_cut_to_one_short_line():
    facts = watched(JSON_TYPE, HTTP_SERVER_ERROR, json.dumps({'error': 'e' * answer_watch.ERROR_SHOWN_CHARS * 2})
                    .encode())
    assert len(facts.error) <= answer_watch.ERROR_SHOWN_CHARS


def test_the_kind_of_an_error_is_kept_apart_from_its_message():
    """The message may quote the request; the kind is an identifier and safe to show."""
    body = json.dumps({'error': {'message': 'Invalid content: What should we offer', 'type': 'invalid_request_error',
                                 'code': 'bad_content'}}).encode()
    facts = watched(JSON_TYPE, HTTP_NOT_FOUND, body)
    assert facts.error_kind == 'invalid_request_error'
    assert facts.error == 'Invalid content: What should we offer'


def test_without_a_type_the_code_is_the_kind_and_without_both_there_is_none():
    assert watched(JSON_TYPE, HTTP_NOT_FOUND, json.dumps({'error': {'message': 'm', 'code': 'not_found'}}).encode()
                   ).error_kind == 'not_found'
    assert watched(JSON_TYPE, HTTP_NOT_FOUND, json.dumps({'error': 'text only'}).encode()).error_kind is None
