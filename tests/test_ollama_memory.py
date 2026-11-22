"""Telling Ollama how long to keep a model in memory, and loading models at start, through its own endpoints."""
import asyncio
import logging

import httpx
import pytest
import pytest_asyncio

from socketllama import events, ollama_memory
from socketllama.config import DEFAULT_BACKEND_READ_TIMEOUT_SECONDS, KIND_OLLAMA, Backend
from socketllama.ollama_memory import (
    OLLAMA_DEFAULT_MAX_LOADED_MODELS, ModelMemory, keep_loaded_in_words, load_model,
)

from .conftest import wait_until
from .fakes import BACKEND_API_KEY, HTTP_NOT_FOUND, HTTP_OK, MODEL, FakeBackend, ManualTimeouts

KEEP_LOADED = '30m'
# Only what the client itself logs is looked at (the HTTP library logs every request at this level)
CLIENT_LOGGER = 'socketllama'
SECOND_MODEL = 'qwen3:14b'
TOO_LARGE_MODEL = 'huge:70b'
NOT_PULLED_MODEL = 'never-pulled:1b'
GENERATE = '/api/generate'
PROCESS_LIST = '/api/ps'
LOAD_SECONDS = 2.5
SETTLE_TIMEOUT_SECONDS = 5.0
QUIET_SECONDS = 0.05
REPEATS = 3
MANY = 20
LONG_READ_TIMEOUT_SECONDS = 1800.0
UNEXPECTED_FAILURE = 'a bug, not a failing request'
KEEP_MODEL_LOADED = {'model': MODEL, 'keep_alive': KEEP_LOADED}


def ollama(fake: FakeBackend, **settings) -> Backend:
    return Backend(**{'name': 'ollama', 'kind': KIND_OLLAMA, 'base_url': fake.base_url, 'models': (MODEL,),
                      'keep_loaded': KEEP_LOADED, **settings})


def sent(fake: FakeBackend, path: str) -> list:
    """The JSON bodies of the requests the fake got for ``path``."""
    return [request['json'] for request in fake.requests if request['path'] == path]


async def wait_for_requests(fake: FakeBackend, path: str, count: int):
    await wait_until(lambda: len(sent(fake, path)) >= count)


@pytest_asyncio.fixture
async def http():
    async with httpx.AsyncClient() as client:
        yield client


@pytest_asyncio.fixture
async def memory(http):
    keeper = ModelMemory(http)
    yield keeper
    await keeper.close()


async def answered(memory: ModelMemory, backend: Backend, model: str = MODEL):
    """Tell that a model answered through the tunnel, and wait for what the client does about it."""
    memory.model_answered(backend, model)
    await asyncio.wait_for(memory.idle(), SETTLE_TIMEOUT_SECONDS)


def ticking_clock(step: float):
    """A clock that is ``step`` seconds further each time it is read."""
    now = 0.0

    def clock() -> float:
        nonlocal now
        now += step
        return now
    return clock


# ----- one request to Ollama -----

@pytest.mark.parametrize('keep_loaded', ['30m', '1h30m', 3600, -1, 0.5])
async def test_loading_a_model_is_an_empty_generate_request_with_the_time_as_it_was_configured(http, keep_loaded):
    async with FakeBackend() as fake:
        await load_model(ollama(fake, keep_loaded=keep_loaded, api_key=BACKEND_API_KEY), http, MODEL)
    assert sent(fake, GENERATE) == [{'model': MODEL, 'keep_alive': keep_loaded}]
    assert fake.requests[-1]['method'] == 'POST'
    assert fake.requests[-1]['headers']['authorization'] == f'Bearer {BACKEND_API_KEY}'
    assert fake.in_memory == {MODEL: keep_loaded}


async def test_without_a_time_none_is_sent_and_ollamas_own_applies(http):
    async with FakeBackend() as fake:
        await load_model(ollama(fake, keep_loaded=None), http, MODEL)
    assert sent(fake, GENERATE) == [{'model': MODEL}]


async def test_a_model_that_cannot_be_loaded_is_an_error_with_ollamas_reason(http):
    async with FakeBackend(models_that_do_not_fit=[TOO_LARGE_MODEL]) as fake:
        with pytest.raises(httpx.HTTPStatusError):
            await load_model(ollama(fake), http, TOO_LARGE_MODEL)


# ----- after an answer through the tunnel -----

async def test_a_model_that_answered_and_is_in_memory_is_told_how_long_to_stay(memory):
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None  # loaded by the request that was relayed, with Ollama's default
        await answered(memory, ollama(fake))
    assert sent(fake, GENERATE) == [{'model': MODEL, 'keep_alive': KEEP_LOADED}]
    assert fake.in_memory == {MODEL: KEEP_LOADED}


async def test_a_model_named_without_its_tag_is_recognized_in_memory(memory):
    async with FakeBackend(more_models=['llama3.1']) as fake:
        fake.in_memory['llama3.1:latest'] = None
        await answered(memory, ollama(fake), 'llama3.1')
    assert sent(fake, GENERATE) == [{'model': 'llama3.1', 'keep_alive': KEEP_LOADED}]


async def test_a_model_that_is_not_in_memory_is_never_loaded_by_this(memory, caplog):
    """Unloaded in the meantime (another model took its place, or Ollama is set to unload at once)."""
    async with FakeBackend() as fake:
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            await answered(memory, ollama(fake))
    assert sent(fake, GENERATE) == [] and fake.in_memory == {}
    assert len(sent(fake, PROCESS_LIST)) == 1
    assert caplog.records == []


async def test_a_backend_without_the_setting_is_left_alone(memory):
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        await answered(memory, ollama(fake, keep_loaded=None))
    assert fake.requests == []


async def test_telling_ollama_never_holds_up_whoever_reports_the_answer(memory):
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        fake.hold_generate.clear()
        memory.model_answered(ollama(fake), MODEL)
        await asyncio.sleep(QUIET_SECONDS)
        assert fake.in_memory == {MODEL: None}  # still on its way: nobody waited for it
        fake.hold_generate.set()
        await asyncio.wait_for(memory.idle(), SETTLE_TIMEOUT_SECONDS)
    assert fake.in_memory == {MODEL: KEEP_LOADED}


async def test_answers_that_end_one_after_another_each_tell_ollama(memory):
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        for _ in range(REPEATS):
            await answered(memory, ollama(fake))
    assert sent(fake, GENERATE) == [{'model': MODEL, 'keep_alive': KEEP_LOADED}] * REPEATS


async def test_while_ollama_is_being_told_further_answers_of_that_model_add_one_more_telling_not_one_each(memory):
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        fake.hold_generate.clear()
        backend = ollama(fake)
        memory.model_answered(backend, MODEL)
        await wait_for_requests(fake, GENERATE, 1)
        for _ in range(MANY):
            memory.model_answered(backend, MODEL)  # requests that ended while the first telling is under way
        await asyncio.sleep(QUIET_SECONDS)
        assert len(sent(fake, GENERATE)) == 1  # one at a time for a model, whatever ends meanwhile
        fake.hold_generate.set()
        await asyncio.wait_for(memory.idle(), SETTLE_TIMEOUT_SECONDS)
    # Each of those requests set Ollama's time anew when it ended, after the first telling had gone out.
    # Told once more after that telling returned: the last word follows the last request that ended
    assert sent(fake, GENERATE) == [{'model': MODEL, 'keep_alive': KEEP_LOADED}] * 2
    assert fake.in_memory == {MODEL: KEEP_LOADED}


async def test_an_answer_that_ends_during_the_second_telling_is_followed_by_a_third(memory):
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        fake.hold_generate.clear()
        backend = ollama(fake)
        memory.model_answered(backend, MODEL)
        await wait_for_requests(fake, GENERATE, 1)
        memory.model_answered(backend, MODEL)
        fake.hold_generate.set()
        await wait_for_requests(fake, GENERATE, 2)
        memory.model_answered(backend, MODEL)
        await asyncio.wait_for(memory.idle(), SETTLE_TIMEOUT_SECONDS)
    assert len(sent(fake, GENERATE)) == 3  # the third answer ended while the second telling was on its way
    assert fake.in_memory == {MODEL: KEEP_LOADED}


async def test_models_and_backends_are_told_apart(memory):
    async with FakeBackend(more_models=[SECOND_MODEL]) as fake, FakeBackend() as other:
        for server in (fake, other):
            server.in_memory[MODEL] = None
            server.hold_generate.clear()
        fake.in_memory[SECOND_MODEL] = None
        memory.model_answered(ollama(fake), MODEL)
        memory.model_answered(ollama(fake), SECOND_MODEL)
        memory.model_answered(ollama(other, name='other'), MODEL)
        await wait_for_requests(fake, GENERATE, 2)
        await wait_for_requests(other, GENERATE, 1)
        for server in (fake, other):
            server.hold_generate.set()
        await asyncio.wait_for(memory.idle(), SETTLE_TIMEOUT_SECONDS)
    assert fake.in_memory == {MODEL: KEEP_LOADED, SECOND_MODEL: KEEP_LOADED} and other.in_memory == {MODEL: KEEP_LOADED}


async def test_a_name_with_and_without_its_tag_is_one_model(memory):
    async with FakeBackend(more_models=['llama3.1', 'llama3.1:latest']) as fake:
        fake.in_memory['llama3.1:latest'] = None
        fake.hold_generate.clear()
        memory.model_answered(ollama(fake), 'llama3.1')
        await wait_for_requests(fake, GENERATE, 1)
        memory.model_answered(ollama(fake), 'llama3.1:latest')
        await asyncio.sleep(QUIET_SECONDS)
        assert len(sent(fake, GENERATE)) == 1
        fake.hold_generate.set()
        await asyncio.wait_for(memory.idle(), SETTLE_TIMEOUT_SECONDS)
    assert len(sent(fake, GENERATE)) == 2


async def test_telling_ollama_has_a_short_time_limit_of_its_own(memory, monkeypatch, caplog):
    """Not the backend's limit for a whole answer: this small request must not hold a connection for minutes."""
    monkeypatch.setattr(ollama_memory, 'KEEP_LOADED_TIMEOUT_SECONDS', QUIET_SECONDS)
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        fake.hold_generate.clear()
        with caplog.at_level(logging.WARNING, logger=CLIENT_LOGGER):
            await answered(memory, ollama(fake, read_timeout=LONG_READ_TIMEOUT_SECONDS))
    assert len(caplog.records) == 1 and 'could not tell Ollama' in caplog.records[0].getMessage()
    assert ollama_memory.KEEP_LOADED_TIMEOUT_SECONDS < DEFAULT_BACKEND_READ_TIMEOUT_SECONDS


async def test_a_telling_that_runs_out_of_time_while_another_answer_ended_is_followed_by_exactly_one_more(
        memory, monkeypatch):
    monkeypatch.setattr(ollama_memory, 'KEEP_LOADED_TIMEOUT_SECONDS', QUIET_SECONDS)
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        fake.hold_generate.clear()
        backend = ollama(fake)
        memory.model_answered(backend, MODEL)
        await wait_for_requests(fake, GENERATE, 1)
        memory.model_answered(backend, MODEL)  # ended while the first telling was on its way
        await asyncio.wait_for(memory.idle(), SETTLE_TIMEOUT_SECONDS)
    assert sent(fake, GENERATE) == [KEEP_MODEL_LOADED] * 2


async def test_an_unexpected_failure_is_logged_and_leaves_no_telling_owed(memory, monkeypatch, caplog):
    """A bug in telling Ollama must neither end in an exception nobody retrieves nor tell a later answer twice."""
    started, release = asyncio.Event(), asyncio.Event()

    async def failing_load(*arguments: object, **settings: object) -> None:
        started.set()
        await release.wait()
        raise RuntimeError(UNEXPECTED_FAILURE)
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        backend = ollama(fake)
        with monkeypatch.context() as patch, caplog.at_level(logging.WARNING, logger=CLIENT_LOGGER):
            patch.setattr(ollama_memory, 'load_model', failing_load)
            memory.model_answered(backend, MODEL)
            await asyncio.wait_for(started.wait(), SETTLE_TIMEOUT_SECONDS)
            memory.model_answered(backend, MODEL)  # owed one more telling when the failing one is over
            release.set()
            tasks = list(memory._telling.values())
            await asyncio.wait_for(memory.idle(), SETTLE_TIMEOUT_SECONDS)
        assert [task.exception() for task in tasks] == [None]  # swallowed, so asyncio has nothing to complain about
        assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.ERROR
        assert caplog.records[0].exc_info[1].args == (UNEXPECTED_FAILURE,)
        assert 'Backend ollama' in caplog.records[0].getMessage()
        await answered(memory, backend)
    assert sent(fake, GENERATE) == [KEEP_MODEL_LOADED]  # told once: nothing was left owed by the failure


async def test_loading_at_start_keeps_the_time_a_model_may_take_to_load(http):
    """Loading takes as long as it takes: the backend's own limit applies, not the short one, to the request
    as a whole; the HTTP library bounds no single phase of it."""
    recorded = []

    async def recording_post(url, **kwargs):
        recorded.append(kwargs['timeout'])
        return httpx.Response(HTTP_OK, json={}, request=httpx.Request('POST', url))
    http.post = recording_post
    timeouts = ManualTimeouts()
    async with FakeBackend() as fake:
        backend = ollama(fake, load_at_start=True, read_timeout=LONG_READ_TIMEOUT_SECONDS)
        await ModelMemory(http, timeout=timeouts).load_at_start([backend])
    assert timeouts.asked == [LONG_READ_TIMEOUT_SECONDS]
    assert [timeout.read for timeout in recorded] == [None]


async def test_a_failure_is_logged_once_in_plain_words_until_it_works_again(memory, caplog):
    async with FakeBackend(models_that_do_not_fit=[MODEL]) as fake:
        fake.in_memory[MODEL] = None
        backend = ollama(fake)
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            for _ in range(REPEATS):
                await answered(memory, backend)
            failures = [record for record in caplog.records if record.levelno == logging.WARNING]
            assert len(failures) == 1
            assert 'Backend ollama: could not tell Ollama to keep llama3.1:8b in memory for 30m' in (
                failures[0].getMessage())
            assert 'requires more system memory' in failures[0].getMessage()
            assert 'logged once' in failures[0].getMessage()
            fake.models_that_do_not_fit.clear()
            await answered(memory, backend)
            assert 'works again' in caplog.records[-1].getMessage()
            assert caplog.records[-1].levelno == logging.INFO
            fake.models_that_do_not_fit.append(MODEL)
            await answered(memory, backend)
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 2


async def test_an_ollama_that_does_not_tell_what_is_in_memory_is_not_asked_to_load_anything(memory, caplog):
    async with FakeBackend(process_list_status=HTTP_NOT_FOUND) as fake:
        fake.in_memory[MODEL] = None
        with caplog.at_level(logging.WARNING):
            for _ in range(REPEATS):
                await answered(memory, ollama(fake))
    assert sent(fake, GENERATE) == []
    assert len(caplog.records) == 1 and 'what it has in memory' in caplog.records[0].getMessage()


async def test_an_ollama_that_is_gone_fails_nothing_and_is_logged_once(memory, caplog):
    async with FakeBackend() as fake:
        gone = ollama(fake)
    with caplog.at_level(logging.WARNING):
        for _ in range(REPEATS):
            await answered(memory, gone)
    assert len(caplog.records) == 1


async def test_closing_drops_what_is_still_on_its_way(http):
    memory = ModelMemory(http)
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        fake.hold_generate.clear()
        memory.model_answered(ollama(fake), MODEL)
        await asyncio.sleep(QUIET_SECONDS)
        await asyncio.wait_for(memory.close(), SETTLE_TIMEOUT_SECONDS)
        assert fake.in_memory == {MODEL: None}


async def test_closing_leaves_nothing_on_its_way_and_nothing_owed(http):
    memory = ModelMemory(http)
    async with FakeBackend() as fake:
        backend = ollama(fake)
        memory.model_answered(backend, MODEL)
        memory.model_answered(backend, MODEL)  # owed one more telling: the first has not even started
        await asyncio.wait_for(memory.close(), SETTLE_TIMEOUT_SECONDS)
    assert memory._telling == {} and memory._tell_again == set()


async def test_a_telling_owed_when_closing_is_never_sent(http):
    memory = ModelMemory(http)
    async with FakeBackend() as fake:
        fake.in_memory[MODEL] = None
        fake.hold_generate.clear()
        backend = ollama(fake)
        memory.model_answered(backend, MODEL)
        await wait_for_requests(fake, GENERATE, 1)
        memory.model_answered(backend, MODEL)  # owed one more telling when the first is over
        await asyncio.wait_for(memory.close(), SETTLE_TIMEOUT_SECONDS)
        fake.hold_generate.set()
        await asyncio.sleep(QUIET_SECONDS)
        assert len(sent(fake, GENERATE)) == 1
        await answered(memory, backend)
    assert len(sent(fake, GENERATE)) == 2  # a later answer is told once: nothing owed outlived the closing


# ----- a load finished for a request that was cancelled -----

REQUEST_NUMBER = 7


async def wait_until_none_is_finished(memory: ModelMemory):
    await wait_until(lambda: not memory._finishing, SETTLE_TIMEOUT_SECONDS)


async def test_a_load_to_finish_is_one_generate_request_and_the_log_says_how_it_ended(http, caplog):
    memory = ModelMemory(http, clock=ticking_clock(LOAD_SECONDS))
    async with FakeBackend() as fake:
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            memory.finish_loading(ollama(fake, keep_loaded=None), MODEL, REQUEST_NUMBER)
            await wait_until_none_is_finished(memory)
    assert sent(fake, GENERATE) == [{'model': MODEL}] and fake.in_memory == {MODEL: None}
    assert [record.getMessage() for record in caplog.records] == [
        events.load_goes_on(REQUEST_NUMBER, 'ollama', MODEL),
        events.load_finished(REQUEST_NUMBER, 'ollama', MODEL, LOAD_SECONDS)]


async def test_asking_to_finish_a_load_never_holds_up_whoever_asks(memory):
    async with FakeBackend() as fake:
        fake.hold_generate.clear()
        memory.finish_loading(ollama(fake), MODEL, REQUEST_NUMBER)  # returns although Ollama does not answer
        await wait_for_requests(fake, GENERATE, 1)
        assert fake.in_memory == {}
        fake.hold_generate.set()
        await wait_until_none_is_finished(memory)
    assert fake.in_memory == {MODEL: KEEP_LOADED}


async def test_a_model_being_loaded_is_not_loaded_a_second_time_under_either_of_its_names(memory, caplog):
    async with FakeBackend(more_models=['llama3.1', 'llama3.1:latest']) as fake:
        fake.hold_generate.clear()
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            memory.finish_loading(ollama(fake), 'llama3.1', REQUEST_NUMBER)
            await wait_for_requests(fake, GENERATE, 1)
            memory.finish_loading(ollama(fake), 'llama3.1:latest', REQUEST_NUMBER + 1)
            await asyncio.sleep(QUIET_SECONDS)
            fake.hold_generate.set()
            await wait_until_none_is_finished(memory)
    assert len(sent(fake, GENERATE)) == 1
    assert events.load_already_going_on(REQUEST_NUMBER + 1, 'ollama', 'llama3.1:latest') in caplog.messages


async def test_once_a_load_is_over_the_model_can_be_loaded_again(memory):
    async with FakeBackend() as fake:
        for _ in range(REPEATS):
            memory.finish_loading(ollama(fake), MODEL, REQUEST_NUMBER)
            await wait_until_none_is_finished(memory)
    assert len(sent(fake, GENERATE)) == REPEATS


async def test_the_clients_own_loads_go_to_ollama_one_after_another(memory):
    """Those at start and those it finishes: the second is not sent before the first was answered."""
    async with FakeBackend(more_models=[SECOND_MODEL]) as fake:
        fake.hold_generate.clear()
        at_start = asyncio.create_task(memory.load_at_start([ollama(fake, load_at_start=True)]))
        await wait_for_requests(fake, GENERATE, 1)
        memory.finish_loading(ollama(fake), SECOND_MODEL, REQUEST_NUMBER)
        await asyncio.sleep(QUIET_SECONDS)
        assert sent(fake, GENERATE) == [KEEP_MODEL_LOADED]
        fake.hold_generate.set()
        await asyncio.wait_for(at_start, SETTLE_TIMEOUT_SECONDS)
        await wait_until_none_is_finished(memory)
    assert sent(fake, GENERATE) == [KEEP_MODEL_LOADED, {'model': SECOND_MODEL, 'keep_alive': KEEP_LOADED}]


@pytest.mark.parametrize('keep_loaded', [0, '0s'])
async def test_no_load_is_finished_for_a_backend_that_lets_no_model_stay(memory, caplog, keep_loaded):
    async with FakeBackend() as fake:
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            memory.finish_loading(ollama(fake, keep_loaded=keep_loaded), MODEL, REQUEST_NUMBER)
            await asyncio.sleep(QUIET_SECONDS)
    assert fake.requests == [] and caplog.records == []


async def test_an_ollama_that_is_gone_is_a_warning_not_an_exception(memory, caplog):
    async with FakeBackend() as fake:
        gone = ollama(fake)
    with caplog.at_level(logging.WARNING, logger=CLIENT_LOGGER):
        memory.finish_loading(gone, MODEL, REQUEST_NUMBER)
        await wait_until_none_is_finished(memory)
    assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.WARNING
    assert caplog.messages[0].startswith(events.load_failed(REQUEST_NUMBER, 'ollama', MODEL, '').split(': ')[0])


async def test_an_unexpected_failure_of_a_load_to_finish_is_logged_and_leaves_nothing_behind(
        memory, monkeypatch, caplog):
    async def failing_load(*arguments: object, **settings: object) -> None:
        raise RuntimeError(UNEXPECTED_FAILURE)
    async with FakeBackend() as fake:
        with monkeypatch.context() as patch, caplog.at_level(logging.WARNING, logger=CLIENT_LOGGER):
            patch.setattr(ollama_memory, 'load_model', failing_load)
            memory.finish_loading(ollama(fake), MODEL, REQUEST_NUMBER)
            [task] = memory._finishing.values()
            await wait_until_none_is_finished(memory)
        assert task.exception() is None  # swallowed, so asyncio has nothing to complain about
        assert len(caplog.records) == 1 and caplog.records[0].levelno == logging.ERROR
        assert caplog.records[0].exc_info[1].args == (UNEXPECTED_FAILURE,)
        memory.finish_loading(ollama(fake), MODEL, REQUEST_NUMBER)  # and the next one loads
        await wait_until_none_is_finished(memory)
    assert sent(fake, GENERATE) == [KEEP_MODEL_LOADED]


async def test_closing_drops_a_load_to_finish_also_one_that_never_ran(http, caplog):
    memory = ModelMemory(http)
    async with FakeBackend(more_models=[SECOND_MODEL]) as fake:
        fake.hold_generate.clear()
        with caplog.at_level(logging.WARNING, logger=CLIENT_LOGGER):
            memory.finish_loading(ollama(fake), MODEL, REQUEST_NUMBER)
            await wait_for_requests(fake, GENERATE, 1)
            memory.finish_loading(ollama(fake), SECOND_MODEL, REQUEST_NUMBER)  # waits for its turn
            memory.finish_loading(ollama(fake, name='other'), MODEL, REQUEST_NUMBER)  # has not even started
            await asyncio.wait_for(memory.close(), SETTLE_TIMEOUT_SECONDS)
            await asyncio.sleep(QUIET_SECONDS)
        # Ollama had not answered the one load that was sent; the others were never sent
        assert memory._finishing == {} and fake.in_memory == {} and len(sent(fake, GENERATE)) == 1
        assert caplog.records == []
        fake.hold_generate.set()
        memory.finish_loading(ollama(fake), SECOND_MODEL, REQUEST_NUMBER)  # nothing is left in the way of a later one
        await wait_until_none_is_finished(memory)
    assert sent(fake, GENERATE)[1:] == [{'model': SECOND_MODEL, 'keep_alive': KEEP_LOADED}]


async def test_a_load_waits_for_ollama_as_long_as_the_gateway_waits_for_a_model_if_that_is_longer(http):
    """Those at start and those the client finishes alike; without a word of the gateway, the backend's time."""
    async def answering_post(url, **kwargs):
        return httpx.Response(HTTP_OK, json={}, request=httpx.Request('POST', url))
    http.post = answering_post
    timeouts = ManualTimeouts()
    memory = ModelMemory(http, timeout=timeouts)
    async with FakeBackend() as fake:
        backend = ollama(fake, load_at_start=True, read_timeout=LOAD_SECONDS)
        await memory.load_at_start([backend])
        memory.gateway_waits_for_loading(LONG_READ_TIMEOUT_SECONDS)
        await memory.load_at_start([backend])
        memory.finish_loading(backend, MODEL, REQUEST_NUMBER)
        await wait_until_none_is_finished(memory)
        memory.gateway_waits_for_loading(None)
        await memory.load_at_start([backend])
    assert timeouts.asked == [LOAD_SECONDS, LONG_READ_TIMEOUT_SECONDS, LONG_READ_TIMEOUT_SECONDS, LOAD_SECONDS]


async def test_a_load_that_ran_out_of_time_names_the_wait_it_ran_with_not_a_later_one(http, caplog):
    """What the gateway waits may change with a reconnect while the load is on its way."""
    asked = asyncio.Event()

    async def post_that_never_answers(url, **kwargs):
        asked.set()
        await asyncio.Event().wait()
    http.post = post_that_never_answers
    timeouts = ManualTimeouts()
    memory = ModelMemory(http, timeout=timeouts)
    async with FakeBackend() as fake:
        memory.gateway_waits_for_loading(LONG_READ_TIMEOUT_SECONDS)
        with caplog.at_level(logging.WARNING, logger=CLIENT_LOGGER):
            memory.finish_loading(ollama(fake, read_timeout=LOAD_SECONDS), MODEL, REQUEST_NUMBER)
            await asyncio.wait_for(asked.wait(), SETTLE_TIMEOUT_SECONDS)
            memory.gateway_waits_for_loading(None)
            await timeouts.run_out(LONG_READ_TIMEOUT_SECONDS)
            await wait_until_none_is_finished(memory)
    assert f'did not answer within {LONG_READ_TIMEOUT_SECONDS:g} s' in caplog.messages[0]


# ----- loading at start -----

async def test_the_models_listed_are_loaded_one_after_another_and_the_log_says_how_long_each_took(http, caplog):
    memory = ModelMemory(http, clock=ticking_clock(LOAD_SECONDS))
    async with FakeBackend(more_models=[SECOND_MODEL]) as fake:
        backend = ollama(fake, models=(MODEL, SECOND_MODEL), load_at_start=True)
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            await memory.load_at_start([backend])
    assert sent(fake, GENERATE) == [{'model': MODEL, 'keep_alive': KEEP_LOADED},
                                    {'model': SECOND_MODEL, 'keep_alive': KEEP_LOADED}]
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == 'Backend ollama: loading 2 model(s) into memory, to stay for 30m after their last request'
    assert f'Backend ollama: {MODEL} is in memory after {LOAD_SECONDS:.1f} s' in messages
    assert f'Backend ollama: {SECOND_MODEL} is in memory after {LOAD_SECONDS:.1f} s' in messages
    assert messages[-1] == f'Backend ollama: in memory now: {MODEL}, {SECOND_MODEL}'
    assert all(record.levelno == logging.INFO for record in caplog.records)


async def test_only_backends_that_ask_for_it_load_at_start(memory):
    async with FakeBackend() as fake:
        await memory.load_at_start([ollama(fake), ollama(fake, keep_loaded=None)])
    assert fake.requests == []


async def test_without_a_time_the_models_are_loaded_for_as_long_as_ollama_keeps_them(memory, caplog):
    async with FakeBackend() as fake:
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            await memory.load_at_start([ollama(fake, keep_loaded=None, load_at_start=True)])
    assert sent(fake, GENERATE) == [{'model': MODEL}]
    assert 'as long as Ollama itself keeps them' in caplog.records[0].getMessage()


async def test_a_model_that_does_not_fit_or_is_not_there_is_a_warning_and_the_others_are_loaded(memory, caplog):
    async with FakeBackend(models_that_do_not_fit=[TOO_LARGE_MODEL]) as fake:
        backend = ollama(fake, models=(TOO_LARGE_MODEL, NOT_PULLED_MODEL, MODEL), load_at_start=True)
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            await memory.load_at_start([backend])
    assert fake.in_memory == {MODEL: KEEP_LOADED}
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert f'Backend ollama: {TOO_LARGE_MODEL} could not be loaded' in warnings[0]
    assert 'requires more system memory (12.3 GiB) than is available (7.9 GiB)' in warnings[0]
    assert f"model '{NOT_PULLED_MODEL}' not found" in warnings[1]
    assert caplog.records[-1].getMessage() == f'Backend ollama: in memory now: {MODEL}'


async def test_an_ollama_that_is_down_at_start_is_a_warning_per_backend_not_per_model(memory, caplog):
    async with FakeBackend(more_models=[SECOND_MODEL]) as fake:
        gone = ollama(fake, models=(MODEL, SECOND_MODEL), load_at_start=True)
    with caplog.at_level(logging.WARNING):
        await memory.load_at_start([gone])
    assert len(caplog.records) == 1
    assert 'Backend ollama: no model could be loaded' in caplog.records[0].getMessage()


async def test_more_models_than_ollama_keeps_by_default_are_loaded_with_a_word_about_its_limit(memory, caplog):
    names = [f'model-{number}' for number in range(OLLAMA_DEFAULT_MAX_LOADED_MODELS + 1)]
    async with FakeBackend(more_models=names) as fake:
        with caplog.at_level(logging.INFO, logger=CLIENT_LOGGER):
            await memory.load_at_start([ollama(fake, models=tuple(names), load_at_start=True)])
            assert 'OLLAMA_MAX_LOADED_MODELS' in caplog.text
            caplog.clear()
            await memory.load_at_start([ollama(fake, models=tuple(names[:OLLAMA_DEFAULT_MAX_LOADED_MODELS]),
                                               load_at_start=True)])
            assert 'OLLAMA_MAX_LOADED_MODELS' not in caplog.text
    assert len(fake.in_memory) == len(names)


@pytest.mark.parametrize('value, words', [
    ('30m', 'for 30m'),
    (3600, 'for 3600 seconds'),
    (9223372036, 'for 9223372036 seconds'),
    (1.5, 'for 1.5 seconds'),
    (-1, 'until Ollama stops'),
    ('-1m', 'until Ollama stops'),
    (0, 'not at all'),
    ('0s', 'not at all'),
    (None, 'as long as Ollama itself keeps them'),
])
def test_a_time_in_words(value, words):
    assert words in keep_loaded_in_words(value)
