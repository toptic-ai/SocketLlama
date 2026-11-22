"""The models report is trimmed to what the gateway accepts before it is sent."""
import json
import logging

from socketllama import protocol
from socketllama.backends import MODELS_REPORT_MAX_BYTES, fit_report_to_gateway

# Printable but not ASCII: json.dumps escapes every such character to six (\u00fc), so names
# within the length limit can still make a report at the count limits larger than a frame
LONG_NAME_PADDING = '\u00fc' * (protocol.MODEL_NAME_MAX_LENGTH - len('model-0000-0000'))
SMALL_LIMIT = 2


def backend_entry(name: str, model_names) -> dict:
    return {'name': name, 'kind': 'openai', 'models': [{'name': model} for model in model_names]}


def frame_size(report: list) -> int:
    return len(json.dumps({'type': protocol.FRAME_MODELS, 'backends': report}))


def test_a_report_within_the_limits_is_unchanged(caplog):
    report = [backend_entry('ollama', ['a', 'b'])]
    with caplog.at_level(logging.WARNING):
        assert fit_report_to_gateway(report) == report
    assert caplog.text == ''


def test_backends_beyond_the_limit_are_dropped_and_named(monkeypatch, caplog):
    monkeypatch.setattr(protocol, 'MAX_BACKENDS_PER_REPORT', SMALL_LIMIT)
    report = [backend_entry(f'b{n}', [f'm{n}']) for n in range(SMALL_LIMIT + 1)]
    fitted = fit_report_to_gateway(report)
    assert [backend['name'] for backend in fitted] == ['b0', 'b1']
    assert 'b2' in caplog.text and 'm2' in caplog.text


def test_models_beyond_the_per_backend_limit_are_dropped_and_named(monkeypatch, caplog):
    monkeypatch.setattr(protocol, 'MAX_MODELS_PER_BACKEND', SMALL_LIMIT)
    fitted = fit_report_to_gateway([backend_entry('proxy', ['m0', 'm1', 'm2'])])
    assert [model['name'] for model in fitted[0]['models']] == ['m0', 'm1']
    assert 'proxy' in caplog.text and 'm2' in caplog.text


def test_names_the_gateway_would_refuse_are_dropped(caplog):
    too_long = 'n' * (protocol.MODEL_NAME_MAX_LENGTH + 1)
    fitted = fit_report_to_gateway([backend_entry('proxy', ['fine', too_long, 'tab\there'])])
    assert [model['name'] for model in fitted[0]['models']] == ['fine']
    assert 'proxy' in caplog.text


def test_the_largest_report_stays_safely_below_the_gateway_frame_limit(caplog):
    report = [backend_entry(f'backend-{b}', [f'model-{b:04}-{m:04}{LONG_NAME_PADDING}'
                                             for m in range(protocol.MAX_MODELS_PER_BACKEND)])
              for b in range(protocol.MAX_BACKENDS_PER_REPORT)]
    assert frame_size(report) > protocol.MAX_GATEWAY_INBOUND_FRAME_BYTES
    fitted = fit_report_to_gateway(report)
    assert frame_size(fitted) <= MODELS_REPORT_MAX_BYTES < protocol.MAX_GATEWAY_INBOUND_FRAME_BYTES
    # dropped from the end: the first backends are reported completely
    assert fitted[0]['models'] == report[0]['models']
    assert sum(len(backend['models']) for backend in fitted) < sum(len(backend['models']) for backend in report)
    assert 'size limit' in caplog.text
