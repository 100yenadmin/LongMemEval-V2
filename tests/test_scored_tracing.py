import asyncio
from argparse import Namespace
from types import SimpleNamespace

from evaluation import harness, qa_eval_metrics


class _AsyncCompletions:
    def __init__(self, response):
        self._response = response
        self.request = None

    async def create(self, **request):
        self.request = request
        return self._response


class _SyncCompletions:
    def __init__(self, response):
        self._response = response
        self.request = None

    def create(self, **request):
        self.request = request
        return self._response


def _response(*, text, model, prompt_tokens, completion_tokens, response_id="resp-fixture"):
    return SimpleNamespace(
        id=response_id,
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=text, reasoning=None),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def test_reader_trace_records_actual_transport_without_messages_or_secrets():
    response = _response(
        text="\\boxed{fixture}",
        model="Qwen/Qwen3.5-9B",
        prompt_tokens=41,
        completion_tokens=7,
    )
    completions = _AsyncCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    args = Namespace(
        model="Qwen/Qwen3.5-9B",
        base_url="http://127.0.0.1:8023/v1",
        max_completion_tokens=20000,
        reasoning_effort=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=None,
        presence_penalty=None,
        reader_enable_thinking=True,
        timeout_seconds=30.0,
    )

    text, usage, trace = asyncio.run(
        harness.call_reader_model_async(
            client,
            args,
            [{"role": "user", "content": "synthetic fixture"}],
        )
    )

    assert text == "\\boxed{fixture}"
    assert usage == {"prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 48}
    assert trace["kind"] == "model"
    assert trace["transport"] == "openai_compatible_chat_completions"
    assert trace["requested_model"] == "Qwen/Qwen3.5-9B"
    assert trace["actual_model"] == "Qwen/Qwen3.5-9B"
    assert trace["finish_reason"] == "stop"
    assert trace["latency_seconds"] >= 0
    assert trace["usage"] == usage
    serialized = repr(trace).lower()
    assert "synthetic fixture" not in serialized
    assert "api_key" not in serialized
    assert "messages" not in trace


def test_judge_trace_records_raw_result_usage_and_latency():
    response = _response(
        text='{"label": 1, "reason": "synthetic match"}',
        model="gpt-5.2-2026-07-01",
        prompt_tokens=53,
        completion_tokens=11,
    )
    completions = _SyncCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    trace = {}

    text = qa_eval_metrics._call_chat_completion(
        client=client,
        model="gpt-5.2",
        messages=[{"role": "user", "content": "synthetic judge fixture"}],
        max_completion_tokens=2048,
        reasoning_effort="medium",
        temperature=None,
        top_p=None,
        timeout_seconds=30.0,
        trace_out=trace,
    )

    assert text == '{"label": 1, "reason": "synthetic match"}'
    assert trace["transport"] == "openai_chat_completions"
    assert trace["requested_model"] == "gpt-5.2"
    assert trace["actual_model"] == "gpt-5.2-2026-07-01"
    assert trace["reasoning_effort"] == "medium"
    assert trace["finish_reason"] == "stop"
    assert trace["latency_seconds"] >= 0
    assert trace["usage"] == {
        "prompt_tokens": 53,
        "completion_tokens": 11,
        "total_tokens": 64,
    }
    assert trace["judge_output"] == text
    assert "synthetic judge fixture" not in repr(trace)


def test_score_prediction_emits_deterministic_stage_trace(monkeypatch):
    monkeypatch.setattr(harness, "eval_from_spec", lambda *_args, **_kwargs: True)
    row = {
        "eval_name": "norm_phrase_set_match",
        "eval_function": "norm_phrase_set_match",
        "response_parsed_boxed": "fixture",
        "response_raw": "\\boxed{fixture}",
        "answer_gold": "fixture",
        "is_unknown": False,
    }

    score, eval_name, is_unknown, trace = harness.score_prediction(row, {})

    assert score is True
    assert eval_name == "norm_phrase_set_match"
    assert is_unknown is False
    assert trace["kind"] == "deterministic"
    assert trace["eval_name"] == "norm_phrase_set_match"
    assert trace["score_bool"] is True
    assert trace["latency_seconds"] >= 0
    assert trace["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_score_prediction_passes_llm_trace_sink_without_changing_score(monkeypatch):
    def fake_eval(_spec, _prediction, _answer, **kwargs):
        trace = kwargs["evaluator_trace"]
        trace.update(
            {
                "kind": "model",
                "requested_model": kwargs["evaluator_model"],
                "actual_model": "gpt-5.2-fixture",
                "judge_output": '{"label": 1}',
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        )
        return True

    monkeypatch.setattr(harness, "eval_from_spec", fake_eval)
    row = {
        "eval_name": "llm_gotchas_checker",
        "eval_function": "llm_gotchas_checker",
        "question_item": {"question": {"text": "synthetic"}},
        "response_parsed_boxed": "fixture",
        "response_raw": "\\boxed{fixture}",
        "answer_gold": "fixture",
        "is_unknown": False,
    }
    config = {
        "evaluator_model": "gpt-5.2",
        "evaluator_base_url": None,
        "evaluator_api_key": "test-only",
        "evaluator_reasoning_effort": "medium",
        "evaluator_max_completion_tokens": 2048,
        "evaluator_timeout_seconds": 30.0,
    }

    score, _eval_name, _is_unknown, trace = harness.score_prediction(row, config)

    assert score is True
    assert trace["kind"] == "model"
    assert trace["actual_model"] == "gpt-5.2-fixture"
    assert trace["score_bool"] is True


def test_trace_aggregation_separates_model_and_deterministic_stages():
    records = [
        {
            "judge_trace": {
                "kind": "model",
                "actual_model": "gpt-5.2-fixture",
                "latency_seconds": 1.5,
                "score_latency_seconds": 1.6,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }
        },
        {
            "judge_trace": {
                "kind": "deterministic",
                "actual_model": None,
                "latency_seconds": 0.01,
                "score_latency_seconds": 0.01,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        },
    ]

    summary = harness.aggregate_trace_metrics(
        records,
        trace_key="judge_trace",
        stage_latency_key="score_latency_seconds",
    )

    assert summary["record_count"] == 2
    assert summary["model_call_count"] == 1
    assert summary["deterministic_call_count"] == 1
    assert summary["actual_models"] == {"gpt-5.2-fixture": 1}
    assert summary["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
    assert summary["latency_seconds"]["avg"] == 0.805
    assert summary["latency_seconds"]["max"] == 1.6


def test_reader_stage_record_preserves_evidence_without_encoded_request_or_secret():
    row = {
        "index": 3,
        "stream_index": 4,
        "question_id": "synthetic-q",
        "question_type": "static-environment",
        "category": "static",
        "is_abstention_problem": False,
        "eval_function": "norm_phrase_set_match",
        "eval_name": "norm_phrase_set_match",
        "question_item": {"question": {"text": "Synthetic question"}},
        "question_text": "Synthetic question",
        "question_image": None,
        "haystack_ids": ["trajectory-a"],
        "memory_context": [{"type": "text", "value": "exact evidence"}],
        "memory_query_duration_seconds": 0.4,
        "memory_post_query_duration_seconds": 0.1,
        "memory_post_query_metadata": {"exact_refs": ["trajectory-a:state:2"]},
        "memory_context_original_token_count": 12,
        "memory_context_token_count": 12,
        "memory_context_was_truncated": False,
        "prompt_messages": [{"role": "user", "content": "Synthetic question"}],
        "answer_gold": "fixture",
        "messages": [{"role": "user", "content": "data:image/png;base64,secretish"}],
        "api_key": "must-not-persist",
    }
    output = {
        "response_raw": "\\boxed{fixture}",
        "response_parsed_boxed": "fixture",
        "is_unknown": False,
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        "reader_trace": {"kind": "model", "actual_model": "Qwen/Qwen3.5-9B"},
    }

    record = harness.build_reader_stage_record(row, output)

    assert record["memory_post_query_metadata"]["exact_refs"] == [
        "trajectory-a:state:2"
    ]
    assert record["response_parsed_boxed"] == "fixture"
    assert record["reader_trace"]["actual_model"] == "Qwen/Qwen3.5-9B"
    serialized = repr(record)
    assert "data:image/png;base64" not in serialized
    assert "must-not-persist" not in serialized
    assert "messages" not in record
    assert "api_key" not in record
