"""Result is the whole measurement logic; drive it with hand-made payloads and fake clocks."""

import json

from bench.stream import Result, short_error, sse_data


def payload(delta=None, finish=None, usage=None):
    return json.dumps({"choices": [{"delta": delta or {}, "finish_reason": finish}], "usage": usage})


def test_sse_data_filters_non_data_lines():
    assert sse_data("") is None
    assert sse_data(": heartbeat") is None
    assert sse_data("event: foo") is None
    assert sse_data("data: {}") == "{}"
    assert sse_data("data: [DONE]") == "[DONE]"


def test_happy_path_measures_ttft_ttfc_and_throughput():
    r = Result("m")
    assert r.feed(100, payload({"reasoning": "hmm"})) == {"reasoning": "hmm"}
    assert r.feed(300, payload({"content": "Hi"})) == {"content": "Hi"}
    assert r.feed(400, payload({"content": "!"}, finish="stop")) == {"content": "!"}
    assert (
        r.feed(
            450,
            payload(
                usage={
                    "prompt_tokens": 3,
                    "completion_tokens": 10,
                    "completion_tokens_details": {"reasoning_tokens": 4},
                }
            ),
        )
        is None
    )
    assert r.feed(500, "[DONE]") is None
    r.close(600)

    assert r.status == "ok" and r.finished and r.stream_complete
    assert (r.ttft_ms, r.ttfc_ms, r.total_ms) == (100, 300, 600)
    assert (r.content, r.reasoning, r.finish_reason) == ("Hi!", "hmm", "stop")
    assert (r.prompt_tokens, r.completion_tokens, r.reasoning_tokens) == (3, 10, 4)
    assert r.tokens_per_s == 20.0  # 10 tokens over (600-100) ms
    assert not r.tokens_estimated


def test_missing_usage_falls_back_to_chunk_count():
    r = Result("m")
    for t, w in enumerate(["a", "b", "c"]):
        r.feed(t * 10, payload({"content": w}))
    r.feed(40, "[DONE]")
    r.close(50)
    assert r.completion_tokens == 3 and r.tokens_estimated


def test_stream_without_done_is_an_error():
    r = Result("m")
    r.feed(10, payload({"content": "partial"}))
    r.close(20)
    assert r.status == "error" and "[DONE]" in r.error
    assert r.content == "partial"  # what arrived is kept


def test_in_stream_error_event_fails_and_finishes():
    r = Result("m")
    r.feed(10, payload({"content": "x"}))
    r.feed(20, json.dumps({"status_code": 500, "error": {"type": "InternalServerError", "message": "boom"}}))
    assert r.finished
    r.close(30)
    assert r.status == "error" and r.error == "HTTP 500 · InternalServerError · boom"


def test_garbage_payload_is_ignored():
    r = Result("m")
    assert r.feed(1, "not json") is None
    assert r.status == "ok"


def test_short_error_handles_gateway_and_backend_shapes():
    gateway = {"type": "virtual_key_not_found", "error": {"message": "no key"}}
    backend = {"error": {"type": "BadRequestError", "param": "temperature", "message": "out of range"}}
    assert short_error(gateway, 401) == "HTTP 401 · virtual_key_not_found · no key"
    assert short_error(backend, 400) == "HTTP 400 · BadRequestError · out of range"
    assert short_error("<html>nope</html>", 502) == "HTTP 502: <html>nope</html>"


def test_tool_call_fragments_are_merged_by_index():
    r = Result("m")
    first = {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]}
    out = r.feed(120, payload(first))
    assert out == {"tool_call": {"index": 0, "name": "get_weather", "arguments": ""}}
    assert r.ttft_ms == 120 and r.ttfc_ms is None
    r.feed(150, payload({"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]}))
    r.feed(160, payload({"tool_calls": [{"index": 1, "id": "call_2", "function": {"name": "get_weather"}}]}))
    r.feed(170, payload({"tool_calls": [{"index": 0, "function": {"arguments": '"Rome"}'}}]}))
    r.feed(
        180, payload({"tool_calls": [{"index": 1, "function": {"arguments": '{"city": "Oslo"}'}}]}, finish="tool_calls")
    )
    r.feed(200, "[DONE]")
    r.close(210)
    assert r.status == "ok" and r.finish_reason == "tool_calls"
    assert [(c["id"], c["name"], json.loads(c["arguments"])) for c in r.tool_calls] == [
        ("call_1", "get_weather", {"city": "Rome"}),
        ("call_2", "get_weather", {"city": "Oslo"}),
    ]
    assert r.content == "" and r.chunks == 5


def test_plain_answer_has_no_tool_calls_and_no_steps():
    r = Result("m")
    r.feed(10, payload({"content": "hi"}, finish="stop"))
    assert r.tool_calls == [] and r.steps is None
