"""Multi-turn episodes against the fake provider: tool call, mocked result, final answer."""

import asyncio

from bench.episode import run_episode
from bench.tasks import Expect, Task, Tool

FIND = Tool(
    "find_customer",
    "Look up a customer.",
    {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
    [{"when": {"email": "mario@example.com"}, "return": {"customer_id": "C-42"}}],
)


def task(max_steps=4) -> Task:
    expect = Expect(
        calls=[{"name": "find_customer", "args": {"email": "mario@example.com"}}], final_contains=["shipped"]
    )
    return Task("t", "Status of Mario's order?", [FIND], expect, max_steps=max_steps)


def run(models, http_client, model_id, **kw):
    return asyncio.run(run_episode(models[model_id], task(**kw), client=http_client))


def test_episode_calls_tool_then_answers(models, http_client):
    events = []

    async def go():
        async def emit(ev):
            events.append(ev)

        return await run_episode(models["fake/agent"], task(), client=http_client, emit=emit)

    r = asyncio.run(go())
    assert r.status == "ok" and r.steps == 2 and r.finish_reason == "stop"
    assert r.content == "Order O-1017 has shipped."
    assert r.tool_calls == [
        {
            "step": 1,
            "name": "find_customer",
            "arguments": '{"email": "mario@example.com"}',
            "args": {"email": "mario@example.com"},
            "valid": True,
            "result": '{"customer_id": "C-42"}',
        }
    ]
    assert r.completion_tokens == 8 + 5 and r.prompt_tokens == 20 + 12
    assert r.ttft_ms is not None and r.total_ms >= r.ttft_ms and r.tokens_per_s
    assert r.check_score == 1.0
    tool_events = [e for e in events if e["type"] == "tool"]
    assert len(tool_events) == 1 and tool_events[0]["model_id"] == "fake/agent"
    assert any(e["type"] == "delta" and "tool_call" in e for e in events)


def test_episode_stops_at_max_steps(models, http_client):
    r = run(models, http_client, "fake/agent", max_steps=1)
    assert r.status == "ok" and r.steps == 1 and r.finish_reason == "max_steps" and r.content == ""
    assert r.check_score < 1
    assert {"ok": False, "label": "finished within 1 steps"} in r.check_detail


def test_failed_turn_fails_the_episode_without_a_score(models, http_client):
    r = run(models, http_client, "fake/agentfail")
    assert r.status == "error" and r.error.startswith("turn 2: HTTP 429")
    assert r.steps == 2 and len(r.tool_calls) == 1 and r.check_score is None


def test_model_that_answers_directly_is_one_step(models, http_client):
    r = run(models, http_client, "fake/ok")
    assert r.status == "ok" and r.steps == 1 and r.tool_calls == [] and r.content == "Hello world!"
    assert {"ok": False, "label": "calls find_customer(email=mario@example.com)"} in r.check_detail
