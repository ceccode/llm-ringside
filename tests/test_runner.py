"""run_model against the in-process fake provider: real HTTP semantics, no network."""

import asyncio

from bench.runner import build_request, run_model


def run(models, client, scenario, **kw):
    return asyncio.run(run_model(models[f"fake/{scenario}"], [{"role": "user", "content": "hi"}], client=client, **kw))


def test_build_request_merges_provider_extras(models):
    mc = models["fake/ok"]
    mc.extra = {"chat_template_kwargs": {"enable_thinking": False}}
    body = build_request(mc, [{"role": "user", "content": "hi"}], max_tokens=64, temperature=None)
    assert body["model"] == "ok" and body["stream"] is True and body["max_tokens"] == 64
    assert "temperature" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_ok_scenario_collects_everything(models, http_client):
    r = run(models, http_client, "ok")
    assert r.status == "ok" and r.finish_reason == "stop"
    assert r.content == "Hello world!" and r.reasoning == "Let me think."
    assert (r.http_status, r.request_id) == (200, "req-1")
    assert (r.prompt_tokens, r.completion_tokens, r.reasoning_tokens) == (12, 5, 2)
    assert r.ttft_ms is not None and r.ttft_ms <= r.ttfc_ms <= r.total_ms


def test_deltas_are_emitted_in_order(models, http_client):
    seen = []

    async def emit(ev):
        seen.append(ev)

    asyncio.run(run_model(models["fake/ok"], [{"role": "user", "content": "hi"}], client=http_client, emit=emit, rep=2))
    assert all(ev["type"] == "delta" and ev["rep"] == 2 for ev in seen)
    assert "".join(ev.get("content", "") for ev in seen) == "Hello world!"
    assert "".join(ev.get("reasoning", "") for ev in seen) == "Let me think."


def test_http_error_before_stream(models, http_client):
    r = run(models, http_client, "http429")
    assert r.status == "error" and r.http_status == 429
    assert r.error == "HTTP 429 · request_limited · slow down"


def test_error_inside_stream(models, http_client):
    r = run(models, http_client, "inerror")
    assert r.status == "error" and r.content == "Hello world" and "boom" in r.error


def test_stream_closed_without_done(models, http_client):
    r = run(models, http_client, "nodone")
    assert r.status == "error" and not r.stream_complete


def test_missing_key_never_hits_the_network(models, http_client):
    r = run(models, http_client, "missingkey")
    assert r.status == "error" and r.error == "missing NO_SUCH_KEY" and r.http_status is None
