"""The I/O edge: build the request, stream it over httpx, time it, hand bytes to Result."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from .config import ModelConfig
from .stream import Result, short_error, sse_data

Emit = Callable[[dict], Awaitable[None]]


def build_request(
    mc: ModelConfig, messages: list[dict], *, max_tokens: int, temperature: float | None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": mc.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    body.update(mc.extra)
    return body


async def run_model(
    mc: ModelConfig,
    messages: list[dict],
    *,
    max_tokens: int = 1024,
    temperature: float | None = None,
    timeout_s: float = 180.0,
    rep: int = 0,
    emit: Emit | None = None,
    client: httpx.AsyncClient | None = None,
) -> Result:
    """One streamed chat completion, measured. `client` is injectable for tests."""
    result = Result(model_id=mc.id, rep=rep)
    key = mc.provider.api_key
    if not key:
        result.fail(f"missing {mc.provider.api_key_env}")
        result.close(0)
        return result

    body = build_request(mc, messages, max_tokens=max_tokens, temperature=temperature)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    url = f"{mc.provider.base_url}/chat/completions"
    timeout = httpx.Timeout(timeout_s, connect=15)

    own_client = client is None
    client = client or httpx.AsyncClient()
    t0 = time.perf_counter()

    def elapsed_ms() -> float:
        return (time.perf_counter() - t0) * 1000

    try:
        async with client.stream("POST", url, json=body, headers=headers, timeout=timeout) as resp:
            result.http_status = resp.status_code
            result.request_id = resp.headers.get("x-request-id")
            if resp.status_code != 200:
                raw = (await resp.aread()).decode(errors="replace")
                result.fail(short_error(raw, resp.status_code))
            else:
                async for line in resp.aiter_lines():
                    if (data := sse_data(line)) is None:
                        continue
                    delta = result.feed(elapsed_ms(), data)
                    if delta and emit:
                        await emit({"type": "delta", "model_id": mc.id, "rep": rep, **delta})
                    if result.finished:
                        break
    except httpx.TimeoutException:
        result.fail(f"timeout after {timeout_s:.0f}s")
    except httpx.HTTPError as e:
        result.fail(f"http error: {e}")
    finally:
        result.close(elapsed_ms())
        if own_client:
            await client.aclose()
    return result
