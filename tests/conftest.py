"""A fake OpenAI-compatible provider served in-process through httpx's ASGI transport.

The model name picks the behaviour, so a test reads like a scenario:
  ok | nousage | nodone | inerror | http429 | missingkey
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from bench.config import ModelConfig, Provider

SCENARIOS = ["ok", "nousage", "nodone", "inerror", "http429"]


def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
    obj = {
        "id": "x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "usage": usage,
    }
    return f"data: {json.dumps(obj)}\n\n"


def fake_provider() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        mode = body["model"]
        if mode == "http429":
            return JSONResponse(
                {"type": "request_limited", "status_code": 429, "error": {"message": "slow down"}}, status_code=429
            )

        async def gen():
            yield ": heartbeat\n\n"
            if mode == "ok":
                for w in ["Let me ", "think."]:
                    yield chunk({"reasoning": w})
            for i, w in enumerate(["Hello", " world", "!"]):
                if mode == "inerror" and i == 2:
                    yield (
                        "data: "
                        + json.dumps({"status_code": 500, "error": {"type": "InternalServerError", "message": "boom"}})
                        + "\n\n"
                    )
                    return
                yield chunk({"content": w})
            if mode == "nodone":
                return
            yield chunk({}, finish="stop")
            if mode != "nousage":
                yield chunk(
                    {},
                    usage={
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "total_tokens": 17,
                    },
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"X-Request-Id": "req-1"})

    return app


@pytest.fixture
def provider(monkeypatch) -> Provider:
    monkeypatch.setenv("FAKE_API_KEY", "test-key")
    return Provider(name="fake", base_url="http://fake/v1", api_key_env="FAKE_API_KEY")


@pytest.fixture
def models(provider) -> dict[str, ModelConfig]:
    ms = {f"fake/{s}": ModelConfig(id=f"fake/{s}", provider=provider, model=s) for s in SCENARIOS}
    ms["fake/missingkey"] = ModelConfig(
        id="fake/missingkey", provider=Provider("nokey", "http://fake/v1", "NO_SUCH_KEY"), model="ok"
    )
    os.environ.pop("NO_SUCH_KEY", None)
    return ms


@pytest.fixture
def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_provider()))
