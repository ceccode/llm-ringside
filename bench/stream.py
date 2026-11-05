"""The measurable core: SSE payloads in, a Result out. No I/O, no clocks.

The runner feeds this with (elapsed_ms, payload) pairs as they arrive on the
wire; tests feed it with lists. Everything we report about a response is
decided here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

DONE = "[DONE]"


def sse_data(line: str) -> str | None:
    """Payload of a `data:` line, or None for blank lines, comments and other fields."""
    if not line or line.startswith(":") or not line.startswith("data:"):
        return None
    return line[5:].strip()


@dataclass
class Result:
    model_id: str
    rep: int = 0
    status: str = "ok"  # ok | error
    error: str | None = None
    http_status: int | None = None
    request_id: str | None = None
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    stream_complete: bool = False  # saw [DONE]
    ttft_ms: float | None = None  # first delta of any kind (reasoning or content)
    ttfc_ms: float | None = None  # first content delta
    total_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    tokens_estimated: bool = False  # no usage from provider: chunk count used
    chunks: int = 0
    tokens_per_s: float | None = None

    # -- building -----------------------------------------------------------

    def fail(self, error: str) -> None:
        self.status, self.error = "error", error

    @property
    def finished(self) -> bool:
        return self.stream_complete or self.status == "error"

    def feed(self, elapsed_ms: float, data: str) -> dict | None:
        """Consume one SSE payload. Returns the delta to forward to a UI, if any."""
        if data == DONE:
            self.stream_complete = True
            return None
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return None
        if "error" in obj and "choices" not in obj:
            self.fail(short_error(obj, obj.get("status_code")))
            return None
        if usage := obj.get("usage"):
            self.prompt_tokens = usage.get("prompt_tokens")
            self.completion_tokens = usage.get("completion_tokens")
            self.reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

        delta_out: dict = {}
        for choice in obj.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if reasoning := delta.get("reasoning") or delta.get("reasoning_content"):
                self._first_token(elapsed_ms)
                self.reasoning += reasoning
                self.chunks += 1
                delta_out["reasoning"] = delta_out.get("reasoning", "") + reasoning
            if content := delta.get("content"):
                self._first_token(elapsed_ms)
                if self.ttfc_ms is None:
                    self.ttfc_ms = elapsed_ms
                self.content += content
                self.chunks += 1
                delta_out["content"] = delta_out.get("content", "") + content
        return delta_out or None

    def close(self, elapsed_ms: float) -> None:
        """Stream is over (cleanly or not): derive the final numbers."""
        self.total_ms = elapsed_ms
        if self.status == "ok" and not self.stream_complete:
            self.fail("stream ended without [DONE]")
        if self.completion_tokens is None and self.chunks:
            self.completion_tokens, self.tokens_estimated = self.chunks, True
        if self.completion_tokens and self.ttft_ms is not None:
            gen_s = max((self.total_ms - self.ttft_ms) / 1000, 1e-3)
            self.tokens_per_s = round(self.completion_tokens / gen_s, 1)

    def _first_token(self, elapsed_ms: float) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = elapsed_ms

    def to_dict(self) -> dict:
        return asdict(self)


def short_error(body: dict | str, status: int | None) -> str:
    """One line out of the two error shapes providers return (gateway and backend)."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return f"HTTP {status}: {body[:300]}"
    err = body.get("error")
    msg = err.get("message") if isinstance(err, dict) else err
    typ = body.get("type") or (err.get("type") if isinstance(err, dict) else None)
    parts = [f"HTTP {status}" if status else None, typ, msg]
    return " · ".join(str(p) for p in parts if p)[:500]
