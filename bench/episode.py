"""The agentic I/O edge: run a task as a multi-turn conversation with mocked tools.

Each model turn is an ordinary `run_model` call. When the model asks for tools we answer from the
task's canned responses (tasks.mock_call) and go round again, until it answers in plain text or
`max_steps` turns are spent. The episode is reported as one Result, then scored by tasks.score.
"""

from __future__ import annotations

import time

import httpx

from .config import ModelConfig
from .runner import Emit, run_model
from .stream import Result
from .tasks import Task, mock_call, openai_tools, score


def task_messages(task: Task) -> list[dict]:
    system = [{"role": "system", "content": task.system}] if task.system else []
    return system + [{"role": "user", "content": task.prompt}]


def _add(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else a + b


async def run_episode(
    mc: ModelConfig,
    task: Task,
    *,
    max_tokens: int = 1024,
    temperature: float | None = None,
    timeout_s: float = 180.0,
    rep: int = 0,
    emit: Emit | None = None,
    client: httpx.AsyncClient | None = None,
) -> Result:
    """TTFT is the first token of the first turn; total and tok/s span the whole episode."""
    ep = Result(model_id=mc.id, rep=rep, steps=0, stream_complete=True)
    messages, tools, trace = task_messages(task), openai_tools(task), []
    hit_max_steps = False
    t0 = time.perf_counter()

    for step in range(1, task.max_steps + 1):
        offset = (time.perf_counter() - t0) * 1000
        r = await run_model(
            mc,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            rep=rep,
            emit=emit,
            client=client,
            tools=tools,
        )
        ep.steps = step
        _absorb(ep, r, offset)
        if r.status == "error":
            ep.fail(f"turn {step}: {r.error}")
            break
        if not r.tool_calls:
            ep.content, ep.finish_reason = r.content, r.finish_reason
            break

        calls = [{**c, "id": c["id"] or f"call_{step}_{c['index']}"} for c in r.tool_calls]
        messages.append(
            {
                "role": "assistant",
                "content": r.content or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    }
                    for c in calls
                ],
            }
        )
        for c in calls:
            out = mock_call(task, c["name"], c["arguments"])
            entry = {
                "step": step,
                "name": c["name"],
                "arguments": c["arguments"],
                "args": out.args,
                "valid": out.valid,
                "result": out.content,
            }
            trace.append(entry)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out.content})
            if emit:
                await emit({"type": "tool", "model_id": mc.id, "rep": rep, **entry})
    else:
        hit_max_steps = True
        ep.finish_reason = "max_steps"

    ep.tool_calls = trace
    ep.close((time.perf_counter() - t0) * 1000)
    if ep.status == "ok":
        check = score(task, trace, ep.content, hit_max_steps)
        ep.check_score, ep.check_detail = check.score, check.items
    return ep


def _absorb(ep: Result, r: Result, offset_ms: float) -> None:
    """Fold one turn's measurements into the episode."""
    if ep.ttft_ms is None and r.ttft_ms is not None:
        ep.ttft_ms = offset_ms + r.ttft_ms
    if ep.ttfc_ms is None and r.ttfc_ms is not None:
        ep.ttfc_ms = offset_ms + r.ttfc_ms
    ep.http_status, ep.request_id = r.http_status, r.request_id or ep.request_id
    ep.stream_complete = ep.stream_complete and r.stream_complete
    ep.prompt_tokens = _add(ep.prompt_tokens, r.prompt_tokens)
    ep.completion_tokens = _add(ep.completion_tokens, r.completion_tokens)
    ep.reasoning_tokens = _add(ep.reasoning_tokens, r.reasoning_tokens)
    ep.tokens_estimated = ep.tokens_estimated or r.tokens_estimated
    ep.chunks += r.chunks
    if r.reasoning:
        ep.reasoning = f"{ep.reasoning}\n\n{r.reasoning}" if ep.reasoning else r.reasoning
