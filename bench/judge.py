"""LLM-as-a-judge, pure part: build the grading prompt, parse the verdict. No I/O."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

RUBRIC = """You are grading an AI assistant's answer to a user request.

Score the answer from 1 to 5:
5 = correct, complete, clear, does exactly what was asked
4 = correct with minor gaps or noise
3 = partially correct or incomplete
2 = mostly wrong, off-topic or ignores the request
1 = wrong, empty or harmful

Judge only what is written. Do not reward length: a short precise answer beats a long padded one.
If grading criteria or an expected answer are given, they take precedence over your own judgement.

Reply with JSON only, no prose, no code fences:
{"score": <1-5>, "reason": "<one sentence>"}"""


@dataclass
class Verdict:
    score: int | None
    reason: str


def build_messages(prompt: str, system: str | None, answer: str, criteria: str | None) -> list[dict]:
    """The judge sees the request, the optional criteria and the answer. Never the model's reasoning."""
    parts = []
    if system:
        parts.append(f"<system_prompt>\n{system}\n</system_prompt>")
    parts.append(f"<request>\n{prompt}\n</request>")
    if criteria:
        parts.append(f"<criteria>\n{criteria}\n</criteria>")
    parts.append(f"<answer>\n{answer}\n</answer>")
    return [{"role": "system", "content": RUBRIC}, {"role": "user", "content": "\n\n".join(parts)}]


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def parse_verdict(text: str | None) -> Verdict:
    """Lenient: strips code fences, takes the first {...} block. Never raises."""
    raw = _FENCE.sub("", text or "")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return Verdict(None, f"unparseable: {raw.strip()[:80]!r}")
    try:
        obj = json.loads(raw[start : end + 1])
        score = int(obj["score"])
    except (ValueError, TypeError, KeyError):
        return Verdict(None, f"unparseable: {raw.strip()[:80]!r}")
    if not 1 <= score <= 5:
        return Verdict(None, f"score out of range: {score}")
    return Verdict(score, str(obj.get("reason") or "").strip())
