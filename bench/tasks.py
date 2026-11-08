"""Agentic tasks, pure part: load the YAML, mock the tools, score an episode. No network, no clock.

A task is a prompt, a set of tools with canned responses, and expectations a machine can check.
See tasks/*.yaml and the README for the format.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import yaml

from .config import ROOT

TASKS_DIR = ROOT / "tasks"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    responses: list[dict] = field(default_factory=list)  # [{when: {...}, return: ...}], first match wins


@dataclass
class Expect:
    calls: list[dict] = field(default_factory=list)  # [{name, args: {...}}], args is a subset match
    ordered: bool = False
    allow_extra: bool = True
    never: list[dict] = field(default_factory=list)  # [{name, args?}], any attempt fails, even a malformed one
    final_contains: list[str | list[str]] = field(default_factory=list)  # a list entry = any of these


@dataclass
class Task:
    id: str
    prompt: str
    tools: list[Tool]
    expect: Expect
    system: str | None = None
    max_steps: int = 6
    kind: ClassVar[str] = "agentic"

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "tools": [t.name for t in self.tools],
            "max_steps": self.max_steps,
        }


@dataclass
class ToolOutcome:
    valid: bool  # the call was well formed: known tool, JSON object, required args present
    args: dict | None
    content: str  # what goes back to the model in the `tool` message


@dataclass
class Check:
    score: float  # passed / total
    items: list[dict]  # [{ok: bool, label: str}]


# -- loading ------------------------------------------------------------------


def load_task(path: Path):
    """A Task, or a coding.CodingTask when the file says `kind: coding`."""
    raw = yaml.safe_load(path.read_text()) or {}
    if raw.get("kind") == "coding":
        from . import coding  # coding imports Check from here

        return coding.load(raw, path.name)
    for key in ("id", "prompt", "tools", "expect"):
        if key not in raw:
            raise ValueError(f"{path.name}: missing '{key}'")
    tools = [
        Tool(t["name"], t.get("description", ""), t.get("parameters") or {"type": "object"}, t.get("responses") or [])
        for t in raw["tools"]
    ]
    exp = raw["expect"] or {}
    expect = Expect(
        calls=exp.get("calls") or [],
        ordered=bool(exp.get("ordered", False)),
        allow_extra=bool(exp.get("allow_extra", True)),
        never=exp.get("never") or [],
        final_contains=exp.get("final_contains") or [],
    )
    return Task(raw["id"], raw["prompt"], tools, expect, raw.get("system"), int(raw.get("max_steps", 6)))


def load_tasks(directory: Path | None = None) -> dict:
    """Every tasks/*.yaml by id. Default dir overridable with BENCH_TASKS_DIR."""
    directory = directory or Path(os.environ.get("BENCH_TASKS_DIR", TASKS_DIR))
    tasks: dict[str, Task] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        task = load_task(path)
        if task.id in tasks:
            raise ValueError(f"{path.name}: duplicate task id '{task.id}'")
        tasks[task.id] = task
    return tasks


def openai_tools(task: Task) -> list[dict]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in task.tools
    ]


# -- mocking ------------------------------------------------------------------


_NUMBER = re.compile(r"-?\d+(\.\d+)?")


def _norm(v) -> str:
    """Trimmed, case-insensitive; numbers compare by value, so 103.2, "103.20" and 103.20 match."""
    s = (v if isinstance(v, str) else json.dumps(v, sort_keys=True)).strip().lower()
    return repr(float(s)) if _NUMBER.fullmatch(s) else s


def _matches(subset: dict, args: dict) -> bool:
    return all(k in args and _norm(args[k]) == _norm(v) for k, v in subset.items())


def mock_call(task: Task, name: str, arguments: str) -> ToolOutcome:
    """Answer a tool call from the task's canned responses. Malformed calls get an error the model can read."""
    tool = next((t for t in task.tools if t.name == name), None)
    if tool is None:
        known = ", ".join(t.name for t in task.tools)
        return ToolOutcome(False, None, f"error: unknown tool '{name}'. Available tools: {known}")
    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError:
        return ToolOutcome(False, None, "error: arguments are not valid JSON")
    if not isinstance(args, dict):
        return ToolOutcome(False, None, "error: arguments must be a JSON object")
    missing = [k for k in tool.parameters.get("required", []) if k not in args]
    if missing:
        return ToolOutcome(False, args, f"error: missing required argument(s): {', '.join(missing)}")
    for r in tool.responses:
        if _matches(r.get("when") or {}, args):
            ret = r.get("return")
            return ToolOutcome(True, args, ret if isinstance(ret, str) else json.dumps(ret))
    return ToolOutcome(True, args, json.dumps({"error": "no result"}))


# -- scoring ------------------------------------------------------------------


def _label(call: dict, verb: str = "calls") -> str:
    args = ", ".join(f"{k}={v}" for k, v in (call.get("args") or {}).items())
    return f"{verb} {call['name']}({args})"


def score(task: Task, calls: list[dict], final: str, hit_max_steps: bool) -> Check:
    """`calls` is the episode trace: [{name, args, valid, ...}] in call order."""
    exp = task.expect
    items: list[dict] = []
    valid = [c for c in calls if c.get("valid")]
    used: set[int] = set()
    cursor = 0
    for want in exp.calls:
        hit = None
        for i, c in enumerate(valid):
            if i in used or (exp.ordered and i < cursor):
                continue
            if c["name"] == want["name"] and _matches(want.get("args") or {}, c.get("args") or {}):
                hit = i
                break
        if hit is not None:
            used.add(hit)
            cursor = hit + 1
        items.append({"ok": hit is not None, "label": _label(want)})

    for banned in exp.never:
        tried = any(
            c["name"] == banned["name"] and _matches(banned.get("args") or {}, c.get("args") or {}) for c in calls
        )
        items.append({"ok": not tried, "label": _label(banned, "never calls")})

    if not exp.allow_extra:
        extra = len(valid) - len(used)
        items.append({"ok": extra == 0, "label": "no unexpected calls" if not extra else f"{extra} unexpected call(s)"})
    if calls:
        bad = len(calls) - len(valid)
        items.append({"ok": bad == 0, "label": "all calls well formed" if not bad else f"{bad} malformed call(s)"})

    text = final.lower()
    for want in exp.final_contains:
        alts = want if isinstance(want, list) else [want]
        items.append(
            {"ok": any(str(a).lower() in text for a in alts), "label": f"answer mentions {' or '.join(map(str, alts))}"}
        )

    items.append({"ok": not hit_max_steps, "label": f"finished within {task.max_steps} steps"})
    passed = sum(i["ok"] for i in items)
    return Check(round(passed / len(items), 3), items)
