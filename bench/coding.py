"""Coding tasks: the model writes a Python module, hidden tests grade it in a subprocess.

One turn, no tools. The answer's code block becomes `solution.py` in a throwaway directory, next to
the task's hidden tests and a tiny harness that runs each `test_*` function and reports pass/fail as
JSON. The subprocess gets a timeout, its own process group (killed whole on timeout), a stripped
environment (no API keys), and a cap on file size. It is not a sandbox: model-written code runs on
this machine with your user's permissions. See the README.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import httpx

from .config import ModelConfig
from .runner import Emit, run_model
from .stream import Result
from .tasks import Check

SYSTEM = (
    "You are writing a Python 3.11 module named solution.py. Hidden tests will import from it. "
    "Reply with the complete module in a single ```python code block. Standard library only."
)
MARK = "@@RINGSIDE@@"

# Runs inside the subprocess. Test names come in argv so a broken solution still reports every test.
HARNESS = (
    r'''
import json, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
except Exception:
    pass
out, real = {}, sys.stdout
sys.stdout = open(os.devnull, "w")

def describe(e):
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("test_hidden.py")]
    text = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    if frames and frames[-1].line and isinstance(e, AssertionError) and not str(e):
        text += f": {frames[-1].line}"
    return text[:300]

try:
    import test_hidden
except BaseException as e:
    out = {"import_error": describe(e)}
else:
    for name in json.loads(sys.argv[1]):
        try:
            getattr(test_hidden, name)()
            out[name] = None
        except BaseException as e:
            out[name] = describe(e)
real.write("'''
    + MARK
    + r"""" + json.dumps(out) + "\n")
"""
)

# Importable from the hidden tests: `from checks import raises`.
CHECKS = """
class raises:
    def __init__(self, exc):
        self.exc = exc
    def __enter__(self):
        return self
    def __exit__(self, kind, value, tb):
        if kind is None:
            raise AssertionError(f"expected {self.exc.__name__}")
        return issubclass(kind, self.exc)
"""


@dataclass
class CodingTask:
    id: str
    prompt: str
    tests: str  # source of the hidden test module; never sent to the model
    test_names: list[str]
    system: str = SYSTEM
    timeout_s: float = 10.0
    max_tokens: int = 16384  # floor for the run: thinking models spend 8-30k tokens before answering
    reference: str | None = None  # a known-good solution, used by our own tests only
    kind: ClassVar[str] = "coding"

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "tests": len(self.test_names),
            "timeout_s": self.timeout_s,
            "max_tokens": self.max_tokens,
        }


def load(raw: dict, name: str) -> CodingTask:
    for key in ("id", "prompt", "tests"):
        if key not in raw:
            raise ValueError(f"{name}: missing '{key}'")
    tree = ast.parse(raw["tests"], filename=name)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    if not names:
        raise ValueError(f"{name}: 'tests' defines no test_* functions")
    return CodingTask(
        id=raw["id"],
        prompt=raw["prompt"],
        tests=raw["tests"],
        test_names=names,
        system=raw.get("system") or SYSTEM,
        timeout_s=float(raw.get("timeout_s", 10)),
        max_tokens=int(raw.get("max_tokens", 16384)),
        reference=raw.get("reference"),
    )


_FENCE = re.compile(r"```([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(answer: str) -> str:
    """The longest Python (or unlabelled) fenced block; the whole answer if there is none."""
    blocks = [body for lang, body in _FENCE.findall(answer) if lang.lower() in ("", "python", "py", "python3")]
    return max(blocks, key=len) if blocks else answer.strip()


def grade(
    task: CodingTask, report: dict | None, *, timed_out: bool = False, stderr: str = "", skipped: str | None = None
) -> Check:
    """One check item per hidden test. Pure: `report` is what the harness printed, `skipped` why nothing ran."""
    if skipped:
        why = skipped
    elif timed_out:
        why = f"timed out after {task.timeout_s:g} s"
    elif report is None:
        why = f"no report from the test run{': ' + stderr.strip()[-200:] if stderr.strip() else ''}"
    elif "import_error" in report:
        why = f"solution did not import: {report['import_error']}"
    else:
        why = None
    items = []
    for name in task.test_names:
        err = why or (report.get(name, "not run") if report else "not run")
        items.append({"ok": err is None, "label": name if err is None else f"{name}: {err}"})
    passed = sum(i["ok"] for i in items)
    return Check(round(passed / len(items), 3), items)


def _env(home: str) -> dict:
    """Just enough to start Python. Nothing from .env reaches model-written code."""
    return {"PATH": os.environ.get("PATH", ""), "HOME": home, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}


async def run_tests(task: CodingTask, code: str) -> Check:
    with tempfile.TemporaryDirectory(prefix="ringside-") as d:
        files = {"solution.py": code, "test_hidden.py": task.tests, "checks.py": CHECKS, "harness.py": HARNESS}
        for fname, body in files.items():
            Path(d, fname).write_text(body)
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "harness.py",
            json.dumps(task.test_names),
            cwd=d,
            env=_env(d),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), task.timeout_s)
        except TimeoutError:
            _kill(proc)
            await proc.wait()
            return grade(task, None, timed_out=True)
        finally:
            if proc.returncode is None:
                _kill(proc)
    lines = [ln for ln in out.decode(errors="replace").splitlines() if ln.startswith(MARK)]
    report = json.loads(lines[-1][len(MARK) :]) if lines else None
    return grade(task, report, stderr=err.decode(errors="replace"))


def _kill(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


async def run_coding(
    mc: ModelConfig,
    task: CodingTask,
    *,
    max_tokens: int = 1024,
    temperature: float | None = None,
    timeout_s: float = 180.0,
    rep: int = 0,
    emit: Emit | None = None,
    client: httpx.AsyncClient | None = None,
) -> Result:
    """Speed metrics cover the model's answer only; the test run is not timed into them."""
    messages = [{"role": "system", "content": task.system}, {"role": "user", "content": task.prompt}]
    r = await run_model(
        mc,
        messages,
        max_tokens=max(max_tokens, task.max_tokens),
        temperature=temperature,
        timeout_s=timeout_s,
        rep=rep,
        emit=emit,
        client=client,
    )
    r.steps = 1
    if r.status == "ok":
        code = extract_code(r.content)
        if code:
            check = await run_tests(task, code)
        else:
            cut = " (hit max_tokens before answering)" if r.finish_reason == "length" else ""
            check = grade(task, None, skipped=f"no code in the answer{cut}")
        r.check_score, r.check_detail = check.score, check.items
    return r
