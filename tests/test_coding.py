"""Coding tasks: code extraction, grading, and the subprocess test run (real processes, no network)."""

import asyncio

import pytest

from bench.coding import CodingTask, extract_code, grade, load, run_tests
from bench.tasks import load_tasks

TESTS = """
from solution import add

def test_small():
    assert add(1, 2) == 3

def test_negative():
    assert add(-1, -1) == -2

def helper():
    pass
"""


def mk(tests=TESTS, timeout_s=10.0) -> CodingTask:
    return load({"id": "add", "prompt": "p", "tests": tests, "timeout_s": timeout_s}, "add.yaml")


def run(task, code):
    return asyncio.run(run_tests(task, code))


def test_load_finds_test_functions_and_rejects_none():
    assert mk().test_names == ["test_small", "test_negative"]
    with pytest.raises(ValueError, match="no test_"):
        load({"id": "x", "prompt": "p", "tests": "x = 1"}, "x.yaml")
    with pytest.raises(ValueError, match="missing 'tests'"):
        load({"id": "x", "prompt": "p"}, "x.yaml")


def test_extract_code_prefers_the_longest_python_block():
    answer = (
        "Here:\n```python\ndef add(a, b):\n    return a + b\n```\nUsage:\n```py\nadd(1, 2)\n```\n```bash\nrm -rf x\n```"
    )
    assert extract_code(answer) == "def add(a, b):\n    return a + b\n"
    assert extract_code("```\nx = 1\n```") == "x = 1\n"
    assert extract_code("  def f(): pass  ") == "def f(): pass"


def test_grade_counts_every_hidden_test_whatever_went_wrong():
    t = mk()
    assert grade(t, {"test_small": None, "test_negative": "AssertionError"}).score == 0.5
    for check in (
        grade(t, {"import_error": "SyntaxError: bad"}),
        grade(t, None, timed_out=True),
        grade(t, None, stderr="Segmentation fault"),
    ):
        assert check.score == 0 and len(check.items) == 2
    assert "timed out after 10 s" in grade(t, None, timed_out=True).items[0]["label"]
    assert grade(t, {"test_small": None}).items[1] == {"ok": False, "label": "test_negative: not run"}
    assert grade(t, None, skipped="no code in the answer").items[0]["label"] == "test_small: no code in the answer"


def test_run_passes_and_fails_per_test_with_readable_reasons():
    assert run(mk(), "def add(a, b):\n    return a + b\n").score == 1.0
    check = run(mk(), "def add(a, b):\n    print('noise')\n    return abs(a) + abs(b)\n")
    assert [i["ok"] for i in check.items] == [True, False]
    assert check.items[1]["label"] == "test_negative: AssertionError: assert add(-1, -1) == -2"


def test_run_reports_a_broken_solution_on_every_test():
    check = run(mk(), "def add(a, b) return a + b")
    assert check.score == 0 and all("did not import: SyntaxError" in i["label"] for i in check.items)


def test_run_kills_a_hanging_solution():
    check = run(mk(timeout_s=0.5), "while True:\n    pass\n")
    assert check.score == 0 and "timed out" in check.items[0]["label"]


def test_model_code_does_not_see_api_keys(monkeypatch):
    monkeypatch.setenv("XFERENCE_API_KEY", "secret")
    tests = "import os\n\ndef test_no_keys():\n    assert not [k for k in os.environ if k.endswith('_API_KEY')]\n"
    assert run(mk(tests=tests), "").score == 1.0


def test_raises_helper_is_available_to_hidden_tests():
    tests = "from checks import raises\nfrom solution import f\n\ndef test_raises():\n    with raises(ValueError):\n        f()\n"
    assert run(mk(tests=tests), "def f():\n    raise ValueError\n").score == 1.0
    assert "expected ValueError" in run(mk(tests=tests), "def f():\n    pass\n").items[0]["label"]


SHIPPED = sorted(t.id for t in load_tasks().values() if t.kind == "coding")


def test_coding_tasks_ship():
    assert len(SHIPPED) >= 5


@pytest.mark.parametrize("task_id", SHIPPED)
def test_shipped_reference_solution_passes_every_hidden_test(task_id):
    t = load_tasks()[task_id]
    assert t.reference, f"{task_id} has no reference solution"
    check = run(t, t.reference)
    assert check.score == 1.0, [i["label"] for i in check.items if not i["ok"]]
