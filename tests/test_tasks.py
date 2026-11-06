"""Task loading, tool mocks and scoring: all pure, driven with hand-made traces."""

import json

import pytest

from bench.tasks import Expect, Task, Tool, load_task, load_tasks, mock_call, openai_tools, score

WEATHER = Tool(
    "get_weather",
    "Current weather.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    [{"when": {"city": "Rome"}, "return": {"temp_c": 21}}, {"return": {"error": "unknown city"}}],
)


def task(**expect) -> Task:
    return Task("t", "prompt", [WEATHER], Expect(**expect), max_steps=4)


def call(name="get_weather", valid=True, **args):
    return {"name": name, "args": args, "valid": valid}


def test_shipped_tasks_load():
    tasks = load_tasks()
    assert {"order-status", "weather-compare", "no-tool-needed", "meeting-retry"} <= set(tasks)
    t = tasks["order-status"]
    assert t.expect.ordered and [c["name"] for c in t.expect.calls] == ["find_customer", "get_orders"]
    assert t.public()["tools"] == ["find_customer", "get_orders", "get_invoice"]


def test_load_rejects_missing_fields(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("id: x\nprompt: hi\n")
    with pytest.raises(ValueError, match="bad.yaml: missing 'tools'"):
        load_task(p)


def test_load_rejects_duplicate_ids(tmp_path):
    body = "id: same\nprompt: p\ntools: []\nexpect: {}\n"
    (tmp_path / "a.yaml").write_text(body)
    (tmp_path / "b.yaml").write_text(body)
    with pytest.raises(ValueError, match="duplicate"):
        load_tasks(tmp_path)


def test_openai_tools_hides_mocks():
    [t] = openai_tools(task())
    assert t == {
        "type": "function",
        "function": {"name": "get_weather", "description": "Current weather.", "parameters": WEATHER.parameters},
    }


def test_mock_call_matches_then_falls_back():
    assert json.loads(mock_call(task(), "get_weather", '{"city": "rome "}').content) == {"temp_c": 21}
    out = mock_call(task(), "get_weather", '{"city": "Paris"}')
    assert out.valid and json.loads(out.content) == {"error": "unknown city"}


def test_mock_call_reports_malformed_calls():
    cases = {
        ("nope", "{}"): "unknown tool 'nope'",
        ("get_weather", "{city"): "not valid JSON",
        ("get_weather", "[1]"): "must be a JSON object",
        ("get_weather", ""): "missing required argument(s): city",
    }
    for (name, args), msg in cases.items():
        out = mock_call(task(), name, args)
        assert not out.valid and msg in out.content


def test_score_unordered_with_answer_check():
    t = task(
        calls=[{"name": "get_weather", "args": {"city": "Oslo"}}, {"name": "get_weather", "args": {"city": "Rome"}}],
        final_contains=["warmer", ["69.8", "70"]],
    )
    c = score(t, [call(city="Rome"), call(city="Oslo")], "Rome is warmer at 70°F", hit_max_steps=False)
    assert c.score == 1.0
    assert [i["label"] for i in c.items] == [
        "calls get_weather(city=Oslo)",
        "calls get_weather(city=Rome)",
        "all calls well formed",
        "answer mentions warmer",
        "answer mentions 69.8 or 70",
        "finished within 4 steps",
    ]


def test_score_ordered_requires_order():
    t = task(
        ordered=True,
        calls=[{"name": "get_weather", "args": {"city": "Rome"}}, {"name": "get_weather", "args": {"city": "Oslo"}}],
    )
    c = score(t, [call(city="Oslo"), call(city="Rome")], "", hit_max_steps=False)
    assert [i["ok"] for i in c.items] == [True, False, True, True]


def test_score_restraint_and_malformed_and_max_steps():
    t = task(calls=[], allow_extra=False, final_contains=["51"])
    assert score(t, [], "51", hit_max_steps=False).score == 1.0
    c = score(t, [call(city="Rome"), call("nope", valid=False)], "", hit_max_steps=True)
    assert {i["label"]: i["ok"] for i in c.items} == {
        "1 unexpected call(s)": False,
        "1 malformed call(s)": False,
        "answer mentions 51": False,
        "finished within 4 steps": False,
    }
    assert c.score == 0.0
