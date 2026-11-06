from bench.judge import RUBRIC, build_messages, parse_verdict


def test_parse_plain_json():
    v = parse_verdict('{"score": 4, "reason": "Clear and correct."}')
    assert (v.score, v.reason) == (4, "Clear and correct.")


def test_parse_fenced_and_prosed_json():
    assert parse_verdict('```json\n{"score": 5, "reason": "ok"}\n```').score == 5
    assert parse_verdict('Sure! Here is my verdict: {"score": 2, "reason": "wrong"} Hope it helps.').score == 2


def test_parse_string_score_is_accepted():
    assert parse_verdict('{"score": "3", "reason": ""}').score == 3


def test_parse_out_of_range():
    v = parse_verdict('{"score": 7, "reason": "x"}')
    assert v.score is None and "out of range" in v.reason


def test_parse_garbage_and_empty():
    for text in ["I refuse.", "", None, '{"reason": "no score"}', "{not json}"]:
        v = parse_verdict(text)
        assert v.score is None and v.reason.startswith("unparseable")


def test_build_messages_shape():
    msgs = build_messages("What is 2+2?", None, "4", None)
    assert msgs[0] == {"role": "system", "content": RUBRIC}
    user = msgs[1]["content"]
    assert "<request>\nWhat is 2+2?\n</request>" in user
    assert "<answer>\n4\n</answer>" in user
    assert "<criteria>" not in user and "<system_prompt>" not in user


def test_build_messages_with_system_and_criteria():
    user = build_messages("q", "be terse", "a", "must say 4")[1]["content"]
    assert "<system_prompt>\nbe terse\n</system_prompt>" in user
    assert "<criteria>\nmust say 4\n</criteria>" in user
    assert user.index("<criteria>") < user.index("<answer>")


def test_rubric_discourages_length():
    assert "Do not reward length" in RUBRIC
