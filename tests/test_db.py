from bench import db
from bench.stream import Result


def test_result_fields_cover_every_db_column():
    assert set(db.RESULT_COLS) <= set(Result("m").to_dict())


def test_round_trip_rating_and_delete(tmp_path):
    path = tmp_path / "t.db"
    with db.connect(path) as con:
        run_id = db.create_run(con, "prompt", None, {"repeat": 1})
        r = Result("fake/ok", content="hi", http_status=200)
        r.close(100)
        rid = db.save_result(con, run_id, r.to_dict())
        db.set_rating(con, rid, 4)

    with db.connect(path) as con:
        runs = db.list_runs(con)
        assert len(runs) == 1 and runs[0]["n"] == 1 and runs[0]["errors"] == 1  # no [DONE] -> error
        run = db.get_run(con, run_id)
        assert run["prompt"] == "prompt" and run["params"] == {"repeat": 1}
        assert run["results"][0]["rating"] == 4 and run["results"][0]["content"] == "hi"
        assert db.get_run(con, 999) is None
        db.delete_run(con, run_id)
        assert db.list_runs(con) == []
        assert con.execute("SELECT COUNT(*) FROM results").fetchone()[0] == 0  # cascade


def test_criteria_and_verdict_round_trip(tmp_path):
    with db.connect(tmp_path / "t.db") as con:
        run_id = db.create_run(con, "p", None, {}, criteria="must say 4")
        rid = db.save_result(con, run_id, Result("fake/ok").to_dict())
        db.set_verdict(con, rid, 4, "fine", "fake/judge")
        run = db.get_run(con, run_id)
        assert run["criteria"] == "must say 4"
        x = run["results"][0]
        assert (x["judge_score"], x["judge_reason"], x["judge_model"]) == (4, "fine", "fake/judge")


def test_connect_migrates_a_pre_judge_db(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = db.SCHEMA
    for cols in db.ADDED_COLS.values():
        for name in cols:
            old = "\n".join(line for line in old.splitlines() if name not in line)
    old = old.replace("rating INTEGER,", "rating INTEGER")
    with sqlite3.connect(path) as raw:
        raw.executescript(old)
        raw.execute("INSERT INTO runs (created_at, prompt, params) VALUES ('t', 'p', '{}')")
    with db.connect(path) as con:
        cols = {r["name"] for r in con.execute("PRAGMA table_info(results)")}
        assert {"judge_score", "judge_reason", "judge_model"} <= cols
        assert db.get_run(con, 1)["criteria"] is None
