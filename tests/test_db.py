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
