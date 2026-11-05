"""End to end through FastAPI: SSE fan-out, persistence, rating. Still no network."""

import json

import pytest
from fastapi.testclient import TestClient

from bench.app import create_app


@pytest.fixture
def client(models, http_client, tmp_path):
    return TestClient(create_app(models, db_path=tmp_path / "t.db", http_client=http_client))


def events(resp):
    return [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")]


def test_models_endpoint_reports_key_presence(client):
    ms = {m["id"]: m for m in client.get("/api/models").json()}
    assert ms["fake/ok"]["has_key"] is True
    assert ms["fake/missingkey"]["has_key"] is False


def test_run_fans_out_streams_and_persists(client):
    resp = client.post("/api/run", json={"prompt": "hi", "model_ids": ["fake/ok", "fake/http429"], "repeat": 2})
    assert resp.status_code == 200
    evs = events(resp)
    assert evs[0]["type"] == "run" and evs[-1]["type"] == "done"
    results = [e["result"] for e in evs if e["type"] == "result"]
    assert sorted((r["model_id"], r["rep"]) for r in results) == [
        ("fake/http429", 0),
        ("fake/http429", 1),
        ("fake/ok", 0),
        ("fake/ok", 1),
    ]
    assert any(e["type"] == "delta" and e.get("content") for e in evs)

    run_id = evs[0]["run_id"]
    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == run_id and runs[0]["n"] == 4 and runs[0]["errors"] == 2
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["params"]["repeat"] == 2 and len(detail["results"]) == 4


def test_rating_round_trip_and_delete(client):
    evs = events(client.post("/api/run", json={"prompt": "hi", "model_ids": ["fake/ok"]}))
    rid = next(e["result"]["id"] for e in evs if e["type"] == "result")
    run_id = evs[0]["run_id"]
    assert client.post(f"/api/results/{rid}/rating", json={"rating": 5}).status_code == 200
    assert client.get(f"/api/runs/{run_id}").json()["results"][0]["rating"] == 5
    assert client.post(f"/api/results/{rid}/rating", json={"rating": 9}).status_code == 422
    assert client.delete(f"/api/runs/{run_id}").status_code == 200
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def test_unknown_model_is_rejected(client):
    assert client.post("/api/run", json={"prompt": "hi", "model_ids": ["nope"]}).status_code == 400
