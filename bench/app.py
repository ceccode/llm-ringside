"""HTTP layer. `create_app` takes its collaborators so tests can hand in fakes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import db
from .config import ROOT, ModelConfig, load
from .episode import run_episode
from .judge import Verdict, build_messages, parse_verdict
from .runner import run_model
from .tasks import Task, load_tasks


class RunRequest(BaseModel):
    prompt: str = ""
    task_id: str | None = None  # an agentic task from tasks/; its prompt and system replace these
    system: str | None = None
    criteria: str | None = None  # grading criteria / expected answer, read by the judge only
    model_ids: list[str] = Field(min_length=1)
    max_tokens: int = 1024
    temperature: float | None = None
    repeat: int = Field(default=1, ge=1, le=10)
    timeout_s: float = 180

    def messages(self) -> list[dict]:
        system = [{"role": "system", "content": self.system}] if self.system else []
        return system + [{"role": "user", "content": self.prompt}]


class RatingRequest(BaseModel):
    rating: int | None = Field(default=None, ge=1, le=5)


JUDGE_MAX_TOKENS = 2048  # room for models that think silently first (Claude 5.5 does, over the OpenAI endpoint)


class JudgeRequest(BaseModel):
    judge_model_id: str


def create_app(
    models: dict[str, ModelConfig],
    db_path: Path | str = db.DB_PATH,
    http_client: httpx.AsyncClient | None = None,
    tasks: dict[str, Task] | None = None,
) -> FastAPI:
    app = FastAPI(title="ai-benchmark")
    tasks = tasks or {}

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/models")
    def list_models():
        return [m.public() for m in models.values()]

    @app.get("/api/tasks")
    def list_tasks():
        return [t.public() for t in tasks.values()]

    @app.post("/api/run")
    async def run(req: RunRequest):
        unknown = [m for m in req.model_ids if m not in models]
        if unknown:
            raise HTTPException(400, f"unknown model ids: {unknown}")
        task = None
        if req.task_id:
            task = tasks.get(req.task_id)
            if not task:
                raise HTTPException(400, f"unknown task id: {req.task_id}")
            req.prompt, req.system = task.prompt, task.system
        elif not req.prompt.strip():
            raise HTTPException(400, "prompt is empty")

        con = db.connect(db_path)
        params = req.model_dump(exclude={"prompt", "system", "criteria", "task_id"})
        run_id = db.create_run(con, req.prompt, req.system, params, criteria=req.criteria, task_id=req.task_id)
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def one(model_id: str, rep: int):
            common = {
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "timeout_s": req.timeout_s,
                "rep": rep,
                "emit": queue.put,
                "client": http_client,
            }
            if task:
                result = await run_episode(models[model_id], task, **common)
            else:
                result = await run_model(models[model_id], req.messages(), **common)
            d = result.to_dict()
            d["id"] = db.save_result(con, run_id, d)
            await queue.put({"type": "result", "model_id": model_id, "rep": rep, "result": d})

        async def fan_out():
            jobs = [one(m, rep) for rep in range(req.repeat) for m in req.model_ids]
            await asyncio.gather(*jobs, return_exceptions=True)
            await queue.put(None)

        async def events():
            task = asyncio.create_task(fan_out())
            yield _sse({"type": "run", "run_id": run_id})
            try:
                while (ev := await queue.get()) is not None:
                    yield _sse(ev)
                yield _sse({"type": "done", "run_id": run_id})
            finally:
                task.cancel()
                con.close()

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    @app.get("/api/runs")
    def runs():
        with db.connect(db_path) as con:
            return db.list_runs(con)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: int):
        with db.connect(db_path) as con:
            r = db.get_run(con, run_id)
        if not r:
            raise HTTPException(404)
        return r

    @app.delete("/api/runs/{run_id}")
    def run_delete(run_id: int):
        with db.connect(db_path) as con:
            db.delete_run(con, run_id)
        return {"ok": True}

    @app.post("/api/runs/{run_id}/judge")
    async def judge(run_id: int, body: JudgeRequest):
        """Grade every successful answer of a run with one judge model. Re-judging overwrites."""
        judge_mc = models.get(body.judge_model_id)
        if not judge_mc:
            raise HTTPException(400, f"unknown judge model id: {body.judge_model_id}")
        if not judge_mc.provider.api_key:
            raise HTTPException(400, f"judge model has no key: {body.judge_model_id}")
        with db.connect(db_path) as con:
            run = db.get_run(con, run_id)
        if not run:
            raise HTTPException(404)

        async def grade(x: dict) -> dict:
            msgs = build_messages(run["prompt"], run["system"], x["content"], run["criteria"])
            r = await run_model(judge_mc, msgs, max_tokens=JUDGE_MAX_TOKENS, client=http_client)
            v = parse_verdict(r.content) if r.status == "ok" else Verdict(None, r.error or "judge failed")
            if v.score is None and r.finish_reason == "length":
                v = Verdict(None, f"judge hit its {JUDGE_MAX_TOKENS}-token limit before finishing its verdict")
            return {"id": x["id"], "judge_score": v.score, "judge_reason": v.reason}

        gradable = [x for x in run["results"] if x["status"] == "ok" and x["content"]]
        verdicts = await asyncio.gather(*(grade(x) for x in gradable))
        with db.connect(db_path) as con:
            for v in verdicts:
                db.set_verdict(con, v["id"], v["judge_score"], v["judge_reason"], judge_mc.id)
        return {"judge_model": judge_mc.id, "results": verdicts}

    @app.post("/api/results/{result_id}/rating")
    def rate(result_id: int, body: RatingRequest):
        with db.connect(db_path) as con:
            db.set_rating(con, result_id, body.rating)
        return {"ok": True}

    return app


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


app = create_app(load()[1], tasks=load_tasks())
