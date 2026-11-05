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
from .runner import run_model


class RunRequest(BaseModel):
    prompt: str
    system: str | None = None
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


def create_app(
    models: dict[str, ModelConfig],
    db_path: Path | str = db.DB_PATH,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    app = FastAPI(title="ai-benchmark")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/models")
    def list_models():
        return [m.public() for m in models.values()]

    @app.post("/api/run")
    async def run(req: RunRequest):
        unknown = [m for m in req.model_ids if m not in models]
        if unknown:
            raise HTTPException(400, f"unknown model ids: {unknown}")

        con = db.connect(db_path)
        run_id = db.create_run(con, req.prompt, req.system, req.model_dump(exclude={"prompt", "system"}))
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def one(model_id: str, rep: int):
            result = await run_model(
                models[model_id],
                req.messages(),
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                timeout_s=req.timeout_s,
                rep=rep,
                emit=queue.put,
                client=http_client,
            )
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

    @app.post("/api/results/{result_id}/rating")
    def rate(result_id: int, body: RatingRequest):
        with db.connect(db_path) as con:
            db.set_rating(con, result_id, body.rating)
        return {"ok": True}

    return app


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


app = create_app(load()[1])
