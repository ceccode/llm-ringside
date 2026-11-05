# ai-benchmark

One prompt, several LLM APIs, answers streamed side by side with the numbers that matter:
time to first token, total time, tokens per second, token usage, and what went wrong.

Built to compare [Xference](https://www.xference.ai) with Anthropic, OpenAI and OpenCode Go,
but it works with any endpoint that speaks the OpenAI `chat/completions` format.
No framework, no build step: a FastAPI server, a SQLite file and one HTML page.

## Quick start

```bash
cp .env.example .env            # add the API keys you have
uv sync
uv run uvicorn bench.app:app --port 8000
```

Open <http://localhost:8000>, type a prompt, tick the models, press Run (or ⌘↩).
Models whose key is missing are shown greyed out. Every run is saved and can be reopened
from the **history** tab, where each answer can also be rated 1–5.

## Adding a model

Everything lives in [`models.yaml`](models.yaml). A provider is a base URL and the name of
the environment variable holding its key; a model is a provider, a model name and optional
`extra` fields merged verbatim into the request body:

```yaml
providers:
  xference:
    base_url: https://api.xference.ai/v1
    api_key_env: XFERENCE_API_KEY

models:
  - id: xf/qwen3.6-35b-a3b:think
    provider: xference
    model: Qwen3.6-35B-A3B
    extra: { chat_template_kwargs: { enable_thinking: true } }
```

To see which model names a provider actually serves:

```bash
uv run python -m bench.list_models xference
```

Anthropic is reached through its OpenAI-compatible endpoint, so one adapter covers all providers.

## What is measured

| Column | Meaning |
|---|---|
| **TTFT** | time to the first streamed token, reasoning or content |
| **total** | wall-clock time for the whole request |
| **tok/s** | completion tokens ÷ (total − TTFT). A `~` means the provider sent no `usage`, so streamed chunks were counted instead |
| **out tokens** | `usage.completion_tokens`, with reasoning tokens in parentheses when reported |
| **pill** | `stop`, `length`, `tool_calls`, or `error` (hover for the reason) |
| **footer** | HTTP status, prompt tokens, `no [DONE]` when a stream closed without the sentinel, provider request id |

Errors are first-class results: HTTP refusals, error events inside the stream, timeouts and
streams that end without `[DONE]` are all recorded and shown. **repeat** sends the same prompt
N times per model to see variance.

## How it works

```
bench/
  config.py   models.yaml + .env  →  Provider / ModelConfig
  stream.py   SSE payloads in, Result out. Pure: no I/O, no clock
  runner.py   httpx streaming; timestamps each payload and feeds stream.Result
  db.py       SQLite, two tables: runs and results
  app.py      FastAPI: /api/run fans a prompt out to N models and relays SSE to the browser
static/index.html   the whole UI, vanilla JS
```

The design rule is simple: everything that decides *what we report* sits in `stream.py` and takes
plain values, while `runner.py` and `app.py` only move bytes. That is what keeps the tests fast and
the code small.

## Development

```bash
uv sync --group dev
uv run pytest          # ~20 tests, < 1 s, no network, no keys
uv run ruff check .
```

Tests talk to a fake OpenAI-compatible provider served in-process through httpx's ASGI transport
(see [`tests/conftest.py`](tests/conftest.py)). The model name selects a scenario, so adding a
provider quirk means adding a scenario there and a test next to it.

`BENCH_MODELS_YAML=/path/to/other.yaml` points the server at a different model list.

## Not in scope (yet)

- Cost per run: usage is recorded, prices are not. Easy to add if you need it.
- Batch suites and LLM-as-judge scoring: today the quality signal is your own 1–5 rating.

## License

MIT, see [LICENSE](LICENSE).
