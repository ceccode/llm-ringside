"""Load models.yaml + .env. Nothing clever."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class Provider:
    name: str
    base_url: str
    api_key_env: str

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


@dataclass
class ModelConfig:
    id: str
    provider: Provider
    model: str
    extra: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "id": self.id,
            "provider": self.provider.name,
            "model": self.model,
            "has_key": self.provider.api_key is not None,
        }


def load(path: Path | None = None) -> tuple[dict[str, Provider], dict[str, ModelConfig]]:
    """Default: models.yaml in the repo root, overridable with BENCH_MODELS_YAML."""
    path = path or Path(os.environ.get("BENCH_MODELS_YAML", ROOT / "models.yaml"))
    raw = yaml.safe_load(path.read_text())
    providers = {
        name: Provider(name=name, base_url=p["base_url"].rstrip("/"), api_key_env=p["api_key_env"])
        for name, p in raw["providers"].items()
    }
    models: dict[str, ModelConfig] = {}
    for m in raw["models"]:
        models[m["id"]] = ModelConfig(
            id=m["id"], provider=providers[m["provider"]], model=m["model"], extra=m.get("extra") or {}
        )
    return providers, models
