from bench.config import load


def test_load_models_yaml(tmp_path, monkeypatch):
    (tmp_path / "models.yaml").write_text("""
providers:
  p: { base_url: "https://p.example/v1/", api_key_env: P_KEY }
models:
  - { id: p/a, provider: p, model: a }
  - { id: p/b, provider: p, model: b, extra: { x: 1 } }
""")
    monkeypatch.delenv("P_KEY", raising=False)
    providers, models = load(tmp_path / "models.yaml")
    assert providers["p"].base_url == "https://p.example/v1"  # trailing slash stripped
    assert models["p/a"].extra == {} and models["p/b"].extra == {"x": 1}
    assert models["p/a"].public() == {"id": "p/a", "provider": "p", "model": "a", "has_key": False}
    monkeypatch.setenv("P_KEY", "k")
    assert models["p/a"].public()["has_key"] is True


def test_repo_models_yaml_loads():
    _, models = load()
    assert "xf/qwen3.6-35b-a3b" in models
