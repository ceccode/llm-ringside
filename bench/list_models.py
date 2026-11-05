"""List the model ids a provider exposes: uv run python -m bench.list_models openai"""

import sys

import httpx

from .config import load


def main():
    providers, _ = load()
    names = sys.argv[1:] or list(providers)
    for name in names:
        p = providers[name]
        print(f"== {name} ({p.base_url})")
        if not p.api_key:
            print(f"   missing {p.api_key_env}")
            continue
        try:
            resp = httpx.get(f"{p.base_url}/models", headers={"Authorization": f"Bearer {p.api_key}"}, timeout=30)
            resp.raise_for_status()
            for m in sorted(resp.json().get("data", []), key=lambda m: m.get("id", "")):
                print(f"   {m.get('id')}")
        except Exception as e:  # noqa: BLE001
            print(f"   error: {e}")


if __name__ == "__main__":
    main()
