"""Discover and cache Ollama model statistics via the Ollama API."""

import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from ollama_codeeval.config import OLLAMA_HOST, OUTPUT_BASE

DEFAULT_CACHE = OUTPUT_BASE / "model_stats.json"


def api_get(path: str) -> dict:
    with urlopen(f"{OLLAMA_HOST}{path}") as r:  # nosec
        return json.loads(r.read())


def api_post(path: str, body: dict) -> dict:
    req = Request(
        f"{OLLAMA_HOST}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as r:  # nosec B310
        return json.loads(r.read())


def list_models() -> list[dict]:
    return api_get("/api/tags")["models"]


def show_model(name: str) -> dict:
    return api_post("/api/show", {"name": name})


def discover(cache_path: Path = DEFAULT_CACHE) -> dict:
    """Query Ollama for all models and their details, write to cache."""
    models = list_models()
    stats = {}
    for m in models:
        name = m["name"]
        print(f"  {name}...", end=" ", flush=True)
        details = m.get("details", {})
        try:
            info = show_model(name)
        except URLError as e:
            print(f"error: {e}")
            continue

        model_info = info.get("model_info", {})
        stats[name] = {
            "digest": m["digest"],
            "size": m["size"],
            "modified_at": m["modified_at"],
            "family": details.get("family"),
            "families": details.get("families"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "format": details.get("format"),
            "context_length": model_info.get(
                f"{details.get('family', '')}.context_length"
            ),
            "embedding_length": model_info.get(
                f"{details.get('family', '')}.embedding_length"
            ),
            "parameter_count": model_info.get("general.parameter_count"),
            "capabilities": info.get("capabilities"),
        }
        print("ok")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {len(stats)} models to {cache_path}")
    return stats


def refresh_missing(cache_path: Path = DEFAULT_CACHE) -> dict:
    """Add stats for any Ollama models not already in the cache.

    Loads the existing cache (if any), queries Ollama for the current model
    list, fetches details only for models absent from the cache, then saves.
    Returns the updated stats dict.
    """
    stats: dict = {}
    if cache_path.exists():
        stats = json.loads(cache_path.read_text())

    try:
        models = list_models()
    except URLError as e:
        print(f"Cannot connect to Ollama: {e}", file=sys.stderr)
        return stats

    added = 0
    for m in models:
        name = m["name"]
        if name in stats:
            continue
        print(f"  {name} (new)...", end=" ", flush=True)
        details = m.get("details", {})
        try:
            info = show_model(name)
        except URLError as e:
            print(f"error: {e}")
            continue
        model_info = info.get("model_info", {})
        stats[name] = {
            "digest": m["digest"],
            "size": m["size"],
            "modified_at": m["modified_at"],
            "family": details.get("family"),
            "families": details.get("families"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "format": details.get("format"),
            "context_length": model_info.get(
                f"{details.get('family', '')}.context_length"
            ),
            "embedding_length": model_info.get(
                f"{details.get('family', '')}.embedding_length"
            ),
            "parameter_count": model_info.get("general.parameter_count"),
            "capabilities": info.get("capabilities"),
        }
        print("ok")
        added += 1

    if added:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(stats, indent=2))
        print(f"Added {added} model(s) to {cache_path}")
    else:
        print("model_stats.json up to date")

    return stats


def load_cache(cache_path: Path = DEFAULT_CACHE) -> dict:
    """Load cached model stats from JSON file."""
    return json.loads(cache_path.read_text())


if __name__ == "__main__":
    try:
        discover()
    except URLError:
        print("Error: cannot connect to Ollama at", OLLAMA_HOST, file=sys.stderr)
        sys.exit(1)
