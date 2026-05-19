#!/usr/bin/env python3
"""Benchmark Ollama models: VRAM usage and tokens/s across context sizes."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

DEFAULT_CTX = [2048, 4096, 8192, 16384]
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_OUT = Path("output")
LOAD_TIMEOUT = 120       # seconds to wait for model to appear in /api/ps
BENCH_TIMEOUT = 180     # seconds for the benchmark generate call

BENCH_PROMPT = """\
Complete the following Python function. Output only the function body, no markdown.

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"Check if in given list of numbers, are any two numbers closer to each other
    than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""

SKIP_FAMILIES = {"bert", "nomic", "clip"}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def is_generative(model: dict) -> bool:
    """Return True if model is a generative LLM (not embedding/base)."""
    name: str = model.get("name", "")
    if "embed" in name.lower():
        return False
    tag = name.split(":")[-1] if ":" in name else ""
    if tag == "base":
        return False
    families = model.get("details", {}).get("families") or []
    if any(f.lower() in SKIP_FAMILIES for f in families):
        return False
    return True


def parse_nvidia_smi(raw: str) -> dict[str, int]:
    """Parse nvidia-smi memory.used output into {GPU-N: mb} dict."""
    result = {}
    for i, line in enumerate(raw.strip().splitlines()):
        val = line.strip().replace(" MiB", "").replace("MiB", "")
        if val.isdigit():
            result[f"GPU-{i}"] = int(val)
    return result


def vram_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Compute per-GPU VRAM delta (clamp to 0)."""
    return {k: max(0, after.get(k, 0) - before.get(k, 0)) for k in after}


def calc_toks_per_sec(count: int, duration_ns: int) -> float | None:
    """Convert Ollama token count + nanosecond duration to tokens/s."""
    if not count or not duration_ns:
        return None
    return round(count / (duration_ns / 1e9), 2)


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def api_get(host: str, path: str, timeout: int = 10) -> dict:
    with urlopen(f"{host}{path}", timeout=timeout) as r:  # nosec
        return json.loads(r.read())


def api_post(host: str, path: str, body: dict, timeout: int = 10) -> dict:
    req = Request(
        f"{host}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as r:  # nosec
        return json.loads(r.read())


def list_models(host: str) -> list[dict]:
    return api_get(host, "/api/tags")["models"]


def loaded_models(host: str) -> list[str]:
    """Return names of currently loaded models from /api/ps."""
    try:
        return [m["name"] for m in api_get(host, "/api/ps").get("models", [])]
    except Exception:
        return []


def ps_size_vram(host: str, model: str) -> int | None:
    """Return size_vram in MB for a loaded model, or None."""
    try:
        models = api_get(host, "/api/ps").get("models", [])
        for m in models:
            if m["name"] == model:
                return m.get("size_vram", 0) // (1024 * 1024)
    except Exception:
        pass
    return None


def unload_model(host: str, model: str) -> None:
    try:
        api_post(host, "/api/generate", {"model": model, "keep_alive": 0}, timeout=15)
    except Exception:
        pass


def load_model(host: str, model: str, ctx: int) -> None:
    """Send a no-op generate to load the model with num_ctx."""
    api_post(host, "/api/generate", {
        "model": model,
        "prompt": "",
        "num_predict": 0,
        "keep_alive": "5m",
        "options": {"num_ctx": ctx},
        "stream": False,
    }, timeout=LOAD_TIMEOUT)



def run_benchmark(host: str, model: str, ctx: int) -> dict:
    """Run a generate and return timing stats."""
    resp = api_post(host, "/api/generate", {
        "model": model,
        "prompt": BENCH_PROMPT,
        "keep_alive": "5m",
        "options": {"num_ctx": ctx},
        "stream": False,
    }, timeout=BENCH_TIMEOUT)
    return {
        "prefill_tokens_per_sec": calc_toks_per_sec(
            resp.get("prompt_eval_count", 0),
            resp.get("prompt_eval_duration", 0),
        ),
        "decode_tokens_per_sec": calc_toks_per_sec(
            resp.get("eval_count", 0),
            resp.get("eval_duration", 0),
        ),
        "prompt_tokens": resp.get("prompt_eval_count"),
        "generated_tokens": resp.get("eval_count"),
    }


# ---------------------------------------------------------------------------
# VRAM via nvidia-smi
# ---------------------------------------------------------------------------

def query_vram() -> dict[str, int] | None:
    """Query per-GPU used VRAM in MB. Returns None if nvidia-smi unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        parsed = parse_nvidia_smi(result.stdout)
        if not parsed and result.stdout.strip():
            print(f"  [warn] nvidia-smi returned unexpected output: {result.stdout.strip()!r}", file=sys.stderr)
        return parsed or None
    except FileNotFoundError:
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  [warn] nvidia-smi failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def bench_model_ctx(host: str, model: str, ctx: int) -> dict:
    """Run one (model, ctx) benchmark. Returns a result record."""
    record: dict = {
        "model": model,
        "ctx": ctx,
        "vram_delta_mb": None,
        "vram_total_mb": None,
        "ollama_size_vram_mb": None,
        "prefill_tokens_per_sec": None,
        "decode_tokens_per_sec": None,
        "prompt_tokens": None,
        "generated_tokens": None,
        "error": None,
    }

    # Unload all models from memory before measuring baseline VRAM
    for loaded in loaded_models(host):
        unload_model(host, loaded)
    time.sleep(1)

    # Snapshot VRAM before load
    vram_before = query_vram()

    # Load
    print(f"    loading ctx={ctx}...", end=" ", flush=True)
    try:
        load_model(host, model, ctx)
    except Exception as e:
        record["error"] = f"load failed: {e}"
        print(f"FAILED: {e}")
        return record

    print("loaded", end=" ", flush=True)

    # Snapshot VRAM after load
    vram_after = query_vram()
    if vram_before is not None and vram_after is not None:
        record["vram_delta_mb"] = vram_delta(vram_before, vram_after)
        record["vram_total_mb"] = vram_after
    record["ollama_size_vram_mb"] = ps_size_vram(host, model)

    # Benchmark
    print("benchmarking...", end=" ", flush=True)
    try:
        timing = run_benchmark(host, model, ctx)
        record.update(timing)
        print(f"decode={record.get('decode_tokens_per_sec')} tok/s")
    except Exception as e:
        record["error"] = f"benchmark failed: {e}"
        print(f"FAILED: {e}")

    # Unload
    unload_model(host, model)
    time.sleep(2)  # let VRAM settle before next run

    return record


def run_sweep(
    host: str,
    models: list[str],
    ctx_sizes: list[int],
    out_dir: Path,
    resume: Path | None = None,
) -> Path:
    existing: list[dict] = []
    if resume is not None:
        data = json.loads(resume.read_text())
        existing = data.get("results", [])
        out_path = resume
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"vram_bench_{timestamp}.json"

    done: set[tuple[str, int]] = {(r["model"], r["ctx"]) for r in existing}

    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "host": host,
            "context_sizes": ctx_sizes,
        },
        "results": list(existing),
    }

    for model in models:
        print(f"\n[{model}]")
        skip_remaining = False
        for ctx in ctx_sizes:
            if (model, ctx) in done:
                print(f"    ctx={ctx}: already done, skipping")
                continue
            if skip_remaining:
                print(f"    ctx={ctx}: skipped (previous OOM/error)")
                continue
            record = bench_model_ctx(host, model, ctx)
            output["results"].append(record)
            if record["error"]:
                skip_remaining = True

    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(output['results'])} records to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark VRAM and tok/s across models and context sizes.")
    p.add_argument("models", nargs="*", help="Models to benchmark (default: all generative from ollama list)")
    p.add_argument("--ctx", default=",".join(map(str, DEFAULT_CTX)),
                   help=f"Comma-separated context sizes (default: {','.join(map(str, DEFAULT_CTX))})")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory (default: output/)")
    p.add_argument("--resume", metavar="FILE", help="Resume from an existing results file, skipping already-done model+ctx pairs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx_sizes = [int(x) for x in args.ctx.split(",")]
    out_dir = Path(args.out)

    try:
        if args.models:
            models = args.models
        else:
            print(f"Discovering models from {args.host}...")
            all_models = list_models(args.host)
            models = [m["name"] for m in all_models if is_generative(m)]
            print(f"  Found {len(models)} generative models: {', '.join(models)}")
    except URLError as e:
        print(f"Error: cannot connect to Ollama at {args.host}: {e}", file=sys.stderr)
        sys.exit(1)

    if not models:
        print("No models to benchmark.", file=sys.stderr)
        sys.exit(1)

    resume = Path(args.resume) if args.resume else None
    run_sweep(args.host, models, ctx_sizes, out_dir, resume=resume)


if __name__ == "__main__":
    main()
