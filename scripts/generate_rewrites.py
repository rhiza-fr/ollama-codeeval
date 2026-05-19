"""Generate a rewritten-prompt dataset from an Ollama model.

Reads the HumanEval dataset, rewrites each prompt using the specified model,
and writes a new dataset to data/humaneval-rewritten-{model}.jsonl.gz.

Each output row is the original row with:
  - `prompt`               replaced by the rewritten text
  - `original_input_prompt` added, containing the original HumanEval prompt

Already-cached rewrites are reused from diskcache (compatible with the old
--rewrite-model flow), so existing runs don't need to be re-generated.

Usage:
    uv run python scripts/generate_rewrites.py --model qwen3:4b
    uv run python scripts/generate_rewrites.py --model gpt-oss:20b --think
"""

import argparse
import gzip
import json
import re
import sys
import unicodedata
from pathlib import Path

# Ensure the src layout is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diskcache import Cache

from ollama_codeeval.autocontext import AutoContextClient
from ollama_codeeval.config import EXECUTION_CACHE_DIR, OLLAMA_HOST
from ollama_codeeval.data import load_data
from ollama_codeeval.prompts import REWRITE_SYSTEM_PROMPT

DATA_DIR = Path(__file__).parent.parent / "data"


def slugify(value: str) -> str:
    value = str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s.-]", "", value.lower())
    value = value.replace(".", "p")
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def main():
    parser = argparse.ArgumentParser(description="Generate rewritten HumanEval dataset")
    parser.add_argument("--model", required=True, help="Ollama model to use for rewriting")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: data/humaneval-rewritten-{model}.jsonl.gz)")
    args = parser.parse_args()

    model_slug = slugify(args.model)
    output_path = args.output or DATA_DIR / f"humaneval-rewritten-{model_slug}.jsonl.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_data()
    cache = Cache(EXECUTION_CACHE_DIR)
    client = AutoContextClient(host=OLLAMA_HOST)

    # Find which tasks are missing from cache
    missing = [r for r in dataset if not cache.get(f"rewrite_{args.model}_{r['task_id']}")]

    if missing:
        print(f"Generating {len(missing)} rewrites with {args.model} ...")
        client.call(args.model, "Say hello.", think=False, options={})
        for i, row in enumerate(missing, 1):
            prompt = REWRITE_SYSTEM_PROMPT.format(question=row["prompt"])
            response = client.call(args.model, prompt, think=args.think, options={"temperature": 0.3})
            rewritten = response.to_dict()["message"]["content"]
            cache.set(f"rewrite_{args.model}_{row['task_id']}", rewritten)
            print(f"  [{i}/{len(missing)}] {row['task_id']}")
        client.stop(args.model)
    else:
        print(f"All {len(dataset)} rewrites already cached.")

    # Write dataset
    print(f"Writing {output_path} ...")
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        for row in dataset:
            rewritten = cache.get(f"rewrite_{args.model}_{row['task_id']}")
            if not rewritten:
                print(f"WARNING: no cached rewrite for {row['task_id']}, using original prompt")
                rewritten = row["prompt"]
            out_row = dict(row)
            out_row["original_input_prompt"] = row["prompt"]
            out_row["prompt"] = rewritten
            f.write(json.dumps(out_row) + "\n")

    print(f"Done. {len(dataset)} rows written to {output_path}")
    print(f"Dataset name for --dataset-name: humaneval-rewritten-{model_slug}")


if __name__ == "__main__":
    main()
