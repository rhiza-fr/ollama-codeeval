#!/usr/bin/env python
# src/ollama_codeeval/contamination/contamination_report.py
"""Compute contamination scores by comparing original vs. variant evaluation results.

Usage:
  uv run python -m ollama_codeeval.contamination.contamination_report \\
      output/results_qwen3_nothink.jsonl \\
      output/results_qwen3_nothink_variant.jsonl
"""

import argparse
import json
from pathlib import Path

from ollama_codeeval.contamination.variant_utils import contamination_score
from ollama_codeeval.jsonl_io import read_jsonl


def pair_results(
    originals: list[dict],
    variants: list[dict],
) -> list[tuple[str, bool, bool]]:
    """Match original and variant result records by task_id.

    Returns list of (original_task_id, pass_original, pass_variant).
    Variant records must have input.original_task_id to enable matching.
    """
    orig_by_id = {r["input"]["task_id"]: r for r in originals}
    pairs = []
    for v in variants:
        orig_id = v["input"].get("original_task_id")
        if orig_id not in orig_by_id:
            continue
        o = orig_by_id[orig_id]
        pass_orig = o["final_result"]["exit_code"] == 0
        pass_var = v["final_result"]["exit_code"] == 0
        pairs.append((orig_id, pass_orig, pass_var))
    return pairs


def _print_table(rows: list[tuple], score: float) -> None:
    header = f"{'task_id':<35} {'orig':>6} {'var':>6} {'signal':>8}"
    print(header)
    print("-" * len(header))
    for task_id, pass_orig, pass_var in rows:
        signal = ""
        if pass_orig and not pass_var:
            signal = "CONTAM"
        elif not pass_orig and pass_var:
            signal = "reverse"
        orig_s = "PASS" if pass_orig else "fail"
        var_s = "PASS" if pass_var else "fail"
        print(f"{task_id:<35} {orig_s:>6} {var_s:>6} {signal:>8}")
    print("-" * len(header))
    print(f"{'Contamination score':>35}  {score:+.3f}")
    n_contam = sum(1 for _, o, v in rows if o and not v)
    n_reverse = sum(1 for _, o, v in rows if not o and v)
    n_both = sum(1 for _, o, v in rows if o and v)
    n_neither = sum(1 for _, o, v in rows if not o and not v)
    print(
        f"  pass both: {n_both}  fail both: {n_neither}  contam signal: {n_contam}  reverse: {n_reverse}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compute contamination scores")
    parser.add_argument("original_results", help="Path to original eval results JSONL")
    parser.add_argument("variant_results", help="Path to variant eval results JSONL")
    parser.add_argument("--json-out", help="Optional path to write JSON report")
    args = parser.parse_args()

    originals = read_jsonl(args.original_results)
    variants = read_jsonl(args.variant_results)
    pairs = pair_results(originals, variants)

    if not pairs:
        has_orig_id = any(v["input"].get("original_task_id") for v in variants)
        if not has_orig_id:
            print(
                "No matching pairs found. Variant records are missing 'original_task_id' in "
                "their input field. Make sure variants were evaluated with the full task dicts "
                "from variants.jsonl.gz (not stripped-down ones)."
            )
        else:
            print(
                "No matching pairs found. The original_task_id values in variants do not match "
                "any task_id in the original results file."
            )
        return

    bool_pairs = [(o, v) for _, o, v in pairs]
    score = contamination_score(bool_pairs)

    model_name = Path(args.original_results).stem
    print(f"\nContamination report — {model_name}")
    print(f"Matched tasks: {len(pairs)}\n")
    _print_table(pairs, score)

    if args.json_out:
        report = {
            "model": model_name,
            "n_tasks": len(pairs),
            "contamination_score": score,
            "tasks": [
                {"task_id": tid, "pass_original": o, "pass_variant": v}
                for tid, o, v in pairs
            ],
        }
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
