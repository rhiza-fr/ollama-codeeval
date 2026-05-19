"""Greedy cascade optimizer — CLI entry point.

Reads per-model JSONL result files and builds a cascade (model sequence +
per-tier iteration budget) that maximises task coverage under a target
expected-time-per-task constraint, subject to a VRAM budget.

Usage:
    uv run python scripts/optimize_cascade.py --target-time 15 --vram-budget 28000
"""

import argparse
from pathlib import Path

from ollama_codeeval.cascade_optimizer import (
    load_model_stats,
    load_vram_bench,
    calibrate_load_times,
    load_all_models,
    greedy_cascade,
)
from ollama_codeeval.cascade_report import (
    retrospective_check,
    prune_harmful_tiers,
    print_cascade,
    reorder_cascade_for_yield,
    maximize_last_tier,
)


def main():
    parser = argparse.ArgumentParser(description="Greedy cascade optimizer")
    parser.add_argument("--dataset", default="humaneval", help="Dataset name filter")
    parser.add_argument(
        "--tau",
        "-t",
        type=float,
        default=15.0,
        help="Half-credit time τ in seconds. Controls the yield discount: a correct answer "
        "at time t contributes 1/(1+t/τ) to the score. Also used as the hard ETCT cap "
        "— the cascade grows until adding any tier would exceed this. Default: 15.0",
    )
    parser.add_argument(
        "--max-tiers",
        type=int,
        default=None,
        help="Optional hard cap on cascade depth (unlimited by default).",
    )
    parser.add_argument(
        "--output", default="output", help="Output directory with JSONL files"
    )
    parser.add_argument(
        "--vram-budget",
        type=int,
        default=12288,
        help="Total VRAM budget in MB (default: 12288 = 12 GB). Models that fit "
        "cumulatively are kept resident; models that overflow pay a cold-load "
        "time penalty amortised by the fraction of tasks that reach them.",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=8192,
        help="Context size to use when reading VRAM bench data (default: 8192)",
    )

    parser.add_argument(
        "--prune",
        action="store_true",
        help="After building the cascade, iteratively remove net-harmful tiers.",
    )
    parser.add_argument(
        "--no-maximize-last",
        action="store_true",
        help="Skip the final maximize-coverage tier (appended after reorder, ignores ETCT).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    model_stats = load_model_stats(output_dir)

    vram_per_model = load_vram_bench(output_dir, ctx=args.ctx)
    if vram_per_model:
        print(
            f"Loaded VRAM data for {len(vram_per_model)} models from vram bench (ctx={args.ctx})"
        )
    else:
        print("No vram_bench_*.json found — load-time penalties will not be applied")

    print(f"Loading results from {output_dir} (dataset={args.dataset})...")
    model_data, model_load_times = load_all_models(
        output_dir, args.dataset, model_stats
    )
    model_load_times = calibrate_load_times(model_load_times, vram_per_model)
    print(f"Loaded {len(model_data)} models.")

    if not model_data:
        print("No matching JSONL files found.")
        return

    task_counts = {m: len(tasks) for m, tasks in model_data.items()}
    max_count = max(task_counts.values())
    threshold = int(max_count * 0.9)
    dropped = [m for m, c in task_counts.items() if c < threshold]
    if dropped:
        print(
            f"Dropping {len(dropped)} partial-run models (< {threshold} tasks): {dropped}"
        )
        model_data = {m: t for m, t in model_data.items() if m not in dropped}

    all_task_sets = [set(tasks.keys()) for tasks in model_data.values()]
    common_tasks = set.intersection(*all_task_sets)
    total_tasks = len(common_tasks)
    print(f"Common tasks across {len(model_data)} models: {total_tasks}")

    filtered = {
        m: {tid: t for tid, t in tasks.items() if tid in common_tasks}
        for m, tasks in model_data.items()
    }

    print(
        f"VRAM budget: {args.vram_budget:,} MB (models overflowing pay cold-load penalty per task)"
    )

    print(f"Building cascade with τ={args.tau:.1f}s...")
    cascade = greedy_cascade(
        filtered,
        vram_per_model,
        args.tau,
        vram_budget_mb=args.vram_budget,
        max_tiers=args.max_tiers,
        model_load_times=model_load_times,
    )

    excluded = {
        m: t for m, t in filtered.items() if m not in {mo for mo, _, _ in cascade}
    }
    print(f"\nOptimising tier order and trying {len(excluded)} excluded models...")
    cascade = reorder_cascade_for_yield(
        cascade,
        filtered,
        args.tau,
        args.tau,
        common_tasks,
        excluded_models=excluded,
        model_load_times=model_load_times,
        vram_per_model=vram_per_model,
    )

    if not args.no_maximize_last:
        print("\nMaximising last tier...")
        cascade = maximize_last_tier(
            cascade,
            filtered,
            common_tasks,
            model_load_times,
            vram_per_model,
        )

    if args.prune:
        print("\nPruning harmful tiers...")
        cascade = prune_harmful_tiers(
            cascade,
            filtered,
            common_tasks,
            model_load_times,
            vram_per_model,
        )

    retro = retrospective_check(cascade, filtered, total_tasks)
    print_cascade(cascade, total_tasks, retro, tau=args.tau)


if __name__ == "__main__":
    main()
