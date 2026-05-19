"""Compare FixHarder prompt variants against a baseline.

Usage:
    uv run python scripts/compare_fixharder.py --tag fixharder-v2
    uv run python scripts/compare_fixharder.py --tag fixharder-v2 --verbose
    uv run python scripts/compare_fixharder.py --rank

For each model that has both a baseline and a tagged results file, prints:
- Pass rate on all tasks (baseline vs new)
- Pass rate on tasks that hit FixHarderNode in the baseline run
- "Genuine" fixes: baseline hit FH, new run also hit FH, new run passes
- "Lucky" fixes: baseline hit FH, new run passed before reaching FH (re-sample noise)

A fix is only attributable to the new FixHarder prompt if the new run actually
reached FixHarderNode. Tasks solved on iter 0 or 1 in the new run are noise.
"""

import argparse
import glob
import json
import re
from pathlib import Path


def is_stuck(iterations: list) -> bool:
    """Return True if any two consecutive iterations produced identical code (FixHarder trigger)."""
    for i in range(1, len(iterations)):
        c1 = iterations[i - 1].get("message", {}).get("content", "")
        c2 = iterations[i].get("message", {}).get("content", "")
        if c1 and c1 == c2:
            return True
    return False


def load_file(path: str) -> dict:
    """Load a JSONL results file into {task_id: record}."""
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            task_id = r["input"]["task_id"]
            results[task_id] = {
                "pass": r["final_result"]["exit_code"] == 0,
                "n_iterations": len(r["iterations"]),
                "hit_fixharder": is_stuck(r["iterations"]),
            }
    return results


def summarise(results: dict, task_ids: list) -> tuple[int, int]:
    """Return (pass_count, total) for the given task_ids."""
    passed = sum(1 for t in task_ids if results.get(t, {}).get("pass", False))
    return passed, len(task_ids)


def find_pairs(output_dir: str, tag: str) -> list[tuple[str, str, str]]:
    """Return list of (model_label, baseline_path, new_path) for matched pairs."""
    tag_slug = re.sub(r"[-\s]+", "-", tag.lower()).strip("-_")
    pattern = str(Path(output_dir) / f"results_*_{tag_slug}.jsonl")
    new_files = glob.glob(pattern)
    pairs = []
    for new_path in sorted(new_files):
        baseline_path = new_path.replace(f"_{tag_slug}", "")
        if not Path(baseline_path).exists():
            print(f"  [skip] no baseline found for {new_path}")
            continue
        stem = Path(new_path).stem
        label = stem.replace(f"_{tag_slug}", "").replace("results_", "")
        pairs.append((label, baseline_path, new_path))
    return pairs


def compare(output_dir: str, tag: str, verbose: bool) -> None:
    pairs = find_pairs(output_dir, tag)
    if not pairs:
        print(f"No tagged files found in '{output_dir}' with tag '{tag}'.")
        return

    print(f"\n{'Model':<40} {'All tasks':>18}  {'FH pool':>12}  {'Genuine':>8}  {'Lucky':>6}  {'Regressed':>10}")
    print("-" * 102)

    total_genuine = total_lucky = total_regressed = total_fh = 0

    for label, baseline_path, new_path in pairs:
        baseline = load_file(baseline_path)
        new = load_file(new_path)

        all_tasks = list(baseline.keys())
        # Tasks that hit FixHarder in the baseline — the only ones affected by the prompt change
        fh_tasks = [t for t in all_tasks if baseline[t]["hit_fixharder"]]

        b_all, n_total = summarise(baseline, all_tasks)
        n_all, _ = summarise(new, all_tasks)
        b_fh, fh_total = summarise(baseline, fh_tasks)
        n_fh, _ = summarise(new, fh_tasks)

        # Genuine fix: was failing in baseline, new run also hit FH (prompt fired), now passes
        genuine = [
            t for t in fh_tasks
            if not baseline[t]["pass"]
            and new.get(t, {}).get("hit_fixharder", False)
            and new.get(t, {}).get("pass", False)
        ]
        # Lucky fix: was failing in baseline, new run did NOT hit FH (solved before reaching it)
        lucky = [
            t for t in fh_tasks
            if not baseline[t]["pass"]
            and not new.get(t, {}).get("hit_fixharder", False)
            and new.get(t, {}).get("pass", False)
        ]
        # Regression: was passing in baseline, now failing (regardless of FH in new run)
        regressions = [
            t for t in fh_tasks
            if baseline[t]["pass"] and not new.get(t, {}).get("pass", True)
        ]

        total_genuine += len(genuine)
        total_lucky += len(lucky)
        total_regressed += len(regressions)
        total_fh += fh_total

        all_str = f"{b_all}/{n_total} -> {n_all}/{n_total}"
        fh_str = f"{b_fh}/{fh_total} -> {n_fh}/{fh_total}" if fh_total else "none"

        print(f"{label:<40} {all_str:>18}  {fh_str:>12}  {len(genuine):>8}  {len(lucky):>6}  {len(regressions):>10}")

        if verbose and fh_tasks:
            if genuine:
                print(f"  + genuine:   {', '.join(genuine)}")
            if lucky:
                print(f"  ~ lucky:     {', '.join(lucky)}")
            if regressions:
                print(f"  - regressed: {', '.join(regressions)}")

    print("-" * 102)
    print(f"{'TOTAL across ' + str(total_fh) + ' FH tasks':<40} {'':>18}  {'':>12}  {total_genuine:>8}  {total_lucky:>6}  {total_regressed:>10}")
    print()
    print(f"  Genuine fixes (prompt attributable): {total_genuine}")
    print(f"  Lucky fixes   (re-sample noise):     {total_lucky}")
    print(f"  Regressions:                         {total_regressed}")
    if total_genuine + total_lucky > 0:
        noise_pct = 100 * total_lucky / (total_genuine + total_lucky)
        print(f"  Noise fraction of apparent fixes:    {noise_pct:.0f}%")
    print()


def top_stuck_models(output_dir: str, exclude_tag: str | None = None) -> None:
    """Print models ranked by how many tasks hit FixHarderNode (helps pick re-run targets)."""
    from collections import Counter
    counts: Counter = Counter()
    pattern = str(Path(output_dir) / "results_*.jsonl")
    for path in sorted(glob.glob(pattern)):
        stem = Path(path).stem
        if "rewrite" in stem:
            continue
        if exclude_tag and exclude_tag in stem:
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if is_stuck(r["iterations"]):
                    counts[r["model"]] += 1

    print("\nModels ranked by FixHarder triggers (baseline runs only):")
    print(f"  {'Model':<40} {'Stuck tasks':>12}")
    print("  " + "-" * 54)
    for model, n in counts.most_common():
        print(f"  {model:<40} {n:>12}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="fixharder-v2", help="Tag suffix used when running new evals")
    parser.add_argument("--output-dir", default="output", help="Directory containing JSONL result files")
    parser.add_argument("--verbose", action="store_true", help="Show per-task genuine/lucky/regressed lists")
    parser.add_argument("--rank", action="store_true", help="Show models ranked by FixHarder trigger count (no tag needed)")
    args = parser.parse_args()

    if args.rank:
        top_stuck_models(args.output_dir, exclude_tag=args.tag)
    else:
        compare(args.output_dir, args.tag, args.verbose)
