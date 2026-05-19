"""Greedy cascade optimizer — core computation and data loading.

Shared between the CLI script and potential library users.
"""

import json
from collections import defaultdict
from pathlib import Path


def load_model_stats(output_dir: Path) -> dict[str, dict[str, str]]:
    path = output_dir / "model_stats.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def resolve_model_name(name: str, stats: dict[str, dict] | None) -> str:
    resolved = name
    if stats and name in stats:
        digest = stats[name].get("digest")
        if digest:
            aliases: list[str] = [k for k, v in stats.items() if v.get("digest") == digest]
            if len(aliases) > 1:
                non_latest: list[str] = [a for a in aliases if not a.endswith(":latest")]
                candidates: list[str] = non_latest if non_latest else aliases
                candidates.sort(key=len)
                resolved = candidates[0]
    if resolved.endswith(":latest") and stats and resolved in stats:
        size = stats[resolved].get("parameter_size", "")
        if size:
            resolved = resolved[: -len(":latest")] + ":" + size.lower()
    return resolved


def load_vram_bench(output_dir: Path, ctx: int = 8192) -> dict[str, int]:
    """Return {model_name: vram_mb} from the most recent vram_bench_*.json at the given context."""
    bench_files = sorted(output_dir.glob("vram_bench_*.json"), reverse=True)
    if not bench_files:
        return {}
    data = json.loads(bench_files[0].read_text())
    vram = {}
    for rec in data.get("results", []):
        if rec.get("ctx") != ctx or rec.get("error"):
            continue
        mb = rec.get("ollama_size_vram_mb")
        if mb is not None:
            vram[rec["model"]] = mb
    return vram


# Empirically calibrated from 37 models with confirmed cold-load measurements.
_LOAD_BANDWIDTH_MBS = 1785
_COLD_LOAD_THRESHOLD_S = 1.0


def calibrate_load_times(
    measured: dict[str, float],
    vram_per_model: dict[str, int],
) -> dict[str, float]:
    """Replace unreliably-low load times with VRAM-based estimates."""
    result = {}
    for model, t in measured.items():
        if t >= _COLD_LOAD_THRESHOLD_S:
            result[model] = t
        elif vram_per_model.get(model):
            result[model] = vram_per_model[model] / _LOAD_BANDWIDTH_MBS
        else:
            result[model] = t
    return result


def parse_jsonl(path: Path, _model_stats: dict) -> tuple[dict, float]:
    """Return ({task_id: {"first_pass_iter": int|None, "iter_durations": [float]}}, load_time_s)."""
    tasks = {}
    max_load_ns = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = rec["input"]["task_id"]
            iters = rec.get("iterations", [])
            iter_durations = [(it.get("total_duration") or 0) / 1e9 for it in iters]
            first_pass_iter = None
            for i, it in enumerate(iters):
                if (it.get("test_result") or {}).get("exit_code") == 0:
                    first_pass_iter = i + 1
                    break
                load_ns = it.get("load_duration") or 0
                if load_ns > max_load_ns:
                    max_load_ns = load_ns
            tasks[task_id] = {
                "first_pass_iter": first_pass_iter,
                "iter_durations": iter_durations,
            }
    return tasks, max_load_ns / 1e9


def _dataset_variant(stem: str) -> str:
    """Extract the dataset variant prefix from a result filename stem."""
    s = stem.removeprefix("results_")
    for suf in ("_nothink", "_think"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    known: list[str] = [
        "human-eval-enhanced-202307", "humaneval-rewritten-gpt-oss20b",
        "humaneval-rewritten-ministral-314b", "humaneval-rewritten-qwen34b",
        "humaneval-rewritten-qwen3-coder", "humaneval",
    ]
    known.sort(key=len, reverse=True)
    for k in known:
        if s.startswith(k + "_"):
            return k
    return s.split("_")[0]


def load_all_models(
    output_dir: Path, dataset: str, model_stats: dict
) -> tuple[dict[str, dict], dict[str, float]]:
    """Load all JSONL files for the given dataset, grouped by variant."""
    by_variant: dict[str, dict[str, dict]] = defaultdict(dict)
    load_times: dict[str, float] = {}

    for path in sorted(output_dir.glob("*.jsonl")):
        name = path.stem
        if "champion" in name or "cascade" in name:
            continue
        if dataset not in name and dataset.replace("-", "") not in name.replace("-", ""):
            continue
        variant = _dataset_variant(name)
        if "rewritten" in variant and "rewritten" not in dataset:
            continue

        # Resolve the canonical model name from the file contents.
        model_name = name
        try:
            with open(path) as f:
                first = json.loads(f.readline())
            raw = first.get("model", name)
            model_name = resolve_model_name(raw, model_stats)
        except Exception:
            pass

        tasks, load_time_s = parse_jsonl(path, model_stats)
        if not tasks:
            continue

        # Tag think-mode variants so they appear as separate models.
        if "_think" in name and "_nothink" not in name:
            model_name = f"{model_name}:think"

        by_variant[variant][model_name] = tasks
        load_times[model_name] = load_time_s

    if not by_variant:
        return {}, {}

    best_variant = max(by_variant, key=lambda v: (len(by_variant[v]), v))
    if len(by_variant) > 1:
        counts = {v: len(m) for v, m in by_variant.items()}
        print(f"Multiple dataset variants found: {counts}")
        print(f"Using '{best_variant}' ({counts[best_variant]} models). "
              "Pass --dataset with a more specific name to override.")
    return by_variant[best_variant], load_times


def model_summary(tasks: dict, budget: int, task_ids: set | None = None) -> dict:
    """Compute stats for this model at a given budget.

    avg_time_s is the mean per-task time cost at this budget:
    - passing tasks pay cumulative time up to (and including) their pass iteration
    - failing tasks pay cumulative time for all `budget` iterations (then escalate)
    """
    if task_ids is not None:
        tasks = {tid: t for tid, t in tasks.items() if tid in task_ids}
    total_tasks = len(tasks)
    if total_tasks == 0:
        return {"pass_rate": 0.0, "passing_tasks": set(), "avg_time_s": 0.0}
    passing_tasks = set()
    times: list[float] = []
    for tid, t in tasks.items():
        fp = t["first_pass_iter"]
        durs = t["iter_durations"]
        if fp is not None and fp <= budget:
            passing_tasks.add(tid)
            times.append(sum(durs[:fp]))
        else:
            times.append(sum(durs[:budget]))
    return {
        "pass_rate": len(passing_tasks) / total_tasks,
        "passing_tasks": passing_tasks,
        "avg_time_s": sum(times) / len(times) if times else 0.0,
    }


def _vram_mb(model: str, vram_per_model: dict[str, int]) -> int | None:
    """Look up VRAM for a model, handling :think suffix."""
    if model in vram_per_model:
        return vram_per_model[model]
    if model.endswith(":think"):
        return vram_per_model.get(model.removesuffix(":think"))
    return None


def cascade_yield_score(
    cascade: list[tuple[str, int, dict]], tau: float, total_tasks: int
) -> float:
    """Yield score: Σ_k n_passing_at_k / (1 + cumul_time_k / τ) / total_tasks * 100."""
    if not cascade or total_tasks == 0:
        return 0.0
    cumulative_time = 0.0
    total = 0.0
    for _, _, stats in cascade:
        cumulative_time += stats["avg_time_s"]
        discount = 1.0 / (1.0 + cumulative_time / tau)
        total += stats["marginal_tasks"] * discount
    return total / total_tasks * 100


def _best_new_model(
    remaining_models: set[str],
    model_data: dict[str, dict],
    uncovered: set[str],
    max_budget: int,
    cumulative_vram: int,
    vram_per_model: dict[str, int],
    vram_budget_mb: int | None,
    tau: float | None = None,
    cumulative_time_before: float = 0.0,
) -> tuple[list, list[str]]:
    """Find all viable new-model candidates sorted by marginal yield (or coverage).

    Returns ([(score, model, budget, summary), ...], vram_candidates).
    """
    new_candidates = list(remaining_models)
    if vram_budget_mb is not None:
        # Only exclude models whose VRAM alone exceeds the budget (physically impossible).
        # Models that overflow when added cumulatively are still allowed — they pay a
        # load-time penalty in _cascade_expected_time instead.
        new_candidates = [
            m for m in new_candidates
            if (_vram_mb(m, vram_per_model) or 0) <= vram_budget_mb
        ]
    candidates: list[tuple[float, str, int, dict]] = []
    for m in new_candidates:
        for b in range(1, max_budget + 1):
            s = model_summary(model_data[m], b, task_ids=uncovered)
            n = len(s["passing_tasks"])
            if n > 0:
                if tau is not None:
                    discount = 1.0 / (1.0 + (cumulative_time_before + s["avg_time_s"]) / tau)
                    score = n * discount
                else:
                    score = float(n)
                candidates.append((score, m, b, s))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates, new_candidates


def _best_extension(
    entries: list,
    model_data: dict[str, dict],
    max_budget: int,
    tau: float | None = None,
    cumulative_time_before: float = 0.0,
) -> tuple[float, tuple | None]:
    """Find best budget extension for the last tier, scored by marginal yield (or coverage).

    Returns (best_score, (new_budget, summary, extra_tasks, delta_t) | None).
    cumulative_time_before is the sum of avg_time_s for all tiers *before* the last.
    """
    if not entries:
        return -1, None
    last_model, last_budget, entry_unc, last_passing, _ = entries[-1]
    s_base = model_summary(model_data[last_model], last_budget, task_ids=entry_unc)
    best_score: float = -1
    best: tuple | None = None
    for new_b in range(last_budget + 1, max_budget + 1):
        s_new = model_summary(model_data[last_model], new_b, task_ids=entry_unc)
        extra = s_new["passing_tasks"] - last_passing
        if not extra:
            continue
        delta_t = s_new["avg_time_s"] - s_base["avg_time_s"]
        if tau is not None:
            discount = 1.0 / (1.0 + (cumulative_time_before + s_new["avg_time_s"]) / tau)
            score = len(extra) * discount
        else:
            score = float(len(extra))
        if score > best_score:
            best_score = score
            best = (new_b, s_new, extra, max(delta_t, 0.0))
    return best_score, best


def _cascade_expected_time(
    entries: list,
    total_tasks: int,
    vram_budget_mb: int | None = None,
    model_load_times: dict[str, float] | None = None,
    vram_per_model: dict[str, int] | None = None,
) -> float:
    """Compute the global expected time per task for the current cascade.

    E[t] = Σ_k (n_entering_k / N) * inference_time_k
         + Σ_{k not co-resident} (n_entering_k / N) * load_time_k

    Models are co-resident if their cumulative VRAM (tier 1 .. k) fits within
    vram_budget_mb.  Models that overflow must be cold-loaded each time a task
    reaches them, so they pay their full load_time amortised by entry fraction.
    """
    if not entries or total_tasks == 0:
        return 0.0
    etct = 0.0
    cumulative_vram = 0
    for model, _, entry_unc, _, stats in entries:
        n_entry = stats.get("n_uncovered", len(entry_unc))
        etct += (n_entry / total_tasks) * stats["avg_time_s"]
        if vram_budget_mb is not None and vram_per_model is not None:
            model_vram = _vram_mb(model, vram_per_model) or 0
            if cumulative_vram + model_vram <= vram_budget_mb:
                cumulative_vram += model_vram  # co-resident, no load penalty
            else:
                load_t = (model_load_times or {}).get(model, 0.0)
                etct += (n_entry / total_tasks) * load_t
    return etct


def _apply_new_model(
    best_new: tuple, vram_per_model: dict[str, int], model_load_times: dict | None,
    remaining_models: set[str], covered: set[str], uncovered: set[str],
    cumulative_vram: int,
) -> tuple:
    """Apply a new-model action and return the updated state."""
    m, b, s = best_new
    marginal = len(s["passing_tasks"])
    n_uncov = len(uncovered)
    model_vram = _vram_mb(m, vram_per_model)
    cumulative_vram += model_vram or 0
    remaining_models.remove(m)
    entry_unc = set(uncovered)
    covered |= s["passing_tasks"]
    uncovered -= s["passing_tasks"]
    stats = {
        "pass_rate": s["pass_rate"],
        "marginal_tasks": marginal,
        "cumulative_coverage": len(covered),
        "avg_time_s": s["avg_time_s"],
        "n_uncovered": n_uncov,
        "entry_uncovered": entry_unc,
        "load_time_s": (model_load_times or {}).get(m, 0.0),
        "vram_mb": model_vram,
        "cumulative_vram_mb": cumulative_vram if model_vram is not None else None,
    }
    return (m, b, entry_unc, set(s["passing_tasks"]), stats), cumulative_vram


def greedy_cascade(
    model_data: dict[str, dict],
    vram_per_model: dict[str, int],
    tau: float,
    vram_budget_mb: int | None = None,
    max_tiers: int | None = None,
    model_load_times: dict[str, float] | None = None,
) -> list[tuple[str, int, dict]]:
    """Build a greedy cascade that maximises yield (speed-discounted coverage) under a time budget.

    τ (tau) is the half-credit time: the latency at which a correct answer counts for half
    of what an instant answer would.  Actions are scored by marginal yield contribution
    (n_tasks_solved / (1 + cumulative_time / τ)), and admitted only if the cascade's
    expected time per task does not exceed τ.

    Args:
        model_data: {model_name: {task_id: {first_pass_iter, iter_durations}}}
        vram_per_model: {model_name: vram_mb} from VRAM bench.
        tau: Half-credit time in seconds (also used as the ETCT hard cap).
        vram_budget_mb: Total VRAM in MB available for concurrent models.
        max_tiers: Optional hard cap on cascade length.
        model_load_times: Optional dict of measured load times (stats only).
    """
    max_budget = max(
        (len(t["iter_durations"]) for tasks in model_data.values() for t in tasks.values()),
        default=1,
    )
    all_task_ids: set[str] = set(next(iter(model_data.values())).keys())
    total_tasks = len(all_task_ids)
    covered: set[str] = set()
    uncovered = set(all_task_ids)
    entries: list = []
    remaining_models = set(model_data.keys())
    cumulative_vram = 0

    while uncovered:
        if max_tiers is not None and len(entries) >= max_tiers:
            print(f"  (max_tiers={max_tiers} reached)")
            break

        # Cumulative time already spent by tasks that reach the next tier.
        cumul_t = sum(s["avg_time_s"] for _, _, _, _, s in entries)
        # Cumulative time before the last tier (for extension scoring).
        cumul_t_before_last = sum(s["avg_time_s"] for _, _, _, _, s in entries[:-1])

        new_candidates, _ = _best_new_model(
            remaining_models, model_data, uncovered, max_budget,
            cumulative_vram, vram_per_model, vram_budget_mb,
            tau=tau, cumulative_time_before=cumul_t,
        )
        best_ext_score, best_ext = _best_extension(
            entries, model_data, max_budget,
            tau=tau, cumulative_time_before=cumul_t_before_last,
        )

        if not new_candidates and best_ext is None:
            print("  (no valid candidates remain)")
            break

        # Try extension first if it scores higher than the best new-model.
        top_new_score = new_candidates[0][0] if new_candidates else -1
        use_extension = best_ext is not None and best_ext_score >= top_new_score

        action_taken = False

        if use_extension:
            assert best_ext is not None
            new_b, s_new, extra, _delta_t = best_ext
            last_model, _last_budget, entry_unc, last_passing, last_stats = entries[-1]
            trial_stats = dict(last_stats)
            trial_stats["avg_time_s"] = s_new["avg_time_s"]
            trial_stats["pass_rate"] = s_new["pass_rate"]
            trial_entries = list(entries)
            trial_entries[-1] = (last_model, new_b, entry_unc, last_passing | extra, trial_stats)
            trial_etct = _cascade_expected_time(
                trial_entries, total_tasks, vram_budget_mb, model_load_times, vram_per_model,
            )
            if trial_etct <= tau:
                covered |= extra
                uncovered -= extra
                last_stats["marginal_tasks"] += len(extra)
                last_stats["cumulative_coverage"] = len(covered)
                last_stats["pass_rate"] = s_new["pass_rate"]
                last_stats["avg_time_s"] = s_new["avg_time_s"]
                entries[-1] = (last_model, new_b, entry_unc, last_passing | extra, last_stats)
                action_taken = True
            # If extension violates budget, fall through to try new models.

        if not action_taken:
            # Walk the new-model candidates in descending yield-score order,
            # picking the first one that fits within the time budget.
            chosen = None
            for _score, m, b, s in new_candidates:
                trial_entry, _ = _apply_new_model(
                    (m, b, s), vram_per_model, model_load_times,
                    set(remaining_models), set(covered), set(uncovered), cumulative_vram,
                )
                trial_entries = list(entries) + [trial_entry]
                trial_etct = _cascade_expected_time(
                    trial_entries, total_tasks, vram_budget_mb, model_load_times, vram_per_model,
                )
                if trial_etct <= tau:
                    chosen = (m, b, s)
                    break

            if chosen is not None:
                entry, cumulative_vram = _apply_new_model(
                    (chosen[0], chosen[1], chosen[2]),
                    vram_per_model, model_load_times,
                    remaining_models, covered, uncovered, cumulative_vram,
                )
                entries.append(entry)
                action_taken = True

        if not action_taken:
            print(f"  (stopping: no action fits within {tau:.0f}s/task budget)")
            break

    return [(m, b, stats) for m, b, _, _, stats in entries]
