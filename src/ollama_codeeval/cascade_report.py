"""Cascade reporting: print tables and compute retrospective/pruning analysis."""

from itertools import permutations

from ollama_codeeval.cascade_optimizer import _vram_mb, cascade_yield_score, model_summary


def print_cascade(
    cascade: list[tuple[str, int, dict]],
    total_tasks: int,
    retro: list[float] | None = None,
    tau: float | None = None,
) -> None:
    has_vram = any(s.get("vram_mb") is not None for _, _, s in cascade)
    header = (
        f"{'Tier':<6} {'Model':<35} {'Bgt':<4} {'Pass%':<7} {'Mgn':>4}  "
        f"{'Coverage':<18} {'AvgTime':>8}  {'LoadTime':>8}  {'ETCT':>6}  {'Cost/t':>7}"
        f"  {'Retro':>7}"
    )
    if has_vram:
        header += "  VRAM (cumul)"
    print(f"\n{header}")
    print("-" * len(header))

    cumulative_etct = 0.0
    for i, (model, budget, stats) in enumerate(cascade):
        n_uncov = stats.get("n_uncovered", 0)
        cov_pct = stats["cumulative_coverage"] / total_tasks * 100 if total_tasks else 0
        marginal = stats["marginal_tasks"]
        avg_t = stats["avg_time_s"]
        etct = (n_uncov / total_tasks) * avg_t if total_tasks else 0.0
        cost_t = (n_uncov * avg_t) / marginal if marginal else 0.0
        cumulative_etct += etct
        if retro is not None and i < len(retro):
            r = retro[i]
            retro_str = f"{r:+.3f}s"
            retro_flag = " !" if r < 0 else "  "
        else:
            retro_str = "  n/a "
            retro_flag = "  "
        line = (
            f"  {i+1:<4} {model:<35} {budget:<4} "
            f"{stats['pass_rate']*100:>5.1f}%  {marginal:>4}  "
            f"{stats['cumulative_coverage']}/{total_tasks} ({cov_pct:.1f}%)  "
            f"{avg_t:>6.1f}s  "
            f"{stats.get('load_time_s', 0):>7.1f}s  "
            f"{etct:>5.3f}s  "
            f"{cost_t:>6.1f}s  "
            f"{retro_str:>7}{retro_flag}"
        )
        if has_vram:
            vram = stats.get("vram_mb")
            cum = stats.get("cumulative_vram_mb")
            line += f"  {vram:>6} MB (sum {cum:,})" if vram is not None else "  n/a"
        print(line)

    if cascade:
        _, _, last = cascade[-1]
        cov = last["cumulative_coverage"]
        cov_pct = cov / total_tasks * 100 if total_tasks else 0
        vram_str = f" | VRAM: {last['cumulative_vram_mb']:,} MB" if last.get("cumulative_vram_mb") else ""
        yield_str = ""
        if tau is not None:
            y = cascade_yield_score(cascade, tau, total_tasks)
            yield_str = f" | Yield@τ={tau:.0f}s: {y:.1f}%"
        print(f"\n  ETCT: {cumulative_etct:.3f}s/task | Coverage: {cov_pct:.1f}% ({cov}/{total_tasks}){vram_str}{yield_str}\n")

    print("CASCADE = [")
    for model, budget, _ in cascade:
        print(f'    ("{model}", {budget}),')
    print("]")


def retrospective_check(
    cascade: list[tuple[str, int, dict]],
    model_data: dict[str, dict],
    n_total: int,
) -> list[float]:
    """Leave-one-out ETCT analysis. Positive = tier k saves time (beneficial)."""
    current_etct = sum(
        (s["n_uncovered"] / n_total) * s["avg_time_s"] for _, _, s in cascade
    )
    results = []
    for k in range(len(cascade)):
        etct_without = sum(
            (cascade[i][2]["n_uncovered"] / n_total) * cascade[i][2]["avg_time_s"]
            for i in range(k)
        )
        uncovered = set(cascade[k][2]["entry_uncovered"])
        for i in range(k + 1, len(cascade)):
            model, budget, _ = cascade[i]
            s = model_summary(model_data[model], budget, task_ids=uncovered)
            etct_without += (len(uncovered) / n_total) * s["avg_time_s"]
            uncovered -= s["passing_tasks"]
        if uncovered:
            etct_without += len(uncovered) / n_total * 1e6
        results.append(etct_without - current_etct)
    return results


def recompute_cascade_stats(
    models_budgets: list[tuple[str, int]],
    model_data: dict[str, dict],
    all_task_ids: set,
    model_load_times: dict | None = None,
    vram_per_model: dict | None = None,
) -> list[tuple[str, int, dict]]:
    """Recompute cascade stats for a fixed (model, budget) sequence."""
    cascade = []
    covered: set = set()
    uncovered = set(all_task_ids)
    cumulative_vram = 0
    for model, budget in models_budgets:
        s = model_summary(model_data[model], budget, task_ids=uncovered)
        n_uncov = len(uncovered)
        marginal = len(s["passing_tasks"])
        entry_unc = set(uncovered)
        covered |= s["passing_tasks"]
        uncovered -= s["passing_tasks"]
        model_vram = _vram_mb(model, vram_per_model or {})
        cumulative_vram += model_vram or 0
        cascade.append((model, budget, {
            "pass_rate": s["pass_rate"],
            "marginal_tasks": marginal,
            "cumulative_coverage": len(covered),
            "avg_time_s": s["avg_time_s"],
            "n_uncovered": n_uncov,
            "entry_uncovered": entry_unc,
            "load_time_s": (model_load_times or {}).get(model, 0.0),
            "vram_mb": model_vram,
            "cumulative_vram_mb": cumulative_vram if model_vram is not None else None,
        }))
    return cascade


def prune_harmful_tiers(
    cascade: list[tuple[str, int, dict]],
    model_data: dict[str, dict],
    all_task_ids: set,
    model_load_times: dict | None = None,
    vram_per_model: dict | None = None,
) -> list[tuple[str, int, dict]]:
    """Iteratively remove the most harmful tier until all tiers are net-beneficial."""
    while True:
        retro = retrospective_check(cascade, model_data, len(all_task_ids))
        harmful = [(i, r) for i, r in enumerate(retro) if r < 0]
        if not harmful:
            break
        worst_i, worst_r = min(harmful, key=lambda x: x[1])
        removed = cascade[worst_i][0]
        print(f"  Pruning tier {worst_i + 1} ({removed}, retro={worst_r:+.3f}s)")
        remaining = [(m, b) for i, (m, b, _) in enumerate(cascade) if i != worst_i]
        cascade = recompute_cascade_stats(
            remaining, model_data, all_task_ids, model_load_times, vram_per_model,
        )
    return cascade


def maximize_last_tier(
    cascade: list[tuple[str, int, dict]],
    model_data: dict[str, dict],
    all_task_ids: set,
    model_load_times: dict | None = None,
    vram_per_model: dict | None = None,
) -> list[tuple[str, int, dict]]:
    """Replace the last tier with the model that maximizes raw coverage on the tasks entering it.

    The last tier's task assignment (what enters it) is determined by the earlier tiers and
    is not changed. Only the model choice is upgraded from yield-optimised to coverage-maximised.
    """
    if not cascade:
        return cascade

    prior_models = {m for m, _, _ in cascade[:-1]}
    entry_uncov: set[str] = set(all_task_ids)
    for model, budget, _ in cascade[:-1]:
        s = model_summary(model_data[model], budget, task_ids=entry_uncov)
        entry_uncov -= s["passing_tasks"]

    if not entry_uncov:
        return cascade

    max_budget = max(
        len(t["iter_durations"])
        for tasks in model_data.values()
        for t in tasks.values()
    )

    last_model, last_budget, _ = cascade[-1]
    best_s = model_summary(model_data[last_model], last_budget, task_ids=entry_uncov)
    best_count = len(best_s["passing_tasks"])
    best_model, best_budget = last_model, last_budget

    for m in model_data:
        if m in prior_models:
            continue
        for b in range(1, max_budget + 1):
            s = model_summary(model_data[m], b, task_ids=entry_uncov)
            n = len(s["passing_tasks"])
            if n > best_count:
                best_count, best_model, best_budget = n, m, b

    if best_model == last_model and best_budget == last_budget:
        return cascade

    print(f"  Replacing last tier: {last_model} → {best_model} (budget={best_budget}, covers {best_count}/{len(entry_uncov)} entry tasks)")
    new_mb = [(m, b) for m, b, _ in cascade[:-1]] + [(best_model, best_budget)]
    return recompute_cascade_stats(new_mb, model_data, all_task_ids, model_load_times, vram_per_model)


def reorder_cascade_for_yield(
    cascade: list[tuple[str, int, dict]],
    model_data: dict[str, dict],
    tau: float,
    target_time: float,
    all_task_ids: set,
    excluded_models: dict[str, dict] | None = None,
    model_load_times: dict | None = None,
    vram_per_model: dict | None = None,
) -> list[tuple[str, int, dict]]:
    """Improve cascade yield by trying all tier permutations and inserting excluded models.

    Fast models excluded by the greedy (because they were tried after a slow model filled
    the budget) get a second chance here: inserting them before slower tiers often fits
    within the time budget and raises yield.

    Args:
        excluded_models: {model: tasks} for models not currently in the cascade.
            Each is tried at budget=1 inserted at position 0.
    """
    total_tasks = len(all_task_ids)
    best_yield = cascade_yield_score(cascade, tau, total_tasks)
    best = cascade

    def _etct(c: list) -> float:
        return sum((s["n_uncovered"] / total_tasks) * s["avg_time_s"] for _, _, s in c)

    def _try(mb: list[tuple[str, int]]) -> None:
        nonlocal best, best_yield
        c = recompute_cascade_stats(mb, model_data, all_task_ids, model_load_times, vram_per_model)
        if any(s["marginal_tasks"] == 0 for _, _, s in c):
            return  # a tier that solves nothing wastes time for tasks passing through it
        if _etct(c) > target_time:
            return
        y = cascade_yield_score(c, tau, total_tasks)
        if y > best_yield:
            best_yield = y
            best = c

    # Try all permutations of the current tiers.
    n = len(cascade)
    for perm in permutations(range(n)):
        if perm == tuple(range(n)):
            continue
        _try([(cascade[i][0], cascade[i][1]) for i in perm])

    # Try inserting each excluded model (at budget=1) at every position.
    if excluded_models:
        current_mb = [(m, b) for m, b, _ in best]
        for exc_model in excluded_models:
            for pos in range(len(current_mb) + 1):
                mb = list(current_mb)
                mb.insert(pos, (exc_model, 1))
                _try(mb)

    if best is not cascade:
        new_models = [m for m, _, _ in best]
        old_models = [m for m, _, _ in cascade]
        if new_models != old_models:
            print(f"  Reordered tiers for yield: {' → '.join(new_models)}")
        else:
            print(f"  Inserted model(s) for yield: {' → '.join(new_models)}")
        print(f"  Yield@τ={tau:.0f}s: {best_yield:.1f}%")

    return best
