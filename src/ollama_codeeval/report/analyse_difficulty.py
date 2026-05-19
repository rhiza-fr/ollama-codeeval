"""Per-task difficulty signals: description sparsity + error fingerprint."""

import re
from dataclasses import dataclass

from ollama_codeeval.report.metrics import _classify_error

_CONSTRAINT_SINGLES = frozenset(
    [
        "only",
        "not",
        "unless",
        "exactly",
        "inclusive",
        "exclusive",
        "unique",
        "sorted",
        "must",
        "never",
        "always",
    ]
)


def _prompt_metrics(prompt: str) -> tuple[int, int, int]:
    """Return (prompt_chars, n_examples, constraint_keyword_count)."""
    chars = len(prompt)
    n_examples = prompt.count(">>>")
    text_lower = prompt.lower()
    multi = text_lower.count("at least") + text_lower.count("at most")
    words = re.findall(r"\b\w+\b", text_lower)
    singles = sum(1 for w in words if w in _CONSTRAINT_SINGLES)
    return chars, n_examples, singles + multi


def _label(
    base_pass_rate: float, rewrite_sensitivity: float, iter1_assertion_rate: float
) -> str:
    if rewrite_sensitivity > 0.2 and iter1_assertion_rate > 0.3:
        return "description?"
    if (
        base_pass_rate < 0.4
        and abs(rewrite_sensitivity) < 0.1
        and iter1_assertion_rate < 0.2
    ):
        return "algorithm?"
    return ""


@dataclass
class TaskDifficultySignals:
    task_id: str
    prompt_chars: int
    n_examples: int
    constraint_keywords: int
    base_pass_rate: float  # 0–1
    rewrite_sensitivity: float  # -1 to 1; 0.0 if no rewrite pairs
    iter1_assertion_rate: float  # 0–1
    label: str  # "description?" | "algorithm?" | ""


def compute_difficulty_signals(
    all_data: list[dict],
) -> dict[str, TaskDifficultySignals]:
    """Compute per-task difficulty signals from all_data."""
    base_items = [d for d in all_data if d.get("dataset", "humaneval") == "humaneval"]
    rewrite_items = [
        d for d in all_data if d.get("dataset", "humaneval") != "humaneval"
    ]

    base_by_key: dict[tuple, dict] = {}
    for d in base_items:
        base_by_key[(d["model"], d["think"])] = d

    all_task_ids: set[str] = set()
    for d in all_data:
        for task in d["tasks"]:
            all_task_ids.add(task["input"]["task_id"])

    prompt_by_task: dict[str, str] = {}
    base_pass: dict[str, list[bool]] = {tid: [] for tid in all_task_ids}
    base_iter1_assertion: dict[str, list[bool]] = {tid: [] for tid in all_task_ids}
    rw_improved: dict[str, int] = {tid: 0 for tid in all_task_ids}
    rw_regressed: dict[str, int] = {tid: 0 for tid in all_task_ids}
    rw_pairs: dict[str, int] = {tid: 0 for tid in all_task_ids}

    for d in base_items:
        for task in d["tasks"]:
            tid = task["input"]["task_id"]
            if tid not in prompt_by_task:
                prompt_by_task[tid] = task["input"].get("prompt", "")
            base_pass[tid].append(task["final_result"]["exit_code"] == 0)
            iterations = task.get("iterations", [])
            if iterations:
                it = iterations[0]
                tr = it.get("test_result") or {}
                err = _classify_error(
                    it.get("done_reason"), tr.get("exit_code"), tr.get("stderr", "")
                )
                base_iter1_assertion[tid].append(err == "AssertionError")
            else:
                base_iter1_assertion[tid].append(False)

    for rw_item in rewrite_items:
        base_item = base_by_key.get((rw_item["model"], rw_item["think"]))
        if base_item is None:
            continue
        base_task_by_id = {t["input"]["task_id"]: t for t in base_item["tasks"]}
        for task in rw_item["tasks"]:
            tid = task["input"]["task_id"]
            base_task = base_task_by_id.get(tid)
            if base_task is None:
                continue
            rw_pairs[tid] += 1
            base_passed = base_task["final_result"]["exit_code"] == 0
            rw_passed = task["final_result"]["exit_code"] == 0
            if not base_passed and rw_passed:
                rw_improved[tid] += 1
            elif base_passed and not rw_passed:
                rw_regressed[tid] += 1

    result: dict[str, TaskDifficultySignals] = {}
    for tid in all_task_ids:
        prompt = prompt_by_task.get(tid, "")
        chars, n_ex, ckw = _prompt_metrics(prompt)
        bp = base_pass[tid]
        bpr = sum(bp) / len(bp) if bp else 0.0
        ass = base_iter1_assertion[tid]
        iar = sum(ass) / len(ass) if ass else 0.0
        pairs = rw_pairs[tid]
        rws = (rw_improved[tid] - rw_regressed[tid]) / pairs if pairs > 0 else 0.0
        result[tid] = TaskDifficultySignals(
            task_id=tid,
            prompt_chars=chars,
            n_examples=n_ex,
            constraint_keywords=ckw,
            base_pass_rate=bpr,
            rewrite_sensitivity=rws,
            iter1_assertion_rate=iar,
            label=_label(bpr, rws, iar),
        )
    return result
