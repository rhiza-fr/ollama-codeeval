from ollama_codeeval.report.analyse_difficulty import (
    _label,
    _prompt_metrics,
    compute_difficulty_signals,
)


def test_empty_prompt():
    assert _prompt_metrics("") == (0, 0, 0)


def test_counts_examples():
    prompt = "def f(x):\n    '''\n    >>> f(1)\n    2\n    >>> f(2)\n    4\n    '''\n"
    chars, n_ex, ckw = _prompt_metrics(prompt)
    assert n_ex == 2


def test_counts_constraint_keywords():
    prompt = "Return only unique values. Must be sorted. Never return None."
    _, _, ckw = _prompt_metrics(prompt)
    assert ckw >= 4  # only, unique, sorted, never (must counted too if in set)


def test_counts_at_least_at_most():
    prompt = "Return at least 1 and at most 5 values."
    _, _, ckw = _prompt_metrics(prompt)
    assert ckw >= 2  # "at least", "at most"


def test_prompt_chars():
    prompt = "hello"
    chars, _, _ = _prompt_metrics(prompt)
    assert chars == 5





def test_label_description():
    # High rewrite sensitivity + high assertion rate → description problem
    assert _label(base_pass_rate=0.3, rewrite_sensitivity=0.5, iter1_assertion_rate=0.6) == "description?"


def test_label_algorithm():
    # Low pass rate, neutral rewrite, low assertion rate → algorithm problem
    assert _label(base_pass_rate=0.2, rewrite_sensitivity=0.05, iter1_assertion_rate=0.1) == "algorithm?"


def test_label_empty_when_ambiguous():
    assert _label(base_pass_rate=0.8, rewrite_sensitivity=0.0, iter1_assertion_rate=0.0) == ""


def test_label_description_boundary():
    # Exactly at threshold: rewrite_sensitivity=0.2, iter1_assertion_rate=0.3 → NOT description (needs strictly greater)
    assert _label(base_pass_rate=0.3, rewrite_sensitivity=0.2, iter1_assertion_rate=0.3) == ""


def test_label_description_above_boundary():
    assert _label(base_pass_rate=0.3, rewrite_sensitivity=0.21, iter1_assertion_rate=0.31) == "description?"





def _make_iter(exit_code=0, done_reason="stop", stderr=""):
    return {
        "done_reason": done_reason,
        "test_result": {"exit_code": exit_code, "stderr": stderr},
    }


def _make_task(task_id, prompt, exit_code=0, iter1_exit=None, iter1_stderr=""):
    iter1_ec = iter1_exit if iter1_exit is not None else exit_code
    return {
        "input": {"task_id": task_id, "prompt": prompt},
        "final_result": {"exit_code": exit_code},
        "iterations": [_make_iter(iter1_ec, stderr=iter1_stderr)],
    }


def _make_data(model, think, rewrite_model, tasks):
    dataset = f"humaneval-rewritten-{rewrite_model}" if rewrite_model else "humaneval"
    return {"model": model, "think": think, "dataset": dataset, "tasks": tasks}


def test_single_base_run_pass():
    data = [_make_data("m1", False, None, [_make_task("HumanEval/0", "def f(): pass", exit_code=0)])]
    signals = compute_difficulty_signals(data)
    assert "HumanEval/0" in signals
    s = signals["HumanEval/0"]
    assert s.base_pass_rate == 1.0
    assert s.rewrite_sensitivity == 0.0  # no rewrite pairs


def test_single_base_run_fail():
    data = [_make_data("m1", False, None, [_make_task("HumanEval/0", "def f(): pass", exit_code=1)])]
    signals = compute_difficulty_signals(data)
    assert signals["HumanEval/0"].base_pass_rate == 0.0


def test_rewrite_improves():
    """Base fails, rewrite passes → positive sensitivity."""
    base = _make_data("m1", False, None, [_make_task("HumanEval/0", "short prompt", exit_code=1)])
    rw = _make_data("m1", False, "rewriter", [_make_task("HumanEval/0", "short prompt", exit_code=0)])
    signals = compute_difficulty_signals([base, rw])
    assert signals["HumanEval/0"].rewrite_sensitivity == 1.0


def test_rewrite_regresses():
    """Base passes, rewrite fails → negative sensitivity."""
    base = _make_data("m1", False, None, [_make_task("HumanEval/0", "prompt", exit_code=0)])
    rw = _make_data("m1", False, "rewriter", [_make_task("HumanEval/0", "prompt", exit_code=1)])
    signals = compute_difficulty_signals([base, rw])
    assert signals["HumanEval/0"].rewrite_sensitivity == -1.0


def test_iter1_assertion_rate():
    """Iter 1 fails with AssertionError → iter1_assertion_rate = 1.0."""
    base = _make_data("m1", False, None, [
        _make_task("HumanEval/0", "prompt", exit_code=1,
                   iter1_exit=1, iter1_stderr="E   assert result == expected")
    ])
    signals = compute_difficulty_signals([base])
    assert signals["HumanEval/0"].iter1_assertion_rate == 1.0


def test_no_base_runs_for_task():
    """Task only appears in rewrite data (no matching base) → signals still present but zeroed."""
    rw = _make_data("m1", False, "rewriter", [_make_task("HumanEval/0", "prompt", exit_code=0)])
    signals = compute_difficulty_signals([rw])
    assert "HumanEval/0" in signals
    s = signals["HumanEval/0"]
    assert s.base_pass_rate == 0.0
    assert s.rewrite_sensitivity == 0.0


def test_prompt_metrics_populated():
    """Prompt metrics are extracted from the task input."""
    prompt = "def f(x):\n    '''\n    >>> f(1)\n    1\n    '''\n"
    base = _make_data("m1", False, None, [_make_task("HumanEval/0", prompt, exit_code=0)])
    signals = compute_difficulty_signals([base])
    assert signals["HumanEval/0"].n_examples == 1
    assert signals["HumanEval/0"].prompt_chars == len(prompt)
