"""Tests for html_context._extract_context_stats."""

import pytest
from ollama_codeeval.report.html_context import (
    _build_summary_table,
    _chart_explosion_rate,
    _chart_token_usage,
    _extract_context_stats,
)


def _make_task(task_id, iterations, exit_code=0):
    """Build a minimal task record matching the all_data["tasks"][*] shape."""
    return {
        "input": {"task_id": task_id},
        "final_result": {"exit_code": exit_code},
        "iterations": iterations,
    }


def _make_iter(eval_count, prompt_eval_count, done_reason="stop", exit_code=0):
    return {
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "done_reason": done_reason,
        "test_result": {"exit_code": exit_code},
    }


# ── basic counting ────────────────────────────────────────────────────────────


def test_no_explosions():
    data = {
        "tasks": [
            _make_task("HumanEval/0", [_make_iter(200, 100, "stop")]),
        ]
    }
    stats = _extract_context_stats(data)
    assert stats["explosions"] == 0
    assert stats["total_iters"] == 1
    assert stats["expl_pct"] == 0.0
    assert stats["tasks_exploded"] == 0
    assert stats["tasks_stuck"] == 0


def test_one_explosion_passing_task():
    """Task has one exploded iteration but still passes — not 'stuck'."""
    data = {
        "tasks": [
            _make_task(
                "HumanEval/1",
                [
                    _make_iter(16384, 382, "length", exit_code=1),
                    _make_iter(200, 400, "stop", exit_code=0),
                ],
                exit_code=0,
            )
        ]
    }
    stats = _extract_context_stats(data)
    assert stats["explosions"] == 1
    assert stats["total_iters"] == 2
    assert stats["expl_pct"] == pytest.approx(50.0)
    assert stats["tasks_exploded"] == 1
    assert stats["tasks_stuck"] == 0  # task passed overall


def test_one_explosion_failing_task():
    """Task explodes and fails — counts as stuck."""
    data = {
        "tasks": [
            _make_task(
                "HumanEval/2",
                [_make_iter(16384, 382, "length", exit_code=1)],
                exit_code=1,
            )
        ]
    }
    stats = _extract_context_stats(data)
    assert stats["explosions"] == 1
    assert stats["tasks_stuck"] == 1


# ── token totals ─────────────────────────────────────────────────────────────


def test_token_totals():
    data = {
        "tasks": [
            _make_task(
                "HumanEval/0",
                [
                    _make_iter(100, 50),  # total 150
                    _make_iter(200, 100),  # total 300
                ],
            ),
        ]
    }
    stats = _extract_context_stats(data)
    assert stats["max_tokens"] == 300
    assert stats["avg_tokens"] == pytest.approx(225.0)


# ── task_details ──────────────────────────────────────────────────────────────


def test_task_details_only_for_exploded():
    """task_details contains only tasks that had ≥1 explosion."""
    data = {
        "tasks": [
            _make_task("HumanEval/0", [_make_iter(200, 100, "stop")]),
            _make_task(
                "HumanEval/1", [_make_iter(16384, 382, "length", 1)], exit_code=1
            ),
        ]
    }
    stats = _extract_context_stats(data)
    assert len(stats["task_details"]) == 1
    assert stats["task_details"][0]["task_id"] == "HumanEval/1"
    assert stats["task_details"][0]["passed"] is False


def test_task_detail_iteration_sequence():
    """task_details preserves iteration order with tokens and done_reason."""
    iters = [
        _make_iter(4096, 200, "length", 1),
        _make_iter(6144, 200, "length", 1),
        _make_iter(300, 200, "stop", 0),
    ]
    data = {"tasks": [_make_task("HumanEval/5", iters, exit_code=0)]}
    stats = _extract_context_stats(data)
    detail = stats["task_details"][0]
    assert len(detail["iter_rows"]) == 3
    assert detail["iter_rows"][0] == {
        "tokens": 4296,
        "done_reason": "length",
        "exit_code": 1,
    }
    assert detail["iter_rows"][1] == {
        "tokens": 6344,
        "done_reason": "length",
        "exit_code": 1,
    }
    assert detail["iter_rows"][2] == {
        "tokens": 500,
        "done_reason": "stop",
        "exit_code": 0,
    }


# ── edge cases ────────────────────────────────────────────────────────────────


def test_none_eval_counts_treated_as_zero():
    """None eval_count/prompt_eval_count should not crash."""
    data = {
        "tasks": [
            _make_task(
                "HumanEval/0",
                [
                    {
                        "eval_count": None,
                        "prompt_eval_count": None,
                        "done_reason": "stop",
                        "test_result": {"exit_code": 0},
                    }
                ],
            )
        ]
    }
    stats = _extract_context_stats(data)
    assert stats["max_tokens"] == 0


def test_empty_tasks():
    stats = _extract_context_stats({"tasks": []})
    assert stats["explosions"] == 0
    assert stats["total_iters"] == 0
    assert stats["expl_pct"] == 0.0
    assert stats["max_tokens"] == 0
    assert stats["avg_tokens"] == 0.0


def _make_file_entry(
    model, think, tag, expl, total_iters, max_tok, avg_tok, exploded, stuck
):
    """Minimal all_data entry shape for _build_summary_table."""
    return {
        "model": model,
        "think": think,
        "rewrite_model": None,
        "tag": tag,
        "_ctx": {
            "explosions": expl,
            "total_iters": total_iters,
            "expl_pct": expl / total_iters * 100 if total_iters else 0,
            "max_tokens": max_tok,
            "avg_tokens": avg_tok,
            "tasks_exploded": exploded,
            "tasks_stuck": stuck,
        },
    }


def test_summary_table_contains_model_name():
    entries = [
        _make_file_entry("deepseek-r1:1.5b", False, None, 50, 100, 16766, 8000, 30, 28)
    ]
    html = _build_summary_table(entries)
    assert "deepseek-r1:1.5b" in html
    assert "50.0%" in html  # expl_pct


def test_summary_table_zero_explosions_row():
    entries = [_make_file_entry("qwen3:latest", False, None, 0, 164, 500, 300, 0, 0)]
    html = _build_summary_table(entries)
    assert "qwen3:latest" in html
    assert "0.0%" in html


def test_summary_table_sort_order():
    """Higher expl_pct model should appear before lower."""
    entries = [
        _make_file_entry("low-expl", False, None, 5, 100, 500, 300, 3, 2),  # 5%
        _make_file_entry("high-expl", False, None, 80, 100, 16384, 8000, 50, 45),  # 80%
    ]
    html = _build_summary_table(entries)
    assert html.index("high-expl") < html.index("low-expl")


def test_summary_table_html_escaping():
    """Model names with special chars must be HTML-escaped."""
    entries = [_make_file_entry('model<>&"', False, None, 0, 10, 0, 0, 0, 0)]
    html = _build_summary_table(entries)
    assert "<script>" not in html
    assert "model&lt;&gt;&amp;" in html


def _entries_with_ctx(n=3):
    return [
        {
            "model": f"model-{i}",
            "think": False,
            "rewrite_model": None,
            "tag": None,
            "_ctx": {
                "expl_pct": i * 10.0,
                "explosions": i * 5,
                "total_iters": 50,
                "token_totals": [1000 + i * 500] * 10,
            },
        }
        for i in range(n)
    ]


def test_chart_explosion_rate_returns_html():
    html = _chart_explosion_rate(_entries_with_ctx())
    assert "Plotly.newPlot" in html
    assert "chartExplosionRate" in html


def test_chart_token_usage_returns_html():
    html = _chart_token_usage(_entries_with_ctx())
    assert "Plotly.newPlot" in html
    assert "chartTokenUsage" in html


def test_chart_explosion_rate_empty():
    """No entries → empty string (no div rendered)."""
    assert _chart_explosion_rate([]) == ""


def test_chart_token_usage_empty():
    assert _chart_token_usage([]) == ""


def test_chart_token_usage_no_token_data():
    """All entries have no token data → empty string."""
    entries = [
        {
            "model": "m",
            "think": False,
            "rewrite_model": None,
            "tag": None,
            "_ctx": {"expl_pct": 10.0, "token_totals": []},
        },
    ]
    assert _chart_token_usage(entries) == ""
