import pytest
from ollama_codeeval.report._shared import compute_yield
from ollama_codeeval.report.html_index import _compute_all_yield


def _make_task(exit_code, durations_ns):
    return {
        "final_result": {"exit_code": exit_code},
        "iterations": [{"total_duration": d} for d in durations_ns],
        "input": {"task_id": "HumanEval/0"},
    }


class TestComputeYield:
    def test_no_successes(self):
        tasks = [_make_task(1, [2_000_000_000])]
        assert compute_yield(tasks) == 0.0

    def test_single_success(self):
        tasks = [_make_task(0, [5_000_000_000])]
        assert compute_yield(tasks) > 0.0

    def test_two_tasks_one_success(self):
        tasks = [
            _make_task(0, [3_000_000_000]),
            _make_task(1, [7_000_000_000]),
        ]
        assert compute_yield(tasks) > 0.0

    def test_two_successes_sorted(self):
        tasks = [
            _make_task(0, [8_000_000_000]),
            _make_task(0, [3_000_000_000]),
        ]
        assert compute_yield(tasks) == pytest.approx(compute_yield([
            _make_task(0, [3_000_000_000]),
            _make_task(0, [8_000_000_000]),
        ]))


class TestComputeAllYield:
    def test_adds_yield_key(self):
        all_data = [
            {"tasks": [_make_task(0, [1_000_000_000])], "model": "a"},
            {"tasks": [_make_task(0, [10_000_000_000])], "model": "b"},
        ]
        _compute_all_yield(all_data)
        assert "yield" in all_data[0]
        assert "yield" in all_data[1]

    def test_faster_model_scores_higher(self):
        all_data = [
            {"tasks": [_make_task(0, [1_000_000_000])], "model": "fast"},
            {"tasks": [_make_task(0, [10_000_000_000])], "model": "slow"},
        ]
        _compute_all_yield(all_data)
        assert float(str(all_data[0]["yield"])) > float(str(all_data[1]["yield"]))

    def test_no_successes_scores_zero(self):
        all_data = [
            {"tasks": [_make_task(1, [5_000_000_000])], "model": "fail"},
        ]
        _compute_all_yield(all_data)
        assert all_data[0]["yield"] == 0.0
