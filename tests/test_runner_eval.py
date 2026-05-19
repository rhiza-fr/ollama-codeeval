"""Tests for runner.run_eval async function."""
import asyncio
from unittest.mock import MagicMock, patch

from ollama_codeeval.runner import run_eval


def _mock_client():
    """Return a context manager that stubs out client.call and client.stop."""
    return patch("ollama_codeeval.runner.client", MagicMock())


DATASET = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def foo():\n    pass\n",
        "entry_point": "foo",
        "test": "def check(candidate):\n    assert candidate() is None\n",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "def bar():\n    pass\n",
        "entry_point": "bar",
        "test": "def check(candidate):\n    assert candidate() is None\n",
    },
]

SUCCESS_LOG = {
    "task_id": "HumanEval/0",
    "iterations": [],
    "final_result": {"stdout": "", "stderr": "", "exit_code": 0},
}


def _run(coro):
    return asyncio.run(coro)


class TestRunEvalSingleWorker:
    def test_returns_results_for_each_model(self):
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            results = _run(run_eval(DATASET, models=["qwen3"], output_dir=None))
        assert "qwen3" in results
        assert "HumanEval/0" in results["qwen3"]
        assert "HumanEval/1" in results["qwen3"]

    def test_multiple_models(self):
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            results = _run(run_eval(DATASET, models=["model_a", "model_b"], output_dir=None))
        assert "model_a" in results
        assert "model_b" in results

    def test_writes_jsonl_when_output_dir_set(self, tmp_path):
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            with patch("ollama_codeeval.runner.append_jsonl") as mock_append:
                _run(run_eval(DATASET, models=["qwen3"], output_dir=str(tmp_path)))
        assert mock_append.call_count == len(DATASET)

    def test_no_jsonl_when_output_dir_none(self):
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            with patch("ollama_codeeval.runner.append_jsonl") as mock_append:
                _run(run_eval(DATASET, models=["qwen3"], output_dir=None))
        mock_append.assert_not_called()

    def test_think_slug_in_filename(self, tmp_path):
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            with patch("ollama_codeeval.runner.append_jsonl") as mock_append:
                _run(run_eval(DATASET, models=["qwen3"], think=True, output_dir=str(tmp_path)))
        for c in mock_append.call_args_list:
            assert "_think" in c[0][0]

    def test_dataset_name_in_filename(self, tmp_path):
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            with patch("ollama_codeeval.runner.append_jsonl") as mock_append:
                _run(
                    run_eval(
                        DATASET,
                        models=["qwen3"],
                        dataset_name="humaneval-rewritten-phi4",
                        output_dir=str(tmp_path),
                    )
                )
        for c in mock_append.call_args_list:
            assert "humaneval-rewritten-phi4" in c[0][0]

    def test_existing_output_file_deleted(self, tmp_path):
        existing = tmp_path / "results_humaneval_qwen3_nothink.jsonl"
        existing.write_text("old data\n")
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            with patch("ollama_codeeval.runner.append_jsonl"):
                _run(run_eval(DATASET, models=["qwen3"], output_dir=str(tmp_path)))
        assert existing.exists() is False


class TestRunEvalMultipleWorkers:
    def test_parallel_workers_return_all_results(self):
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            results = _run(run_eval(DATASET, models=["qwen3"], workers=2, output_dir=None))
        assert len(results["qwen3"]) == len(DATASET)

    def test_worker_exception_produces_error_result(self):
        def fail_first(row, model, think):
            if row["task_id"] == "HumanEval/0":
                raise RuntimeError("worker crashed")
            return SUCCESS_LOG

        with _mock_client(), patch("ollama_codeeval.runner.run_flow", side_effect=fail_first):
            results = _run(run_eval(DATASET, models=["qwen3"], workers=2, output_dir=None))
        failed = results["qwen3"].get("HumanEval/0")
        assert failed is not None
        assert failed["final_result"]["exit_code"] == 1
        assert "exception" in failed["final_result"]["stderr"].lower()

    def test_parallel_workers_write_jsonl(self, tmp_path):
        with _mock_client(), patch("ollama_codeeval.runner.run_flow", return_value=SUCCESS_LOG):
            with patch("ollama_codeeval.runner.append_jsonl") as mock_append:
                _run(run_eval(DATASET, models=["qwen3"], workers=2, output_dir=str(tmp_path)))
        assert mock_append.call_count == len(DATASET)
