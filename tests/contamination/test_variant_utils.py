# tests/contamination/test_variant_utils.py
from unittest.mock import MagicMock

import pytest
from ollama_codeeval.contamination.contamination_report import pair_results
from ollama_codeeval.contamination.oracle import run_oracle
from ollama_codeeval.contamination.variant_utils import (
    build_check_function,
    contamination_score,
    extract_function_name,
    parse_input_list,
)


class TestParseInputList:
    def test_clean_list(self):
        result = parse_input_list("[(1, 2), (3, 4)]")
        assert result == [(1, 2), (3, 4)]

    def test_with_surrounding_text(self):
        result = parse_input_list("Here are some inputs:\n[(1, 2), (3,)]\n")
        assert result == [(1, 2), (3,)]

    def test_nested_list_args(self):
        result = parse_input_list("[([1.0, 2.0], 0.5), ([], 0.1)]")
        assert result == [([1.0, 2.0], 0.5), ([], 0.1)]

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_input_list("not a list at all")


class TestBuildCheckFunction:
    def test_basic(self):
        pairs = [(([1.0, 2.0, 3.9], 0.3), True), (([1.0, 5.0], 0.5), False)]
        result = build_check_function("my_func", pairs)
        assert "def check(candidate):" in result
        assert "candidate([1.0, 2.0, 3.9], 0.3)" in result
        assert "== True" in result
        assert "candidate([1.0, 5.0], 0.5)" in result
        assert "== False" in result

    def test_string_args_are_repr(self):
        pairs = [(("hello", "world"), "helloworld")]
        result = build_check_function("concat", pairs)
        assert "'hello'" in result or '"hello"' in result

    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError):
            build_check_function("f", [])


class TestExtractFunctionName:
    def test_simple(self):
        code = "def are_temps_close(a, b):\n    return abs(a - b) < 0.5\n"
        assert extract_function_name(code) == "are_temps_close"

    def test_with_preamble(self):
        code = "from typing import List\n\ndef sensor_overlap(xs: List[float], t: float) -> bool:\n    pass\n"
        assert extract_function_name(code) == "sensor_overlap"

    def test_no_def_returns_none(self):
        assert extract_function_name("x = 1") is None


class TestContaminationScore:
    def test_all_asymmetric(self):
        # pass orig, fail var = contamination signal
        pairs = [(True, False), (True, False)]
        assert contamination_score(pairs) == pytest.approx(1.0)

    def test_symmetric(self):
        pairs = [(True, True), (False, False)]
        assert contamination_score(pairs) == pytest.approx(0.0)

    def test_mixed(self):
        # 2 contamination signals, 1 reverse, 1 neutral → (2-1)/4 = 0.25
        pairs = [(True, False), (True, False), (False, True), (True, True)]
        assert contamination_score(pairs) == pytest.approx(0.25)

    def test_empty(self):
        assert contamination_score([]) == 0.0


# --- Oracle tests (no real sandbox needed) ---




class TestRunOracle:
    def test_successful_pairs(self):
        sandbox = MagicMock()
        # Simulate sandbox returning two results: one ok, one error
        sandbox.run.return_value = {
            "stdout": (
                "{'ok': True, 'args': (1, 2), 'result': 3}\n"
                "{'ok': False, 'args': (0, 0), 'result': None, 'error': 'ZeroDivisionError'}\n"
            ),
            "stderr": "",
            "exit_code": 0,
        }
        code = "def add(a, b):\n    return a + b\n"
        inputs = [(1, 2), (0, 0)]
        pairs = run_oracle(code, "add", inputs, sandbox)
        assert pairs == [((1, 2), 3)]  # only the ok one

    def test_sandbox_failure_returns_empty(self):
        sandbox = MagicMock()
        sandbox.run.return_value = {"stdout": "", "stderr": "crash", "exit_code": 1}
        pairs = run_oracle("def f(x): return x", "f", [(1,)], sandbox)
        assert pairs == []





class TestPairResults:
    def test_basic_pairing(self):
        originals = [
            {"input": {"task_id": "HumanEval/0"}, "final_result": {"exit_code": 0}},
            {"input": {"task_id": "HumanEval/1"}, "final_result": {"exit_code": 1}},
        ]
        variants = [
            {"input": {"task_id": "HumanEvalVariant/0", "original_task_id": "HumanEval/0"}, "final_result": {"exit_code": 1}},
            {"input": {"task_id": "HumanEvalVariant/1", "original_task_id": "HumanEval/1"}, "final_result": {"exit_code": 0}},
        ]
        pairs = pair_results(originals, variants)
        # HumanEval/0: pass orig (exit 0), fail var (exit 1) → (True, False)
        # HumanEval/1: fail orig (exit 1), pass var (exit 0) → (False, True)
        assert pairs == [("HumanEval/0", True, False), ("HumanEval/1", False, True)]

    def test_unmatched_variants_skipped(self):
        originals = [
            {"input": {"task_id": "HumanEval/0"}, "final_result": {"exit_code": 0}},
        ]
        variants = [
            {"input": {"task_id": "HumanEvalVariant/99", "original_task_id": "HumanEval/99"}, "final_result": {"exit_code": 0}},
        ]
        pairs = pair_results(originals, variants)
        assert pairs == []
