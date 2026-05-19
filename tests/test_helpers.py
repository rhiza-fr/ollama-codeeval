"""Tests for pure helper functions across runner, agent, pytest_wrapper, and code_formatter modules."""

from ollama_codeeval.runner import is_error, slugify
from ollama_codeeval.agent import generate_prompt
from ollama_codeeval.pytest_wrapper import (
    clean_test_output,
    consume_test_result,
    extract_traceback_details,
    generate_test,
)
from ollama_codeeval.code_formatter import compact_ruff_error


# --- slugify ---


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_characters(self):
        assert slugify("Hello, World!") == "hello-world"

    def test_multiple_spaces(self):
        assert slugify("hello   world") == "hello-world"

    def test_leading_trailing_hyphens(self):
        assert slugify("-hello-") == "hello"

    def test_unicode_stripped(self):
        assert slugify("café") == "cafe"

    def test_unicode_preserved(self):
        assert slugify("café", allow_unicode=True) == "café"

    def test_model_name_with_colon(self):
        assert slugify("qwen3:4b") == "qwen3_4b"

    def test_model_name_latest(self):
        assert slugify("gemma3:latest") == "gemma3_latest"

    def test_model_name_with_version(self):
        assert slugify("qwen2.5-coder:14b") == "qwen2p5-coder_14b"

    def test_empty_string(self):
        assert slugify("") == ""

    def test_slashes(self):
        assert slugify("org/model-name") == "orgmodel-name"


# --- is_error ---


class TestIsError:
    def test_empty_dict(self):
        assert is_error({}) is True

    def test_none_like(self):
        assert is_error(None) is True

    def test_passing_result(self):
        assert is_error({"stderr": "", "stdout": ""}) is False

    def test_stderr_present(self):
        assert is_error({"stderr": "error", "stdout": ""}) is True

    def test_stdout_present(self):
        # Current logic: any non-empty stdout is considered an error
        assert is_error({"stderr": "", "stdout": "output"}) is True

    def test_both_present(self):
        assert is_error({"stderr": "err", "stdout": "out"}) is True

    def test_missing_keys(self):
        # Missing keys default to "" via .get, so both empty = no error
        assert is_error({"exit_code": 0}) is False


# --- generate_prompt ---


class TestGeneratePrompt:
    def test_contains_entrypoint(self):
        result = generate_prompt("def foo():", "foo")
        assert "foo" in result

    def test_contains_original_prompt(self):
        result = generate_prompt("def bar(x):", "bar")
        assert "def bar(x):" in result

    def test_instructs_no_markdown(self):
        result = generate_prompt("def baz():", "baz")
        assert "markdown" in result.lower()

    def test_instructs_def_start(self):
        result = generate_prompt("def baz():", "baz")
        assert "def" in result


# --- generate_test ---


class TestGenerateTest:
    def test_wraps_code_and_test(self):
        code = "def add(a, b): return a + b"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3"
        result = generate_test(code, test, "add")
        assert "import pytest" in result
        assert "import sys" in result
        assert code in result
        assert test in result
        assert "check(add)" in result
        assert "pytest.main" in result

    def test_propagates_exit_code(self):
        # sys.exit() must wrap pytest.main() so the process exits non-zero on failure
        code = "def add(a, b): return a + b"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3"
        result = generate_test(code, test, "add")
        assert "sys.exit(pytest.main(" in result


# --- extract_traceback_details ---


class TestExtractTracebackDetails:
    def test_extracts_after_in_check(self):
        log = (
            "FAILED test.py::test\n"
            "_______ in check\n"
            ">       assert candidate(1) == 2\n"
            "E       AssertionError\n"
            "=========== 1 failed ==========="
        )
        result = extract_traceback_details(log)
        assert "assert candidate(1) == 2" in result
        assert "==========" not in result

    def test_empty_when_no_failure(self):
        assert extract_traceback_details("all good") == ""

    def test_strips_use_v_message(self):
        log = (
            "_______ in check\n"
            ">       assert x == y\n"
            "Use -v to get more diff\n"
            "=========== 1 failed ==========="
        )
        result = extract_traceback_details(log)
        assert "Use -v" not in result


# --- clean_test_output ---


class TestCleanTestOutput:
    def test_no_failures(self):
        assert clean_test_output("1 passed in 0.01s") == ""

    def test_with_failures(self):
        log = (
            "FAILURES\n"
            "_______ in check\n"
            ">       assert 1 == 2\n"
            "=========== 1 failed ==========="
        )
        result = clean_test_output(log)
        assert "assert 1 == 2" in result


# --- consume_test_result ---


class TestConsumeTestResult:
    def test_passes_stderr_through(self):
        result = consume_test_result({"stderr": "real error", "stdout": "", "exit_code": 1})
        assert result["stderr"] == "real error"
        assert result["exit_code"] == 1

    def test_passing_test_clears_output(self):
        result = consume_test_result({"stderr": "", "stdout": "1 passed in 0.01s", "exit_code": 0})
        assert result["stderr"] == ""
        assert result["exit_code"] == 0

    def test_failing_test_moves_to_stderr(self):
        stdout = (
            "FAILURES\n"
            "_______ in check\n"
            ">       assert 1 == 2\n"
            "=========== 1 failed ==========="
        )
        result = consume_test_result({"stderr": "", "stdout": stdout, "exit_code": 1})
        assert "assert 1 == 2" in str(result["stderr"])
        assert result["exit_code"] == 1

    def test_nonzero_exit_empty_stdout_is_failure(self):
        # Docker may drop pytest stdout (reentrancy); trust exit_code as fallback
        result = consume_test_result({"stderr": "", "stdout": "", "exit_code": 1})
        assert result["exit_code"] == 1
        assert result["stderr"] != ""


# --- compact_ruff_error ---


class TestCompactRuffError:
    def test_truncates_at_pointer(self):
        err = (
            "file.py:1:1: E999 SyntaxError\n"
            "  |\n"
            "1 | def foo(\n"
            "  |         ^^\n"
            "  |\n"
            "Found 1 error.\n"
        )
        result = compact_ruff_error(err)
        assert "SyntaxError" in result
        assert "Found 1 error" not in result

    def test_returns_original_when_no_pointer(self):
        err = "some other error format"
        assert compact_ruff_error(err) == err
