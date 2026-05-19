"""Tests for agent.py node classes and flow functions."""

from unittest.mock import MagicMock, patch

import ollama_codeeval.agent as agent_module
from ollama_codeeval.agent import (
    ExecuteNode,
    FixHarderNode,
    FixNode,
    FormatOriginalQuestionNode,
    GenerateNode,
    LintFixNode,
    RespondNode,
    RuffFixNode,
    close_sandbox,
    create_flow,
    run_flow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROMPT = "def foo():\n    pass\n"
ENTRY_POINT = "foo"
TEST_CODE = "def check(candidate):\n    assert candidate() is None\n"


def make_shared(**overrides):
    base = {
        "input": {
            "task_id": "HumanEval/0",
            "prompt": PROMPT,
            "entry_point": ENTRY_POINT,
            "test": TEST_CODE,
        },
        "model": "test-model",
        "think": False,
        "options": {"temperature": 0.3},
        "max_iterations": 5,
        "generated_prompt": "Complete the following function.\n" + PROMPT,
    }
    base.update(overrides)
    return base


def make_iteration(content="def foo():\n    return None\n", test_result=None):
    entry = {"message": {"content": content, "role": "assistant"}}
    if test_result:
        entry["test_result"] = test_result
    return entry


def mock_think_response(content="def foo():\n    return None\n"):
    resp = MagicMock()
    data = {"message": {"content": content, "role": "assistant"}}
    resp.to_dict.return_value = data
    return resp


# ---------------------------------------------------------------------------
# FormatOriginalQuestionNode
# ---------------------------------------------------------------------------


class TestFormatOriginalQuestionNode:
    def test_prep_generates_prompt(self):
        node = FormatOriginalQuestionNode()
        shared = make_shared()
        node.prep(shared)
        assert "foo" in shared["generated_prompt"]
        assert PROMPT in shared["generated_prompt"]

    def test_prep_with_no_input_does_nothing(self):
        node = FormatOriginalQuestionNode()
        shared = {"input": {}}
        node.prep(shared)
        assert "generated_prompt" not in shared

    def test_prep_sets_rewritten_prompt_when_original_present(self):
        node = FormatOriginalQuestionNode()
        shared = make_shared()
        shared["input"]["original_input_prompt"] = "original text"
        shared["input"]["prompt"] = "rewritten text"
        node.prep(shared)
        assert shared["rewritten_prompt"] == "rewritten text"

    def test_prep_no_rewritten_prompt_when_original_absent(self):
        node = FormatOriginalQuestionNode()
        shared = make_shared()
        node.prep(shared)
        assert "rewritten_prompt" not in shared


# ---------------------------------------------------------------------------
# GenerateNode
# ---------------------------------------------------------------------------


class TestGenerateNode:
    def test_prep_builds_call_params(self):
        node = GenerateNode()
        shared = make_shared()
        params = node.prep(shared)
        assert params["model"] == "test-model"
        assert params["think"] is False
        assert "Complete" in params["prompt"]

    def test_exec_empty_prompt_returns_none(self):
        node = GenerateNode()
        result = node.exec({"prompt": "", "model": "m", "think": False, "options": {}})
        assert result is None

    def test_exec_calls_client(self):
        node = GenerateNode()
        mock_resp = mock_think_response()
        with patch.object(agent_module, "client") as mock_client:
            mock_client.call.return_value = mock_resp
            result = node.exec(
                {"prompt": "test prompt", "model": "m", "think": False, "options": {}}
            )
        assert result is mock_resp

    def test_post_none_exec_res_adds_fail_iteration(self):
        node = GenerateNode()
        shared = make_shared()
        result = node.post(shared, {"prompt": ""}, None)
        assert result == "fail"
        assert shared["iterations"][0]["test_result"]["exit_code"] == 1

    def test_post_success_stores_iteration(self):
        node = GenerateNode()
        shared = make_shared()
        mock_resp = mock_think_response("def foo(): return None")
        prep_res = {"prompt": "the prompt"}
        result = node.post(shared, prep_res, mock_resp)
        assert result == "default"
        assert len(shared["iterations"]) == 1
        assert shared["iterations"][0]["prompt"] == "the prompt"


# ---------------------------------------------------------------------------
# ExecuteNode
# ---------------------------------------------------------------------------


class TestExecuteNode:
    def test_prep_with_ruff_fixed_code(self):
        node = ExecuteNode()
        shared = make_shared()
        shared["iterations"] = [
            {
                "message": {"content": ""},
                "ruff_fixed_code": "def foo():\n    return None\n",
            }
        ]
        test = node.prep(shared)
        assert "import pytest" in test
        assert "foo" in test

    def test_prep_without_ruff_fixed_code(self):
        node = ExecuteNode()
        shared = make_shared()
        shared["iterations"] = [make_iteration("def foo():\n    return None\n")]
        test = node.prep(shared)
        assert "import pytest" in test
        assert "foo" in test

    def test_exec_cache_hit(self):
        node = ExecuteNode()
        cached = {"stdout": "", "stderr": "", "exit_code": 0}
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached
        with patch.object(agent_module, "cache", mock_cache):
            result = node.exec("some test code")
        assert result is cached
        mock_cache.set.assert_not_called()

    def test_exec_cache_miss_runs_sandbox(self):
        node = ExecuteNode()
        sandbox_result = {"stdout": "", "stderr": "", "exit_code": 0}
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_sandbox = MagicMock()
        mock_sandbox.run.return_value = {"stdout": "raw", "stderr": "", "exit_code": 0}
        with patch.object(agent_module, "cache", mock_cache):
            with patch("ollama_codeeval.agent._get_sandbox", return_value=mock_sandbox):
                with patch(
                    "ollama_codeeval.agent.consume_test_result",
                    return_value=sandbox_result,
                ):
                    result = node.exec("some test code")
        assert result is sandbox_result
        mock_cache.set.assert_called_once()

    def test_exec_sandbox_returns_none(self):
        node = ExecuteNode()
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_sandbox = MagicMock()
        mock_sandbox.run.return_value = None
        with patch.object(agent_module, "cache", mock_cache):
            with patch("ollama_codeeval.agent._get_sandbox", return_value=mock_sandbox):
                result = node.exec("some test code")
        assert result is not None
        assert result["exit_code"] == 1
        assert "Execution Failure" in result["stderr"]

    def test_post_ok(self):
        node = ExecuteNode()
        shared = make_shared()
        shared["iterations"] = [make_iteration()]
        exec_res = {"stdout": "", "stderr": "", "exit_code": 0}
        result = node.post(shared, "test", exec_res)
        assert result == "ok"

    def test_post_error(self):
        node = ExecuteNode()
        shared = make_shared()
        shared["iterations"] = [make_iteration()]
        exec_res = {"stdout": "", "stderr": "AssertionError", "exit_code": 1}
        result = node.post(shared, "test", exec_res)
        assert result == "error"

    def test_post_fail_when_max_iterations_reached(self):
        node = ExecuteNode()
        shared = make_shared(max_iterations=1)
        shared["iterations"] = [make_iteration()] * 1
        exec_res = {"stdout": "", "stderr": "error", "exit_code": 1}
        result = node.post(shared, "test", exec_res)
        assert result == "fail"


# ---------------------------------------------------------------------------
# FixNode
# ---------------------------------------------------------------------------


class TestFixNode:
    def _make_shared_with_error(self, num_iterations=1):
        shared = make_shared()
        iterations = [
            make_iteration(
                content="def foo():\n    return None\n",
                test_result={
                    "stderr": "AssertionError: candidate failed",
                    "stdout": "",
                    "exit_code": 1,
                },
            )
            for _ in range(num_iterations)
        ]
        shared["iterations"] = iterations
        return shared

    def test_prep_builds_prompt(self):
        node = FixNode()
        shared = self._make_shared_with_error()
        params = node.prep(shared)
        assert params["model"] == "test-model"
        assert "AssertionError" in params["prompt"] or "foo" in params["prompt"]

    def test_prep_escalates_temperature(self):
        node = FixNode()
        shared = self._make_shared_with_error(num_iterations=3)
        params = node.prep(shared)
        # temperature should be > initial 0.3
        assert params["options"]["temperature"] > 0.3

    def test_exec_calls_client(self):
        node = FixNode()
        mock_resp = mock_think_response()
        with patch.object(agent_module, "client") as mock_client:
            mock_client.call.return_value = mock_resp
            result = node.exec(
                {"model": "m", "prompt": "fix this", "think": False, "options": {}}
            )
        assert result is mock_resp

    def test_post_appends_iteration(self):
        node = FixNode()
        shared = self._make_shared_with_error()
        mock_resp = mock_think_response("def foo():\n    return 42\n")
        prep_res = {"prompt": "fix prompt"}
        result = node.post(shared, prep_res, mock_resp)
        assert result == "default"
        assert len(shared["iterations"]) == 2

    def test_post_detects_stuck(self):
        node = FixNode()
        shared = self._make_shared_with_error()
        same_content = "def foo():\n    return None\n"
        shared["iterations"].append(make_iteration(content=same_content))
        mock_resp = mock_think_response(same_content)
        result = node.post(shared, {"prompt": "p"}, mock_resp)
        assert result == "stuck"
        assert shared["iterations"][-1]["test_result"]["exit_code"] == 1


# ---------------------------------------------------------------------------
# FixHarderNode
# ---------------------------------------------------------------------------


class TestFixHarderNode:
    def _make_shared_with_error(self):
        shared = make_shared()
        shared["iterations"] = [
            make_iteration(
                content="def foo():\n    return None\n",
                test_result={"stderr": "AssertionError", "stdout": "", "exit_code": 1},
            )
        ]
        return shared

    def test_prep_builds_prompt(self):
        node = FixHarderNode()
        shared = self._make_shared_with_error()
        params = node.prep(shared)
        assert params["model"] == "test-model"
        assert "foo" in params["prompt"]

    def test_exec_calls_client(self):
        node = FixHarderNode()
        mock_resp = mock_think_response()
        with patch.object(agent_module, "client") as mock_client:
            mock_client.call.return_value = mock_resp
            result = node.exec(
                {"model": "m", "prompt": "fix harder", "think": False, "options": {}}
            )
        assert result is mock_resp

    def test_post_appends_iteration(self):
        node = FixHarderNode()
        shared = self._make_shared_with_error()
        mock_resp = mock_think_response("def foo():\n    return 99\n")
        result = node.post(shared, {"prompt": "p"}, mock_resp)
        assert result == "default"
        assert len(shared["iterations"]) == 2

    def test_post_detects_very_stuck(self):
        node = FixHarderNode()
        shared = self._make_shared_with_error()
        same_content = "def foo():\n    return None\n"
        shared["iterations"].append(make_iteration(content=same_content))
        mock_resp = mock_think_response(same_content)
        result = node.post(shared, {"prompt": "p"}, mock_resp)
        assert result == "verystuck"
        assert "repeated" in shared["iterations"][-1]["test_result"]["stderr"]


# ---------------------------------------------------------------------------
# RuffFixNode
# ---------------------------------------------------------------------------


class TestRuffFixNode:
    def test_prep_extracts_code(self):
        node = RuffFixNode()
        shared = make_shared()
        shared["iterations"] = [make_iteration("def foo():\n    return None\n")]
        code = node.prep(shared)
        assert "def foo" in code

    def test_exec_uses_cache(self):
        node = RuffFixNode()
        cached = ("def foo():\n    return None\n", 0, "")
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached
        with patch.object(agent_module, "cache", mock_cache):
            result = node.exec("def foo():\n    return None\n")
        assert result is cached

    def test_exec_calls_ruff_fix_on_cache_miss(self):
        node = RuffFixNode()
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        ruff_result = ("def foo():\n    return None\n", 0, "")
        with patch.object(agent_module, "cache", mock_cache):
            with patch("ollama_codeeval.agent.ruff_fix", return_value=ruff_result):
                result = node.exec("def foo():\n    return None\n")
        assert result == ruff_result
        mock_cache.set.assert_called_once()

    def test_post_clean_returns_clean(self):
        node = RuffFixNode()
        shared = make_shared()
        shared["iterations"] = [make_iteration()]
        result = node.post(shared, "code", ("def foo(): pass\n", 0, ""))
        assert result == "clean"
        assert shared["iterations"][-1]["ruff_fixed_code"] == "def foo(): pass\n"

    def test_post_lint_error_returns_lint_error(self):
        node = RuffFixNode()
        shared = make_shared()
        shared["iterations"] = [make_iteration()]
        result = node.post(shared, "code", ("def foo(): pass\n", 1, "E001 error"))
        assert result == "lint_error"
        assert shared["iterations"][-1]["test_result"]["exit_code"] == 1

    def test_post_fail_when_max_iterations(self):
        node = RuffFixNode()
        shared = make_shared(max_iterations=1)
        shared["iterations"] = [make_iteration()]
        result = node.post(shared, "code", ("code\n", 1, "error"))
        assert result == "fail"


# ---------------------------------------------------------------------------
# LintFixNode
# ---------------------------------------------------------------------------


class TestLintFixNode:
    def test_prep_builds_prompt(self):
        node = LintFixNode()
        shared = make_shared()
        shared["iterations"] = [
            {
                "message": {"content": "def foo(): pass"},
                "ruff_fixed_code": "def foo(): pass",
                "test_result": {
                    "stderr": "E001 SyntaxError\n  | code\n  |    ^",
                    "stdout": "",
                    "exit_code": 1,
                },
            }
        ]
        params = node.prep(shared)
        assert "foo" in params["prompt"]
        assert params["model"] == "test-model"

    def test_exec_calls_client(self):
        node = LintFixNode()
        mock_resp = mock_think_response()
        with patch.object(agent_module, "client") as mock_client:
            mock_client.call.return_value = mock_resp
            result = node.exec(
                {"model": "m", "prompt": "fix lint", "think": False, "options": {}}
            )
        assert result is mock_resp

    def test_post_appends_iteration(self):
        node = LintFixNode()
        shared = make_shared()
        shared["iterations"] = [make_iteration()]
        mock_resp = mock_think_response("def foo(): pass\n")
        result = node.post(shared, {"prompt": "p"}, mock_resp)
        assert result == "default"
        assert len(shared["iterations"]) == 2


# ---------------------------------------------------------------------------
# RespondNode
# ---------------------------------------------------------------------------


class TestRespondNode:
    def test_post_sets_final_result(self):
        node = RespondNode()
        shared = make_shared()
        test_result = {"stdout": "", "stderr": "", "exit_code": 0}
        shared["iterations"] = [make_iteration(test_result=test_result)]
        node.post(shared, None, None)
        assert shared["final_result"] is test_result


# ---------------------------------------------------------------------------
# create_flow / run_flow / close_sandbox
# ---------------------------------------------------------------------------


class TestCreateFlow:
    def test_returns_a_flow(self):
        from ollama_codeeval.pocketflow import Flow

        flow = create_flow()
        assert isinstance(flow, Flow)


class TestRunFlow:
    def test_run_flow_returns_shared(self):
        test_row = {
            "task_id": "HumanEval/0",
            "prompt": PROMPT,
            "entry_point": ENTRY_POINT,
            "test": TEST_CODE,
        }
        mock_flow = MagicMock()
        with patch("ollama_codeeval.agent.create_flow", return_value=mock_flow):
            result = run_flow(test_row, model="test-model", think=False)
        mock_flow.run.assert_called_once()
        assert result["model"] == "test-model"
        assert result["input"] is test_row


class TestCloseSandbox:
    def test_close_sandbox_when_none(self):
        agent_module._sandbox = None
        close_sandbox()  # should not raise
        assert agent_module._sandbox is None

    def test_close_sandbox_closes_and_clears(self):
        mock_sandbox = MagicMock()
        agent_module._sandbox = mock_sandbox
        close_sandbox()
        mock_sandbox.close.assert_called_once()
        assert agent_module._sandbox is None


class TestGetSandbox:
    def test_creates_sandbox_when_none(self):
        agent_module._sandbox = None
        mock_sb = MagicMock()
        with patch("ollama_codeeval.agent.Sandbox", return_value=mock_sb):
            result = agent_module._get_sandbox()
        assert result is mock_sb
        assert agent_module._sandbox is mock_sb
        agent_module._sandbox = None  # cleanup

    def test_reuses_existing_sandbox(self):
        mock_sb = MagicMock()
        agent_module._sandbox = mock_sb
        result = agent_module._get_sandbox()
        assert result is mock_sb
        agent_module._sandbox = None  # cleanup
