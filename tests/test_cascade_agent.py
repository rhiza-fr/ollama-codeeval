from unittest.mock import MagicMock, patch

from ollama_codeeval.cascade_agent import (
    CascadeExecuteNode,
    CascadeGenerateNode,
    EscalateNode,
    run_cascade_flow,
)


def make_shared(cascade):
    return {
        "input": {"task_id": "HumanEval/0", "prompt": "def foo():\n    pass\n", "entry_point": "foo"},
        "generated_prompt": "Complete the following function.\ndef foo():\n    pass\n",
        "cascade_remaining": list(cascade[1:]),
        "current_model": cascade[0][0],
        "tier_max_iters": cascade[0][1],
        "tier_start_iter": 0,
        "iterations": [],
        "think": False,
        "options": {},
        "model": cascade[0][0],
        "max_iterations": sum(m for _, m in cascade),
    }


def test_cascade_generate_sets_iterations():
    shared = make_shared([("qwen3:4b", 1), ("qwen3:14b", 5)])
    node = CascadeGenerateNode()
    mock_response = MagicMock()
    mock_response.to_dict.return_value = {
        "model": "qwen3:4b",
        "message": {"content": "def foo():\n    return 1\n"},
    }
    with patch("ollama_codeeval.cascade_agent.client") as mock_client:
        mock_client.call.return_value = mock_response
        prep = node.prep(shared)
        result = node.exec(prep)
        action = node.post(shared, prep, result)
    assert action == "default"
    assert len(shared["iterations"]) == 1
    assert shared["iterations"][0]["model"] == "qwen3:4b"


def test_cascade_generate_fail_on_none_response():
    shared = make_shared([("qwen3:4b", 1)])
    node = CascadeGenerateNode()
    with patch("ollama_codeeval.cascade_agent.client") as mock_client:
        mock_client.call.return_value = None
        prep = node.prep(shared)
        result = node.exec(prep)
        action = node.post(shared, prep, result)
    assert action == "fail"
    assert shared["iterations"][-1]["test_result"]["exit_code"] == 1


def test_escalate_pops_next_tier():
    shared = make_shared([("qwen3:4b", 1), ("qwen2.5-coder:latest", 1), ("qwen3:14b", 5)])
    # simulate tier1 exhausted — one iteration recorded
    shared["iterations"] = [{"model": "qwen3:4b", "message": {"content": "x"}}]
    node = EscalateNode()
    action = node.post(shared, None, None)
    assert action == "generate"
    assert shared["current_model"] == "qwen2.5-coder:latest"
    assert shared["model"] == "qwen2.5-coder:latest"
    assert shared["tier_max_iters"] == 1
    assert shared["tier_start_iter"] == 1  # one iteration already recorded


def test_escalate_fails_when_exhausted():
    shared = make_shared([("qwen3:4b", 1)])
    shared["cascade_remaining"] = []
    node = EscalateNode()
    action = node.post(shared, None, None)
    assert action == "fail"


def _exec_shared(n_iters, tier_start, tier_max, cascade_remaining):
    shared = make_shared([("qwen3:4b", 1)])
    shared["cascade_remaining"] = list(cascade_remaining)
    shared["tier_start_iter"] = tier_start
    shared["tier_max_iters"] = tier_max
    shared["iterations"] = [
        {"model": "qwen3:4b", "message": {"content": "x"}, "test": "t"}
        for _ in range(n_iters)
    ]
    return shared


def test_cascade_execute_ok():
    shared = _exec_shared(1, 0, 1, [("qwen3:14b", 5)])
    node = CascadeExecuteNode()
    action = node.post(shared, "test_code", {"exit_code": 0, "stdout": "", "stderr": ""})
    assert action == "ok"
    assert shared["iterations"][-1]["test_result"]["exit_code"] == 0


def test_cascade_execute_escalate_when_tier_exhausted():
    shared = _exec_shared(1, 0, 1, [("qwen3:14b", 5)])
    node = CascadeExecuteNode()
    action = node.post(shared, "test_code", {"exit_code": 1, "stdout": "", "stderr": "err"})
    assert action == "escalate"


def test_cascade_execute_error_within_tier():
    # tier3: 2 of 5 iters used, more budget remains, no more cascade
    shared = _exec_shared(2, 0, 5, [])
    node = CascadeExecuteNode()
    action = node.post(shared, "test_code", {"exit_code": 1, "stdout": "", "stderr": "err"})
    assert action == "error"


def test_cascade_execute_fail_when_all_exhausted():
    # tier3 budget fully used, no remaining cascade
    shared = _exec_shared(5, 0, 5, [])
    node = CascadeExecuteNode()
    action = node.post(shared, "test_code", {"exit_code": 1, "stdout": "", "stderr": "err"})
    assert action == "fail"


# --- integration test ---

TASK_ROW = {
    "task_id": "HumanEval/0",
    "prompt": 'def has_close_elements(numbers, threshold):\n    """ stub """\n',
    "entry_point": "has_close_elements",
    "canonical_solution": "    return False\n",
    "test": "def check(candidate):\n    assert candidate([1.0,2.0,3.0],0.5)==False\n",
}


def test_run_cascade_flow_returns_required_keys():
    """run_cascade_flow returns a dict with iterations, final_result, input."""
    cascade = [("qwen3:4b", 1), ("qwen3:14b", 5)]
    mock_resp = MagicMock()
    mock_resp.to_dict.return_value = {
        "model": "qwen3:4b",
        "message": {"content": "def has_close_elements(numbers, threshold):\n    return False\n"},
    }
    mock_test_result = {"exit_code": 0, "stdout": "1 passed", "stderr": ""}
    with patch("ollama_codeeval.cascade_agent.client") as mock_client, \
         patch("ollama_codeeval.agent._get_sandbox") as mock_sb:
        mock_client.call.return_value = mock_resp
        mock_sb.return_value.run.return_value = MagicMock(
            exit_code=0, stdout="1 passed", stderr=""
        )
        # patch consume_test_result too since ExecuteNode uses it
        with patch("ollama_codeeval.agent.consume_test_result", return_value=mock_test_result), \
             patch("ollama_codeeval.agent.cache") as mock_cache:
            mock_cache.get.return_value = mock_test_result
            result = run_cascade_flow(TASK_ROW, cascade=cascade)
    assert "iterations" in result
    assert "final_result" in result
    assert "input" in result
    assert result["input"]["task_id"] == "HumanEval/0"
