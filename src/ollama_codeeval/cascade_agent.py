"""Cascade flow: routes tasks through fast models first, escalates failures."""

from typing import Any

from ollama_think.client import ThinkResponse

from ollama_codeeval.agent import (
    ExecuteNode as _BaseExecuteNode,
)
from ollama_codeeval.agent import (
    FixNode,
    FormatOriginalQuestionNode,
    RespondNode,
    RuffFixNode,
    _fail_test_result,
    client,
)
from ollama_codeeval.config import MODEL_TEMPERATURE
from ollama_codeeval.pocketflow import Flow, Node


class CascadeGenerateNode(Node):
    """Generate a solution using shared['current_model']."""

    def prep(self, shared: dict[str, Any]) -> dict:
        return {
            "prompt": shared["generated_prompt"],
            "model": shared["current_model"],
            "think": shared.get("think", False),
            "options": shared.get("options", {}),
        }

    def exec(self, prep_res: dict) -> ThinkResponse | None:
        if not prep_res["prompt"]:
            return None
        return client.call(
            prep_res["model"],
            prep_res["prompt"],
            think=prep_res["think"],
            options=prep_res["options"],
        )

    def post(
        self, shared: dict[str, Any], prep_res: Any, exec_res: ThinkResponse | None
    ) -> str:
        if exec_res is None:
            shared["iterations"].append(
                {"test_result": _fail_test_result("LLM generation failed")}
            )
            return "fail"
        result = exec_res.to_dict()
        result["prompt"] = prep_res["prompt"]
        shared["iterations"].append(result)
        return "default"


class CascadeExecuteNode(_BaseExecuteNode):
    """ExecuteNode with tier-aware routing: escalate | error | fail instead of error | fail."""

    def post(self, shared: dict[str, Any], prep_res: Any, exec_res: Any) -> str:
        shared["iterations"][-1]["test_result"] = exec_res
        if exec_res["exit_code"] == 0:
            return "ok"
        tier_iters = len(shared["iterations"]) - shared["tier_start_iter"]
        if tier_iters >= shared["tier_max_iters"]:
            return "escalate" if shared.get("cascade_remaining") else "fail"
        return "error"


class EscalateNode(Node):
    """Pop next tier from cascade_remaining, update current_model and budget."""

    def post(self, shared: dict[str, Any], prep_res: Any, exec_res: Any) -> str:
        remaining = shared.get("cascade_remaining", [])
        if not remaining:
            return "fail"
        model, max_iters = remaining.pop(0)
        shared["current_model"] = model
        shared["model"] = model  # keep FixNode in sync
        shared["tier_max_iters"] = max_iters
        shared["tier_start_iter"] = len(shared["iterations"])
        return "generate"


def create_cascade_flow() -> Flow:
    """Wire the cascade flow."""
    fmt = FormatOriginalQuestionNode()
    generate = CascadeGenerateNode()
    rufffix = RuffFixNode()
    execute = CascadeExecuteNode()
    escalate = EscalateNode()
    fix = FixNode()
    respond = RespondNode()

    fmt - "default" >> generate
    generate - "default" >> rufffix
    generate - "fail" >> respond
    rufffix - "clean" >> execute
    rufffix - "lint_error" >> generate
    rufffix - "fail" >> respond
    execute - "ok" >> respond
    execute - "escalate" >> escalate
    execute - "error" >> fix
    execute - "fail" >> respond
    escalate - "generate" >> generate
    escalate - "fail" >> respond
    fix - "default" >> rufffix
    fix - "stuck" >> escalate

    return Flow(start=fmt)


# DEFAULT_CASCADE: list[tuple[str, int]] = [
#     ("qwen3:4b",             1),
#     ("qwen2.5-coder:latest", 1),
#     ("qwen3:14b",            5),
# ]

# DEFAULT_CASCADE: list[tuple[str, int]] = [
#     ("qwen3:4b", 1),
#     ("qwen2.5-coder:1.5b", 1),
#     ("granite4.1:8b", 1),
#     ("qwen3:latest", 1),
#     ("allenporter/xlam:7b", 1),
# ]


DEFAULT_CASCADE = [
    ("qwen3:4b", 1),
    ("qwen2.5-coder:latest", 1),
    ("gemma4:26b", 2),
]


def run_cascade_flow(
    test_row: dict,
    cascade: list[tuple[str, int]] = DEFAULT_CASCADE,
    think: bool = False,
) -> dict:
    """Run the cascade flow on a single task row."""
    flow = create_cascade_flow()
    first_model, first_max = cascade[0]
    shared: dict[str, Any] = {
        "input": test_row,
        "generated_prompt": "",
        "current_model": first_model,
        "model": first_model,
        "tier_max_iters": first_max,
        "tier_start_iter": 0,
        "cascade_remaining": list(cascade[1:]),
        "think": think,
        "iterations": [],
        "options": {
            "temperature": MODEL_TEMPERATURE,
        },
        "max_iterations": sum(m for _, m in cascade),
    }
    flow.run(shared=shared)
    return shared
