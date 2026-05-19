"""## Module: Flow Control and Processing
This module defines a set of nodes and functions to process prompts through a series of steps, including generating, executing, fixing, and responding. It uses a typed dictionary for parameters and a node-based structure to manage the workflow. The module includes methods for preparing, executing, and posting results at each stage of the process."""

import hashlib
from typing import Any, TypedDict, cast

from diskcache import Cache
from ollama_think.client import ThinkResponse

from ollama_codeeval.apply_solution import apply_solution
from ollama_codeeval.autocontext import AutoContextClient
from ollama_codeeval.code_formatter import compact_ruff_error, ruff_fix
from ollama_codeeval.config import (
    EXECUTION_CACHE_DIR,
    MAX_ITERATIONS,
    MODEL_TEMPERATURE,
    OLLAMA_HOST,
    SANDBOX_IMAGE,
    SANDBOX_LANG,
    SANDBOX_TIMEOUT,
)
from ollama_codeeval.pocketflow import Flow, Node
from ollama_codeeval.prompts import _truncate_code, build_fix_prompt
from ollama_codeeval.pytest_wrapper import consume_test_result, generate_test
from ollama_codeeval.sandbox import Sandbox

client = AutoContextClient(host=OLLAMA_HOST)  # allows for growth of context window if response is cut off

cache = Cache(EXECUTION_CACHE_DIR)


def _fail_test_result(msg: str, exit_code: int = 1) -> dict:
    return {"stderr": msg, "stdout": "", "exit_code": exit_code}


def _escalate_temperature(options: dict, iteration_num: int) -> dict:
    options = dict(options)
    options["temperature"] = min(1.0, options.get("temperature", 0.3) + 0.15 * (iteration_num - 1))
    return options


def _detect_stuck(shared: dict, msg: str) -> bool:
    if (
        len(shared["iterations"]) > 1
        and shared["iterations"][-1]["message"]["content"]
        == shared["iterations"][-2]["message"]["content"]
    ):
        shared["iterations"][-1]["test_result"] = _fail_test_result(msg)
        return True
    return False

_sandbox: Sandbox | None = None


def _get_sandbox() -> Sandbox:
    global _sandbox
    if _sandbox is None:
        try:
            _sandbox = Sandbox(lang=SANDBOX_LANG, image=SANDBOX_IMAGE, execution_timeout=SANDBOX_TIMEOUT)
        except Exception as e:
            print(f"Fatal: Error creating Docker Sandbox - is docker running?\n  {e}")
            exit(1)
    return _sandbox


class CallParams(TypedDict):
    """Parameters for a call, including prompt, model, think flag, and options."""

    prompt: str
    model: str
    think: bool
    options: dict[str, float | int]


def generate_prompt(prompt: str, entrypoint: str) -> str:
    """Generate a prompt for completing a function with a specified entry point.

    Args:
        prompt: The initial prompt text.
        entrypoint: The name of the entry point for the function.

    Returns:
        A formatted prompt string for function completion."""
    prompt = f"Complete the following function.\n{prompt}\n"
    prompt += (
        f"Only output the function with the following entry_point: `{entrypoint}`\n"
    )
    prompt += "Make sure your output begins with 'def'. No explanations needed. Do not format as markdown (such as *```python ... ```*)."
    return prompt


class FormatOriginalQuestionNode(Node):
    """Prepares the original question node by generating a prompt based on the input."""

    def prep(self, shared: dict[str, Any]) -> None:
        """Prepares the original question node by generating a prompt based on the input.

        Generates a prompt using the provided input and entry point, storing it in the shared dictionary."""
        input = shared.get("input", {})
        if not input:
            return None
        shared["generated_prompt"] = generate_prompt(
            input["prompt"], input["entry_point"]
        )
        if input.get("original_input_prompt"):
            shared["rewritten_prompt"] = input["prompt"]



class GenerateNode(Node):
    """Generate node for processing tasks.

    Prepares parameters, executes logic, and generates a response."""

    def prep(self, shared: dict[str, Any]) -> CallParams:
        """Prepare parameters for the generate node based on shared context. This method extracts necessary parameters like prompt, model, think flag, and options from the shared dictionary to create a CallParams object."""
        return CallParams(
            {
                "prompt": shared.get("generated_prompt", {}),
                "model": shared.get("model", "qwen3"),
                "think": shared.get("think", False),
                "options": shared.get("options", {}),
            }
        )

    def exec(self, prep_res: CallParams) -> ThinkResponse | None:
        """Execute the generate logic using the prepared parameters. Calls the client with the model and prompt to generate a response. Returns None if the prompt is empty."""
        if len(prep_res["prompt"]) == 0:
            return None
        return client.call(
            prep_res["model"], prep_res["prompt"], think=prep_res["think"]
        )

    def post(
        self, shared: dict[str, Any], prep_res: Any, exec_res: ThinkResponse | None
    ) -> str:
        """Update shared context with generation results and return status."""
        shared["iterations"] = list()
        if exec_res is None:
            shared["iterations"].append({"test_result": _fail_test_result("LLM Initial generation failed")})
            return "fail"
        result = exec_res.to_dict()
        result["prompt"] = prep_res["prompt"]
        shared["iterations"].append(result)
        return "default"


class ExecuteNode(Node):
    """A node that executes a task with preparation, execution, and post-processing steps."""

    def prep(self, shared: dict[str, Any]) -> str:
        """Prepares the node by extracting the latest iteration, input, and content, then generates code and test using the solution. Returns the test result."""
        latest = shared.get("iterations", [])[-1]
        input = shared.get("input", {})
        if "ruff_fixed_code" in latest:
            code = latest["ruff_fixed_code"]
        else:
            content = latest["message"]["content"]
            code = apply_solution(
                prompt=input["prompt"],
                solution=content,
                entrypoint=input["entry_point"],
            )
        test = generate_test(
            code, shared["input"]["test"], entrypoint=shared["input"]["entry_point"]
        )
        shared["iterations"][-1]["test"] = test
        return test

    def exec(self, prep_res: str) -> Any | None:
        """Executes the task by running the test in a sandbox and caching the result for efficiency."""
        test = prep_res
        key = hashlib.md5(test.encode(), usedforsecurity=False).hexdigest()
        test_result = cache.get(key)
        if test_result is None:
            test_result = _get_sandbox().run(test)
            if test_result is not None:
                test_result = consume_test_result(test_result)
                cache.set(key, test_result)
        if test_result is None:
            return {"stdout": "", "stderr": "Execution Failure (None)", "exit_code": 1}
        return test_result

    def post(self, shared: dict[str, Any], prep_res: Any, exec_res: Any) -> str:
        """Updates the shared state with the execution result and determines the post-processing outcome."""
        shared["iterations"][-1]["test_result"] = exec_res
        if exec_res["exit_code"] == 0:
            return "ok"
        if len(shared["iterations"]) >= shared["max_iterations"]:
            return "fail"
        return "error"


class FixNode(Node):
    """Fixes a node in the execution flow. Handles preparation, execution, and post-processing steps."""

    def prep(self, shared: dict[str, Any]) -> CallParams:
        """Prepare the prompt for the model to regenerate a solution based on test errors.

        Args:
          shared: A dictionary containing shared state, including 'input', 'iterations', and 'model'.
        Returns:
          CallParams: A dictionary with the prepared prompt, model, think settings, and options."""
        input = shared.get("input", {})
        content = shared["iterations"][-1]["message"]["content"]
        code = apply_solution(
            prompt=input["prompt"], solution=content, entrypoint=input["entry_point"]
        )
        entry_point = input.get("entry_point", "")
        error = shared["iterations"][-1]["test_result"]["stderr"]
        error = error.replace("candidate", entry_point)
        prompt = build_fix_prompt(shared["generated_prompt"], code, error, shared["iterations"])
        options = _escalate_temperature(shared.get("options", {}), len(shared["iterations"]))
        return CallParams(
            {
                "prompt": prompt,
                "model": shared["model"],
                "think": shared["think"],
                "options": options,
            }
        )

    def exec(self, prep_res: CallParams) -> ThinkResponse:
        """Returns the response from the model after executing the prepared prompt."""
        return client.call(
            prep_res["model"],
            prep_res["prompt"],
            think=prep_res["think"],
            options=prep_res["options"],
        )

    def post(
        self, shared: dict[str, Any], prep_res: Any, exec_res: ThinkResponse
    ) -> str:
        """Handles post-processing after model execution, logging iterations and checking for stuck states."""
        d = exec_res.to_dict()
        d["prompt"] = prep_res["prompt"]
        shared["iterations"].append(d)
        if _detect_stuck(shared, "LLM repeated the same answer in fix"):
            return "stuck"
        return "default"


class FixHarderNode(Node):
    """Restarts generation from scratch, discarding accumulated failure context.
    Uses the original (possibly rewritten) prompt at elevated temperature."""

    def prep(self, shared: dict[str, Any]) -> CallParams:
        """Restart with the original prompt, discarding all prior failure context."""
        options = _escalate_temperature(shared.get("options", {}), len(shared["iterations"]))
        return CallParams(
            {
                "prompt": shared["generated_prompt"],
                "model": shared["model"],
                "think": shared["think"],
                "options": options,
            }
        )

    def exec(self, prep_res: CallParams) -> ThinkResponse:
        """Execute the model with the prepared prompt and think parameter to generate a response that addresses the test error thoroughly."""
        return client.call(
            prep_res["model"],
            prep_res["prompt"],
            think=prep_res["think"],
            options=prep_res["options"],
        )

    def post(
        self, shared: dict[str, Any], prep_res: Any, exec_res: ThinkResponse
    ) -> str:
        """Update shared state with execution results and check for repeated responses.
        If the last two iterations have the same message content, mark the last iteration
        with a failure status indicating the LLM repeated the same answer."""
        d = exec_res.to_dict()
        d["prompt"] = prep_res["prompt"]
        shared["iterations"].append(d)
        if _detect_stuck(shared, "LLM repeated the same answer in fix harder"):
            return "verystuck"
        return "default"


class RuffFixNode(Node):
    """Runs ruff format + check --fix on the generated code before execution."""

    def prep(self, shared: dict[str, Any]) -> str:
        """Extract code from the latest iteration via apply_solution."""
        latest = shared.get("iterations", [])[-1]
        input = shared.get("input", {})
        content = latest["message"]["content"]
        return apply_solution(
            prompt=input["prompt"],
            solution=content,
            entrypoint=input["entry_point"],
        )

    def exec(self, prep_res: str) -> tuple[str, int, str]:
        """Run ruff fix locally, caching results."""
        key = "ruff_" + hashlib.md5(prep_res.encode(), usedforsecurity=False).hexdigest()
        result = cache.get(key)
        if result is None:
            result = ruff_fix(prep_res)
            cache.set(key, result)
        result = cast(tuple[str,int,str], result)
        return result

    def post(
        self, shared: dict[str, Any], prep_res: str, exec_res: tuple[str, int, str]
    ) -> str:
        """Store fixed code and route based on ruff exit code."""
        fixed_code, exit_code, remaining_errors = exec_res
        shared["iterations"][-1]["ruff_fixed_code"] = fixed_code
        if exit_code == 0:
            return "clean"
        shared["iterations"][-1]["test_result"] = _fail_test_result(remaining_errors, exit_code)
        if len(shared["iterations"]) >= shared["max_iterations"]:
            return "fail"
        return "lint_error"


class LintFixNode(Node):
    """Asks the LLM to fix remaining ruff lint errors that can't be auto-fixed."""

    def prep(self, shared: dict[str, Any]) -> CallParams:
        """Build a prompt with the code and compacted ruff error."""
        latest = shared["iterations"][-1]
        code = latest.get("ruff_fixed_code", latest["message"]["content"])
        error = compact_ruff_error(latest.get("test_result", {}).get("stderr", ""))
        input = shared.get("input", {})
        entry_point = input.get("entry_point", "")
        prompt = (
            "You were originally asked the question:\n\n"
            + shared["generated_prompt"]
            + "\n\n"
            + "You responded (formatted):\n\n"
            + "```python\n"
            + _truncate_code(code)
            + "\n```\n\n"
            + "Ruff linting found the following error:\n\n"
            + error
            + "\n\n"
            + f"Please fix the lint error and regenerate the complete `{entry_point}` function. "
            + "Only output the function. No explanations needed."
        )
        return CallParams(
            {
                "prompt": prompt,
                "model": shared["model"],
                "think": shared["think"],
                "options": shared.get("options", {}),
            }
        )

    def exec(self, prep_res: CallParams) -> ThinkResponse:
        """Call the LLM to fix lint errors."""
        return client.call(
            prep_res["model"],
            prep_res["prompt"],
            think=prep_res["think"],
            options=prep_res["options"],
        )

    def post(
        self, shared: dict[str, Any], prep_res: Any, exec_res: ThinkResponse
    ) -> str:
        """Append new iteration and route back to RuffFixNode."""
        d = exec_res.to_dict()
        d["prompt"] = prep_res["prompt"]
        shared["iterations"].append(d)
        return "default"


class RespondNode(Node):
    """Responds by setting the final result in the shared dictionary based on the last test result."""

    def post(self, shared: dict[str, Any], prep_res: Any, exec_res: Any) -> None:
        """Sets the final result in the shared dictionary based on the last test result from the iterations."""
        shared["final_result"] = shared["iterations"][-1]["test_result"]


def create_flow():
    """Creates a workflow for processing a flow with formatting, generation, execution, and error handling."""
    format = FormatOriginalQuestionNode()
    generate = GenerateNode()
    rufffix = RuffFixNode()
    lintfix = LintFixNode()
    execute = ExecuteNode()
    fix = FixNode()
    fixharder = FixHarderNode()
    respond = RespondNode()
    format >> generate
    generate - "default" >> rufffix
    generate - "fail" >> respond
    rufffix - "clean" >> execute
    rufffix - "lint_error" >> lintfix
    rufffix - "fail" >> respond
    lintfix - "default" >> rufffix
    execute - "ok" >> respond
    execute - "error" >> fix
    execute - "fail" >> respond
    fix - "stuck" >> fixharder
    fix - "default" >> rufffix
    fixharder - "verystuck" >> respond
    fixharder - "default" >> rufffix
    myFlow = Flow(start=format)
    return myFlow


def run_flow(test_row: dict, model="qwen3", think=True):
    """Run a flow with the given test row and model settings.
    Configures the flow with shared parameters and executes it.
    Returns the shared state after execution."""
    myFlow = create_flow()
    options = {
        "temperature": 1.0 if model.startswith("gemma4") else MODEL_TEMPERATURE,  # REMOVE ME FOR FAIR TESTING
    }
    shared = {
        "input": test_row,
        "model": model,
        "think": think,
        "max_iterations": MAX_ITERATIONS,
        "options": options,
    }
    myFlow.run(shared=shared)
    return shared


def close_sandbox():
    """Close the sandbox environment if it was created."""
    global _sandbox
    if _sandbox is not None:
        _sandbox.close()
        _sandbox = None


if __name__ == "__main__":
    from rich import print

    test_row = {
        "task_id": "HumanEval/0",
        "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        "entry_point": "has_close_elements",
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
        "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n",
    }
    res = run_flow(test_row, model="qwen3", think=False)
    print(res)
