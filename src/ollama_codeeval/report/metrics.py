import json
import logging
from pathlib import Path
from statistics import median

from rich.logging import RichHandler

from ollama_codeeval.report._shared import (
    _model_label,
    load_model_stats,
    normalize_dataset,
    resolve_model_name,
)

_model_stats: dict | None = None


def _get_model_stats() -> dict | None:
    global _model_stats
    if _model_stats is None:
        _model_stats = load_model_stats()
    return _model_stats


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

ERROR_TYPES = [
    "ValueError",
    "SyntaxError",
    "AssertionError",
    "TypeError",
    "NameError",
    "AttributeError",
    "IndexError",
    "KeyError",
]
ERROR_ALIASES = {
    # Ruff errors → pytest equivalents
    "Undefined name": "NameError",
    "invalid-syntax": "SyntaxError",
    "Ruff lint": "SyntaxError",
    "statement outside of a function": "SyntaxError",
    "IndentationError": "SyntaxError",
    "TabError": "SyntaxError",
    # Rare errors → OtherError
    "ZeroDivisionError": "OtherError",
    "RecursionError": "OtherError",
    "ModuleNotFoundError": "OtherError",
    "ImportError": "OtherError",
    "UnboundLocalError": "OtherError",
    "OverflowError": "OtherError",
    "StopIteration": "OtherError",
    "re.error": "OtherError",
    "EOFError": "OtherError",
    "Operation timed out": "OtherError",
    "Server Error": "OtherError",
    "referenced before assignment": "NameError",
}
_ERROR_COUNTS_TEMPLATE = {
    "Success": 0,
    "PredictLengthExceededError": 0,
    "RepetitionError": 0,
    "OtherError": 0,
}
for _e in ERROR_TYPES:
    _ERROR_COUNTS_TEMPLATE[_e] = 0


def _classify_error(done_reason, exit_code, stderr):
    """Returns the error category key for a single iteration."""
    if done_reason == "length":
        return "PredictLengthExceededError"
    if exit_code == 0:
        return "Success"
    if any(line.strip().startswith("E   assert") for line in stderr.splitlines()):
        return "AssertionError"
    if "LLM repeated the same answer" in stderr:
        return "RepetitionError"
    for alias_key, target in ERROR_ALIASES.items():
        if alias_key in stderr:
            return target
    for error in ERROR_TYPES:
        if error in stderr:
            return error
    return "OtherError"


def _compute_iteration_stats(tasks, num_iterations=6):
    """Returns (percent_pass_per_iteration, cum_pass_per_iteration)."""
    N = len(tasks)
    P_i = [0] * num_iterations
    C_i = [0] * num_iterations

    for task in tasks:
        already_passed = False
        for i in range(min(len(task["iterations"]), num_iterations)):
            iteration = task["iterations"][i]
            test_res = iteration.get("test_result")
            if not already_passed and test_res and test_res["exit_code"] == 0:
                P_i[i] += 1
                already_passed = True
            C_i[i] += 1

    percent_pass = [
        P_i[i] / C_i[i] * 100 if C_i[i] > 0 else 0 for i in range(num_iterations)
    ]
    cum_pass = [sum(P_i[: i + 1]) / N * 100 for i in range(num_iterations)]
    return percent_pass, cum_pass


def _parse_tag_from_filename(path: str) -> str | None:
    """Extract the tag suffix from a result filename.

    New format: results_{dataset}_{model}_{think}[_{tag}].jsonl
    Returns the tag portion after _think/_nothink, or None."""
    name = Path(path).stem
    if not name.startswith("results_"):
        return None
    rest = name[len("results_") :]
    for marker in ("_nothink", "_think"):
        idx = rest.rfind(marker)
        if idx == -1:
            continue
        after = rest[idx + len(marker) :]
        if after == "" or after.startswith("_"):
            return after.lstrip("_") or None
    return None


def process_jsonl_file(jsonl_file, no_cache=False):
    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            tasks.append(task)
    model = resolve_model_name(_model_label(tasks[0]), _get_model_stats())
    think = any(
        i.get("message", {}).get("thinking")
        for task in tasks
        for i in task.get("iterations", [])
    )
    dataset = normalize_dataset(tasks[0].get("dataset", "humaneval"))
    tag = _parse_tag_from_filename(jsonl_file)
    total_tests = len(tasks)
    passed_tests = sum((1 for task in tasks if task["final_result"]["exit_code"] == 0))
    failed_tests = total_tests - passed_tests
    total_time = 0

    error_counts = dict(_ERROR_COUNTS_TEMPLATE)
    other_errors_found = set()

    total_tokens = 0
    iteration_durations = []
    for task in tasks:
        for i in task.get("iterations", []):
            duration = i.get("total_duration", 0)
            if duration is not None and duration > 0:
                iteration_durations.append(duration / 1_000_000_000.0)
                total_time = total_time + duration
            prompt_tokens = i.get("prompt_eval_count", 0) or 0
            completion_tokens = i.get("eval_count", 0) or 0
            total_tokens += prompt_tokens + completion_tokens

            done_reason = i.get("done_reason")
            test_result = i.get("test_result", {})
            exit_code = test_result.get("exit_code")
            stderr = test_result.get("stderr", "")

            error_key = _classify_error(done_reason, exit_code, stderr)
            error_counts[error_key] += 1
            if error_key == "OtherError":
                other_errors_found.add(stderr)

    # if other_errors_found:
    #     log.warning(
    #         "[%s - Think: %s] Other Errors Found:\n%s",
    #         model,
    #         think,
    #         "\n".join(other_errors_found),
    #     )

    time_per_it = median(iteration_durations) if iteration_durations else 0
    total_time = total_time / 1000000000.0 if total_tests > 0 else 0
    average_time_per_test = total_time / total_tests if total_tests > 0 else 0
    success_per_1k_tokens = (
        passed_tests / (total_tokens / 1000) if total_tokens > 0 else 0
    )
    success_per_minute = passed_tests / (total_time / 60) if total_time > 0 else 0
    percent_pass_per_iteration, cum_pass_per_iteration = _compute_iteration_stats(tasks)

    summary_data = {
        "file": jsonl_file,
        "model": model,
        "think": think,
        "dataset": dataset,
        "tag": tag,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "total_time": total_time,
        "average_time_per_test": average_time_per_test,
        "average_time_per_iteration": time_per_it,
        "percent_pass_per_iteration": percent_pass_per_iteration,
        "cum_pass_per_iteration": cum_pass_per_iteration,
        "success_per_1k_tokens": success_per_1k_tokens,
        "success_per_minute": success_per_minute,
        "tasks": tasks,
        "error_counts": error_counts,
    }

    summary_data["html_file"] = Path(jsonl_file).with_suffix(".html").name
    return summary_data
