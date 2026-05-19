"""This module provides functions for processing prompts, running evaluations, and determining if results indicate errors. It includes utility functions for slugification, prompt generation, evaluation execution, error checking, and a main entry point."""

import logging
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ollama_codeeval.agent import client, run_flow
from ollama_codeeval.cascade_agent import run_cascade_flow
from ollama_codeeval.config import OUTPUT_BASE
from ollama_codeeval.jsonl_io import append_jsonl


def slugify(value, allow_unicode=False):
    """Normalize and sanitize a string to create a URL-friendly slug.

    Args:
        value: The input string to be slugified.
        allow_unicode: If True, Unicode characters are preserved during normalization.

    Returns:
        A string formatted as a slug, with non-alphanumeric characters replaced by hyphens
        and multiple spaces or hyphens reduced to a single hyphen."""
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = value.lower()
    value = value.replace(":", "_")  # preserve Ollama tag separator as _
    value = re.sub("[^\\w\\s.-]", "", value)
    value = value.replace(".", "p")
    return re.sub("[-\\s]+", "-", value).strip("-_")


async def run_eval(
    dataset: list[dict[str, str]],
    models: list[str] = ["qwen3"],
    think=False,
    output_dir: str | None = str(OUTPUT_BASE),
    workers: int = 1,
    dataset_name: str = "humaneval",
    tag: str | None = None,
    cascade: list[tuple[str, int]] | None = None,
):
    """Run evaluation on a dataset using specified models. Executes each model on each task in the dataset,
    logs the execution results, and saves them to JSONL files if an output directory is provided.
    Use workers > 1 to run multiple problems concurrently (useful for small models)."""
    results = dict()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        model_task = progress.add_task("Models", total=len(models))
        problem_task = progress.add_task("Problems", total=len(dataset))
        for model in models:
            progress.update(model_task, description=f"Model: {model}")
            progress.reset(problem_task, total=len(dataset))
            progress.update(
                problem_task,
                description=f"  {model} (workers={workers})",
            )
            output_file_for_model = None
            if output_dir:
                model_slug = "cascade" if cascade is not None else slugify(model)
                think_slug = "_think" if think else "_nothink"
                tag_slug = f"_{slugify(tag)}" if tag else ""
                output_file_for_model = (
                    Path(output_dir)
                    / f"results_{slugify(dataset_name)}_{model_slug}{think_slug}{tag_slug}.jsonl"
                )
                if output_file_for_model.exists():
                    output_file_for_model.unlink()
            results[model] = dict()

            if cascade is None:
                logging.info("Warming up model %s", model)
                client.call(model, "Say hello.", think=False, options={})
            else:
                for cascade_model, _ in cascade:
                    logging.info("Warming up cascade model %s", cascade_model)
                    client.call(cascade_model, "Say hello.", think=False, options={})

            if workers <= 1:
                for r in dataset:
                    if cascade is not None:
                        execution_log = run_cascade_flow(
                            r, cascade=cascade, think=think
                        )
                    else:
                        execution_log = run_flow(r, model, think)
                    execution_log["dataset"] = dataset_name
                    if output_file_for_model:
                        append_jsonl(str(output_file_for_model), execution_log)
                    results[model][r["task_id"]] = execution_log
                    progress.advance(problem_task)
            else:
                _jsonl_lock = threading.Lock()

                def _run_one(row):
                    if cascade is not None:
                        return row["task_id"], run_cascade_flow(
                            row, cascade=cascade, think=think
                        )
                    return row["task_id"], run_flow(row, model, think)

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_run_one, r): r["task_id"] for r in dataset}
                    for future in as_completed(futures):
                        try:
                            task_id, execution_log = future.result()
                        except Exception:
                            task_id = futures[future]
                            logging.exception("Worker failed for %s", task_id)
                            execution_log = {
                                "task_id": task_id,
                                "iterations": [],
                                "final_result": {
                                    "stderr": "Worker thread raised an exception",
                                    "stdout": "",
                                    "exit_code": 1,
                                },
                            }
                        execution_log["dataset"] = dataset_name
                        if output_file_for_model:
                            with _jsonl_lock:
                                append_jsonl(str(output_file_for_model), execution_log)
                        results[model][task_id] = execution_log
                        progress.advance(problem_task)

            if cascade is None:
                logging.info("Unloading model %s", model)
                client.stop(model)
            else:
                for cascade_model, _ in cascade:
                    logging.info("Unloading cascade model %s", cascade_model)
                    client.stop(cascade_model)
            progress.advance(model_task)
    return results


def is_error(results: dict[str, object] | None) -> bool:
    """Checks if the given results dictionary indicates an error condition.
    Returns True if the results dictionary is empty or contains non-empty stderr or stdout values.
    Args:
        results (dict[str, object]): A dictionary containing 'stdout' and 'stderr' keys.
    Returns:
        bool: True if there is an error, False otherwise."""
    if not results:
        return True
    if results.get("stderr", "") or results.get("stdout", ""):
        return True
    return False
