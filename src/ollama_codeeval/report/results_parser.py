"""Processes result files to extract max_load_time and other relevant metadata, saving the results to an output file."""

import glob
import json
import logging
import os
from collections import defaultdict

from ollama_codeeval.report.metrics import _parse_tag_from_filename
from ollama_codeeval.report._shared import normalize_dataset

import numpy as np
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)


def extract_data_from_directory(input_directory, output_file, no_cache=False):
    """
    Processes result files, calculating max_load_time and other metadata.
    """
    raw_model_data = defaultdict(
        lambda: {"iterations": [], "tasks": [], "load_durations": []}
    )
    if not os.path.isdir(input_directory):
        log.error("Directory not found at '%s'", input_directory)
        return
    file_pattern = os.path.join(input_directory, "results_*.jsonl")
    result_files = glob.glob(file_pattern)
    if not result_files:
        log.warning(
            "No files matching 'results_*.jsonl' found in '%s'", input_directory
        )
        return
    if not no_cache and os.path.exists(output_file):
        output_mtime = os.path.getmtime(output_file)
        if all(os.path.getmtime(f) <= output_mtime for f in result_files):
            log.info("processed_results.json is up to date, skipping.")
            return
    log.info("Found %d files to process...", len(result_files))
    for file_path in result_files:
        log.info("Processing %s...", file_path)
        tag = _parse_tag_from_filename(file_path) or ""
        with open(file_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    dataset = normalize_dataset(record.get("dataset", "humaneval"))
                    model = "cascade" if tag.startswith(("cascade", "champion")) else record.get("model")
                    series_title = f"{model}_think={record.get('think', False)}_dataset={dataset}_tag={tag}"
                    for iteration in record.get("iterations", []):
                        if (duration := iteration.get("total_duration")) is not None:
                            raw_model_data[series_title]["iterations"].append(duration)
                        if (
                            load_duration := iteration.get("load_duration")
                        ) is not None:
                            raw_model_data[series_title]["load_durations"].append(
                                load_duration
                            )
                    total_duration_ns = sum(
                        it.get("total_duration") or 0
                        for it in record.get("iterations", [])
                    )
                    total_time_s = total_duration_ns / 1000000000
                    status = "success" if record.get("final_result", {}).get("exit_code") == 0 else "failure"
                    raw_model_data[series_title]["tasks"].append(
                        {
                            "task_id": record["input"]["task_id"],
                            "time_to_success": total_time_s if status == "success" else None,
                            "total_time_s": total_time_s,
                            "status": status,
                        }
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    log.warning(
                        "Skipping a problematic record in %s due to error: %s",
                        file_path,
                        e,
                    )
    processed_data = {}
    for series_title, data in raw_model_data.items():
        # CHANGED: Calculate MAXIMUM load time instead of average.
        all_load_durations_ns = data["load_durations"]
        if not all_load_durations_ns:
            max_load_time_s = 0
        else:
            max_load_time_s = np.max(all_load_durations_ns) / 1000000000
        # ... (rest of the metadata calculation is the same)
        avg_iteration_time_s = (
            np.mean(data["iterations"]) / 1000000000 if data["iterations"] else 0
        )
        tasks = sorted(
            [
                t
                for t in data["tasks"]
                if t["status"] == "success" and t["time_to_success"] is not None
            ],
            key=lambda x: x["time_to_success"],
        )
        timeseries = {"times": [0], "success_rates": [0.0]}
        unique_task_ids = {t["task_id"] for t in data["tasks"]}
        total_tasks = len(data["tasks"])
        if total_tasks > 0:
            for i, task in enumerate(tasks):
                timeseries["times"].append(task["time_to_success"])
                timeseries["success_rates"].append((i + 1) / total_tasks * 100.0)

        # Sequential timeseries: all tasks sorted by total_time_s, X = cumulative time, Y = cumulative successes.
        # Only emit a point on each success — failures silently advance cumulative_time,
        # so horizontal stretches between steps naturally represent failure time consumed.
        seq_timeseries = {"times": [0], "success_rates": [0.0]}
        if total_tasks > 0:
            all_tasks_sorted = sorted(data["tasks"], key=lambda t: t["total_time_s"])
            cumulative_time = 0
            cumulative_successes = 0
            for task in all_tasks_sorted:
                cumulative_time += task["total_time_s"]
                if task["status"] == "success":
                    cumulative_successes += 1
                    seq_timeseries["times"].append(cumulative_time)
                    seq_timeseries["success_rates"].append(cumulative_successes / total_tasks * 100.0)
            # Final point: extend to total cumulative time so the curve doesn't end early
            seq_timeseries["times"].append(cumulative_time)
            seq_timeseries["success_rates"].append(cumulative_successes / total_tasks * 100.0)

        processed_data[series_title] = {
            "metadata": {
                "avg_iteration_time_s": avg_iteration_time_s,
                "max_load_time_s": max_load_time_s,
                "total_tasks": total_tasks,
                "unique_tasks": len(unique_task_ids),
            },
            "timeseries": timeseries,
            "seq_timeseries": seq_timeseries,
        }
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)
    log.info("Processing complete. Data with max_load_time saved to '%s'", output_file)


if __name__ == "__main__":
    extract_data_from_directory("output", "output/processed_results.json")
