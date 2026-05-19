import glob
import json
import logging

from rich.logging import RichHandler

from ollama_codeeval.config import OUTPUT_BASE
from ollama_codeeval.report._shared import (
    CSS_BLOCK,
    HLJS_BLOCK,
    OUTPUT_DIR,
    SORT_SCRIPT,
    ns_to_seconds,
    sanitize_task_id,
)
from ollama_codeeval.report.html_context import generate_context_html
from ollama_codeeval.report.html_index import generate_index_html
from ollama_codeeval.report.html_rewrite import generate_rewrite_html
from ollama_codeeval.report.html_selector import generate_selector_html
from ollama_codeeval.report.html_tasks import generate_task_pages
from ollama_codeeval.report.html_yield import generate_yield_html
from ollama_codeeval.report.html_setup import generate_setup_html
from ollama_codeeval.report.metrics import process_jsonl_file

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

# Re-exported for consumers that import these names from this module (e.g. html_tasks)
__all__ = ["CSS_BLOCK", "HLJS_BLOCK", "OUTPUT_DIR", "SORT_SCRIPT", "ns_to_seconds", "sanitize_task_id"]


def get_all_data(no_cache=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_files = glob.glob(str(OUTPUT_BASE / "*.jsonl"))
    all_data = []
    log.info("found %d files to process: %s", len(jsonl_files), jsonl_files)
    for jsonl_file in jsonl_files:
        summary = process_jsonl_file(jsonl_file, no_cache=no_cache)
        all_data.append(summary)
    return all_data


def write_summary_json(all_data):
    with open(OUTPUT_BASE / "summary.json", "w", encoding="utf-8") as f:
        serializable_data = [
            {k: v for k, v in summary.items() if k != "tasks"} for summary in all_data
        ]
        json.dump(serializable_data, f, indent=2)


def main(no_cache=False):
    all_data = get_all_data(no_cache=no_cache)
    full_count = max(d["total_tests"] for d in all_data)
    all_data = [d for d in all_data if d["total_tests"] == full_count]
    base_data = [d for d in all_data if d.get("dataset", "humaneval") == "humaneval"]
    write_summary_json(base_data)
    generate_index_html(base_data, no_cache=no_cache)
    generate_rewrite_html(all_data)
    generate_task_pages(all_data)
    generate_context_html(base_data)
    generate_yield_html(base_data)
    generate_selector_html(base_data)
    generate_setup_html()


if __name__ == "__main__":
    main()
