"""Typer CLI entry point for ollama-codeeval."""

import asyncio

import typer

app = typer.Typer(help="Automated evaluation of LLM code generation using HumanEval.")


@app.command()
def eval(
    models: str = typer.Argument(
        "qwen3:4b", help="Comma-separated list of model names"
    ),
    host: str = typer.Option(None, help="Ollama host URL (default: from config)"),
    think: bool = typer.Option(False, help="Enable thinking mode"),
    output_dir: str = typer.Option("output", help="Output directory for JSONL results"),
    workers: int = typer.Option(
        1, help="Concurrent problem workers (increase for small models)"
    ),
    dataset: str = typer.Option(
        "human-eval-enhanced-202307",
        help="Dataset slug to evaluate (must match a file in data/<slug>.jsonl.gz)",
    ),
    tag: str = typer.Option(
        None, help="Tag suffix appended to output filename (e.g. 'fixharder-v2')"
    ),
    use_cascade: bool = typer.Option(
        False,
        "--cascade",
        help="Use multi-tier cascade flow (qwen3:4b → qwen2.5-coder → qwen3:14b)",
    ),
):
    """Run HumanEval evaluation on specified models."""

    from rich import print

    from ollama_codeeval.agent import client, close_sandbox
    from ollama_codeeval.cascade_agent import DEFAULT_CASCADE
    from ollama_codeeval.config import OLLAMA_HOST
    from ollama_codeeval.data import DATA_DIR, load_data
    from ollama_codeeval.runner import is_error, run_eval

    client.host = host or OLLAMA_HOST

    dataset_path = DATA_DIR / f"{dataset}.jsonl.gz"
    if not dataset_path.exists():
        print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    print(f"Loading dataset '{dataset}'...")
    dataset_data = load_data(path=dataset_path)
    print(f"Loaded {len(dataset_data)} problems.")
    model_list = [m.strip() for m in models.split(",")]
    cascade = DEFAULT_CASCADE if use_cascade else None

    try:
        result = asyncio.run(
            run_eval(
                dataset_data,
                models=model_list,
                think=think,
                output_dir=output_dir,
                workers=workers,
                dataset_name=dataset,
                tag=tag,
                cascade=cascade,
            )
        )

        for model, tasks in result.items():
            score = 0
            print("-" * 40)
            print(f"Model: {model}")
            for task_id, task in tasks.items():
                err = is_error(task["final_result"])
                iterations = len(task["iterations"])
                print(
                    f"Task: {task_id} ({('OK' if not err else 'ERROR')}) Iterations: {iterations}"
                )
                if not err:
                    score += 1
            print(f"Model {model} scored {score} ( {score / len(tasks)})")
    finally:
        close_sandbox()


@app.command()
def report(
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Regenerate all report files even if inputs are unchanged",
    ),
):
    """Generate HTML reports from evaluation results."""
    from ollama_codeeval.config import OUTPUT_BASE
    from ollama_codeeval.discover_model_stats import refresh_missing
    from ollama_codeeval.report.analyse_envelope import analyze_performance_envelope
    from ollama_codeeval.report.html_report import main as report_main
    from ollama_codeeval.report.results_parser import extract_data_from_directory

    refresh_missing()
    processed = str(OUTPUT_BASE / "processed_results.json")
    report_main(no_cache=no_cache)
    extract_data_from_directory(str(OUTPUT_BASE), processed, no_cache=no_cache)
    analyze_performance_envelope(processed, no_cache=no_cache)


if __name__ == "__main__":
    app()
