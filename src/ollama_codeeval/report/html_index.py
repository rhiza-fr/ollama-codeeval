import html

import numpy as np
from scipy.stats import gaussian_kde

from ollama_codeeval.report._shared import (
    CANONICAL_TAU,
    CSS_BLOCK,
    HEAD_META,
    OUTPUT_DIR,
    SORT_SCRIPT,
    _nav_html,
    compute_yield,
    tag_display_name,
)
from ollama_codeeval.report.html_violin import generate_charts_html

_TOGGLE_SCRIPT = """<script>
function toggleSecondary() {
  var cols = document.querySelectorAll('.col-secondary');
  var btn = document.getElementById('btn-toggle-cols');
  var hidden = cols[0] && cols[0].style.display === 'none' || getComputedStyle(cols[0]).display === 'none';
  cols.forEach(function(el) { el.style.display = hidden ? 'table-cell' : 'none'; });
  btn.textContent = hidden ? 'Hide extra columns' : 'Show extra columns';
}
</script>"""


def _compute_all_yield(all_data: list) -> None:
    """Compute yield score for each model in-place."""
    for d in all_data:
        d["yield"] = compute_yield(d["tasks"])


def _model_label(data):
    dataset = data.get("dataset", "humaneval")
    tag = data.get("tag")
    return f"{data['model']} (Think: {data['think']}{f', Dataset: {dataset}' if dataset != 'humaneval' else ''}{f', Tag: {tag}' if tag else ''})"


def _build_tasks_by_id(all_data):
    tasks_by_id = {}
    for data in all_data:
        model_name = _model_label(data)
        for task in data["tasks"]:
            task_id = task["input"]["task_id"]
            tasks_by_id.setdefault(task_id, {})[model_name] = task
    return tasks_by_id


def _best_model_strategy(all_data, tasks_by_id, initial_task_ids, num_iterations):
    total_problems = len(initial_task_ids)
    best_details = {}
    highest_score = -1
    for data in all_data:
        if data.get("dataset", "humaneval") != "humaneval":
            continue
        if (data.get("tag") or "").startswith(("cascade", "champion")):
            continue
        model_name = _model_label(data)
        unsolved = set(initial_task_ids)
        solved_count = 0
        iter_details = []
        for i in range(num_iterations):
            newly_solved = {
                tid
                for tid in unsolved
                if tasks_by_id.get(tid, {}).get(model_name)
                and i < len(tasks_by_id[tid][model_name]["iterations"])
                and tasks_by_id[tid][model_name]["iterations"][i]
                .get("test_result", {})
                .get("exit_code")
                == 0
            }
            solved_count += len(newly_solved)
            unsolved -= newly_solved
            iter_details.append(
                {"solved": solved_count, "percent": solved_count / total_problems * 100}
            )
        if solved_count > highest_score:
            highest_score = solved_count
            best_details = {"name": model_name, "iterations": iter_details}
    return best_details, highest_score


def _build_strategy_html(best_details, highest_score, total_problems):
    out = "<h2>Best Single Model PassRate@5</h2>"
    out += f"<p>The best performing single model is <b>{best_details['name']}</b>, which solves {highest_score}/{total_problems} problems.</p>"
    out += "\n    <table>\n      <tr><th>Attempt</th><th>Cumulative Solved</th><th>Cumulative %</th></tr>\n    "
    for i, detail in enumerate(best_details["iterations"]):
        out += f"<tr><td>{i + 1}</td><td>{detail['solved']}</td><td>{detail['percent']:.2f}%</td></tr>"
    return out + "</table>"


def _build_error_table(all_data):
    base_data = [d for d in all_data if d.get("dataset", "humaneval") == "humaneval"]
    all_error_keys = sorted(base_data[0]["error_counts"].keys())
    error_keys = [
        k
        for k in all_error_keys
        if any(d["error_counts"].get(k, 0) > 0 for d in base_data)
    ]
    out = "<h2>Iteration Result Statistics</h2>"
    out += "<p>Counts of success and various error types across all iterations for each model.</p>"
    out += '\n    <table>\n      <thead><tr><th class="sortable">Model</th>'
    for key in error_keys:
        out += f'<th class="sortable">{html.escape(key)}</th>'
    out += "</tr></thead>\n<tbody>\n"
    base_data = [d for d in all_data if d.get("dataset", "humaneval") == "humaneval"]
    err_sorted = sorted(
        base_data, key=lambda x: x["error_counts"].get("Success", 0), reverse=True
    )
    for data in err_sorted:
        out += f"<tr><td>{html.escape(_display_name(data))}</td>"
        total_iters = sum(data["error_counts"].values())
        for key in error_keys:
            count = data["error_counts"].get(key, 0)
            pct = (count / total_iters * 100) if total_iters > 0 else 0
            out += f'<td data-sort-value="{count}">{count} ({pct:.1f}%)</td>'
        out += "</tr>\n"
    return out + "</tbody></table>"


def _display_name(d):
    return tag_display_name(d)


def _build_stat_cards(all_data):
    base = [
        d
        for d in all_data
        if d.get("dataset", "humaneval") == "humaneval"
        and not (d.get("tag") or "").startswith(("cascade", "champion"))
    ]
    best_pass = max(d["passed_tests"] / d["total_tests"] * 100 for d in base)
    best_pass_model = next(
        _display_name(d)
        for d in base
        if d["passed_tests"] / d["total_tests"] * 100 == best_pass
    )
    best_spm = max(d.get("success_per_minute", 0) for d in base)
    best_spm_model = next(
        _display_name(d) for d in base if d.get("success_per_minute", 0) == best_spm
    )
    fastest = min(d["average_time_per_iteration"] for d in base)
    fastest_model = next(
        _display_name(d) for d in base if d["average_time_per_iteration"] == fastest
    )
    best_yield = max(d.get("yield", 0) for d in base)
    best_yield_model = next(
        _display_name(d) for d in base if d.get("yield", 0) == best_yield
    )
    return f"""<div class="stat-cards">
  <div class="stat-card accent-blue">
    <div class="stat-label">Best Pass Rate</div>
    <div class="stat-value">{best_pass:.1f}%</div>
    <div class="stat-sub">{html.escape(best_pass_model)}</div>
  </div>
  <div class="stat-card accent-amber">
    <div class="stat-label">Fastest Avg Time / Iter</div>
    <div class="stat-value">{fastest:.1f}s</div>
    <div class="stat-sub">{html.escape(fastest_model)}</div>
  </div>
  <div class="stat-card accent-green">
    <div class="stat-label">Best Pass / Min</div>
    <div class="stat-value">{best_spm:.3f}</div>
    <div class="stat-sub">{html.escape(best_spm_model)}</div>
  </div>
  <div class="stat-card accent-green">
    <div class="stat-label">Best Yield Score τ={CANONICAL_TAU:g}</div>
    <div class="stat-value">{best_yield * 100:.1f}%</div>
    <div class="stat-sub">{html.escape(best_yield_model)}</div>
  </div>
</div>"""


def _compute_distributions(all_data: list) -> dict:
    """Pre-compute KDE x/y arrays for each metric across all humaneval models."""
    models = [d for d in all_data if d.get("dataset", "humaneval") == "humaneval"]
    if len(models) < 2:
        return {}

    def kde_xy(values):
        arr = np.array(values, dtype=float)
        if arr.std() == 0:
            return [], []
        kde = gaussian_kde(arr)
        x = np.linspace(arr.min(), arr.max(), 200)
        return x.tolist(), kde(x).tolist()

    def make(x, y):
        return {"kde_x": x, "kde_y": y}

    pr_x, pr_y = kde_xy([d["passed_tests"] / d["total_tests"] * 100 for d in models])
    at_x, at_y = kde_xy([d["average_time_per_iteration"] for d in models])
    sp_x, sp_y = kde_xy([d.get("success_per_minute", 0) for d in models])
    yl_x, yl_y = kde_xy([d.get("yield", 0) * 100 for d in models])
    return {
        "pass_rate": make(pr_x, pr_y),
        "avg_time": make(at_x, at_y),
        "pass_per_min": make(sp_x, sp_y),
        "yield": make(yl_x, yl_y),
    }


def generate_index_html(all_data, no_cache=False):
    _compute_all_yield(all_data)
    distributions = _compute_distributions(all_data)
    from ollama_codeeval.report.html_model import generate_individual_html

    for d in all_data:
        generate_individual_html(
            d["file"], d, no_cache=no_cache, distributions=distributions
        )
    num_iterations = 6
    initial_task_ids = {task["input"]["task_id"] for task in all_data[0]["tasks"]}
    total_problems = len(initial_task_ids)
    tasks_by_id = _build_tasks_by_id(all_data)
    best_details, highest_score = _best_model_strategy(
        all_data, tasks_by_id, initial_task_ids, num_iterations
    )
    strategy_html = _build_strategy_html(best_details, highest_score, total_problems)
    error_table_html = _build_error_table(all_data)
    stat_cards_html = _build_stat_cards(all_data)

    with open(OUTPUT_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html>
<head>{HEAD_META}<title>Humaneval LLM Benchmark — Overview</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
{CSS_BLOCK}{SORT_SCRIPT}{_TOGGLE_SCRIPT}</head>
<body>
{_nav_html("index")}
<div class="page-content">
  <h1>Overview</h1>
  {stat_cards_html}
<h2>What is HumanEval?</h2>

<a href="https://github.com/openai/human-eval">HumanEval</a> is one of OpenAI's early benchmarks for evaluating LLM code generation. It consists of 164 handwritten Python programming problems. Each problem provides:

<ul>
<li>A function signature and docstring describing what the function should do</li>
<li>Example inputs/outputs in the docstring</li>
<li>A hidden test suite that validates correctness</li>
</ul>

The tasks are simple: complete the functions so that the generated code passes all hidden tests. A simple <a href="setup.html">harness</a> is used around each test.

<p>These test were run on models that can fit in mid range consumer GPUs. - speeds are limited by the GPUs that were used. No cloud models or APIs were tested.</p>

  {generate_charts_html(all_data)}
  <h2>Model Summary</h2>
  <button class="btn-toggle" id="btn-toggle-cols" onclick="toggleSecondary()">Show extra columns</button>
  <table>
    <thead><tr>
      <th class="sortable">Model</th>
      <th class="sortable">Think</th>
      <th class="sortable col-secondary">Dataset</th>
      <th class="sortable col-secondary">Tag</th>
      <th class="sortable col-secondary">Total Tests</th>
      <th class="sortable">Passed</th>
      <th class="sortable col-secondary">Failed</th>
      <th class="sortable col-secondary">Total Time (s)</th>
      <th class="sortable">Avg Time/IT (s)</th>
      <th class="sortable col-secondary">Success/1K Tokens</th>
      <th class="sortable col-secondary">Success/m</th>
      <th class="sortable">Yield τ={CANONICAL_TAU:g}</th>
""")
        for i in range(1, 7):
            cls = "sortable" if i in (1, 5) else "sortable col-secondary"
            f.write(f'<th class="{cls}">Pass@{i}</th>\n')
        f.write("\n    </tr></thead>\n<tbody>\n")
        sorted_data = sorted(
            all_data, key=lambda x: x.get("passed_tests", 0), reverse=True
        )
        for summary in sorted_data:
            if summary.get("dataset", "humaneval") != "humaneval":
                continue
            percentages = "".join(
                f'<td{"" if i in (1, 5) else ' class="col-secondary"'} data-sort-value="{p:.4f}">{p:.2f}%</td>'
                for i, p in enumerate(summary["cum_pass_per_iteration"], start=1)
            )
            passed_percent = summary["passed_tests"] / summary["total_tests"] * 100
            tag_cell = html.escape(summary.get("tag") or "-")
            display_model = _display_name(summary)
            f.write(f"""
            <tr>
            <td><a href="{summary["html_file"]}">{html.escape(display_model)}</a></td>
            <td>{summary["think"]}</td>
            <td class="col-secondary">{html.escape(summary.get("dataset", "humaneval"))}</td>
            <td class="col-secondary">{tag_cell}</td>
            <td class="col-secondary" data-sort-value="{summary["total_tests"]}">{summary["total_tests"]}</td>
            <td data-sort-value="{summary["passed_tests"]}">{summary["passed_tests"]} ({passed_percent:.1f}%)</td>
            <td class="col-secondary" data-sort-value="{summary["failed_tests"]}">{summary["failed_tests"]}</td>
            <td class="col-secondary" data-sort-value="{summary["total_time"]:.3f}">{summary["total_time"]:.3f}</td>
            <td data-sort-value="{summary["average_time_per_iteration"]:.3f}">{summary["average_time_per_iteration"]:.3f}</td>
            <td class="col-secondary" data-sort-value="{summary.get("success_per_1k_tokens", 0):.3f}">{summary.get("success_per_1k_tokens", 0):.3f}</td>
            <td class="col-secondary" data-sort-value="{summary.get("success_per_minute", 0):.3f}">{summary.get("success_per_minute", 0):.3f}</td>
            <td data-sort-value="{summary.get("yield", 0):.4f}">{summary.get("yield", 0) * 100:.1f}%</td>
            {percentages}
            </tr>
            """)
        f.write("\n  </tbody></table>\n")
        f.write(f"\n{strategy_html}\n")
        f.write(f"\n{error_table_html}\n</div>\n</body>\n</html>\n")

    from ollama_codeeval.report.html_cascade import generate_cascade_html

    generate_cascade_html(all_data)
