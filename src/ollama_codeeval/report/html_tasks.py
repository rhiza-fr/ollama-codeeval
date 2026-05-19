"""Task-focused HTML report pages.

Generates tasks.html (index sorted by difficulty) and per-task detail pages
showing cross-model results for each HumanEval task.
"""

import html
import json
import re

from ollama_codeeval.report.analyse_difficulty import compute_difficulty_signals
from ollama_codeeval.report._shared import (
    CSS_BLOCK,
    EXPAND_SCRIPT,
    HEAD_META,
    HLJS_BLOCK,
    OUTPUT_DIR,
    SORT_SCRIPT,
    _nav_html,
    anchor_id,
    expandable_code_html,
    ns_to_seconds,
    sanitize_task_id,
)


def _natural_sort_key(text):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text)]


def _model_key(data):
    dataset = data.get("dataset", "humaneval")
    return f"{data['model']} (Think: {data['think']}{f', Dataset: {dataset}' if dataset != 'humaneval' else ''})"


def _is_rewrite(data):
    return data.get("dataset", "humaneval") != "humaneval"


def generate_task_pages(all_data):
    """Generate task-focused HTML pages from evaluation data.

    Args:
        all_data: List of summary dicts (same as generate_index_html receives).
    """
    if not all_data:
        return

    # Build model lists: base models first, rewrite models second (stable order)
    base_names = []
    rewrite_names = []
    seen = set()
    for data in all_data:
        key = _model_key(data)
        if key not in seen:
            seen.add(key)
            (rewrite_names if _is_rewrite(data) else base_names).append(key)

    rewrite_keys = set(rewrite_names)

    # Build per-task cross-model data: {task_id: {model_key: task_dict}}
    tasks_by_id: dict[str, dict] = {}
    task_inputs: dict[str, dict] = {}
    for data in all_data:
        key = _model_key(data)
        for task in data["tasks"]:
            tid = task["input"]["task_id"]
            if tid not in tasks_by_id:
                tasks_by_id[tid] = {}
                task_inputs[tid] = task["input"]
            tasks_by_id[tid][key] = task

    # Compute per-task stats — pass rate uses base models only for fair sorting
    task_stats = []
    for tid, model_results in tasks_by_id.items():
        base_results = {k: v for k, v in model_results.items() if k not in rewrite_keys}
        passed = sum(
            1 for t in base_results.values() if t["final_result"]["exit_code"] == 0
        )
        total = len(base_results)
        rate = passed / total if total > 0 else 0
        task_stats.append(
            {
                "task_id": tid,
                "passed": passed,
                "total": total,
                "rate": rate,
                "results": model_results,
            }
        )

    # Sort by pass rate ascending (hardest first) for default
    task_stats.sort(key=lambda s: (s["rate"], _natural_sort_key(s["task_id"])))

    difficulty_signals = compute_difficulty_signals(all_data)
    for stat in task_stats:
        stat["difficulty"] = difficulty_signals.get(stat["task_id"])

    # Build model_key → html_file mapping for cross-links in task detail pages
    model_html_files = {
        _model_key(data): data.get("html_file", "") for data in all_data
    }

    _generate_tasks_index(task_stats, base_names)
    for stat in task_stats:
        _generate_task_detail(
            stat, base_names, task_inputs[stat["task_id"]], model_html_files
        )


def _pass_bar(passed, total):
    if total == 0:
        return ""
    pct = passed / total * 100
    fail_pct = 100 - pct
    return (
        f'<div style="display:flex;width:100%;height:10px;border-radius:3px;overflow:hidden" title="{passed}/{total} passed">'
        f'<div style="width:{pct:.1f}%;background:#16a34a"></div>'
        f'<div style="width:{fail_pct:.1f}%;background:#dc2626"></div>'
        f"</div>"
    )


def _chart_difficulty_scatter(task_stats: list[dict]) -> str:
    """Scatter: pass rate vs prompt length, coloured by task type label."""
    groups: dict[str, list] = {"description?": [], "algorithm?": [], "": []}
    for stat in task_stats:
        sig = stat.get("difficulty")
        groups[sig.label if sig else ""].append(stat)

    colors = {"description?": "#e67e22", "algorithm?": "#2980b9", "": "#95a5a6"}
    names = {"description?": "Description", "algorithm?": "Algorithm", "": "Unclear"}
    traces = []
    for label, items in groups.items():
        if not items:
            continue
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "name": names[label],
                "x": [s["rate"] * 100 for s in items],
                "y": [
                    s["difficulty"].prompt_chars if s.get("difficulty") else 0
                    for s in items
                ],
                "text": [
                    f"{s['task_id']}<br>"
                    f"Pass rate: {s['rate'] * 100:.0f}%<br>"
                    + (
                        f"Examples: {s['difficulty'].n_examples}<br>"
                        f"Chars: {s['difficulty'].prompt_chars}<br>"
                        f"Iter-1 pass: {s['difficulty'].iter1_assertion_rate:.0%}"
                        if s.get("difficulty")
                        else ""
                    )
                    for s in items
                ],
                "hoverinfo": "text",
                "marker": {
                    "size": [
                        max(6, min(20, s["difficulty"].iter1_assertion_rate * 20))
                        if s.get("difficulty")
                        else 8
                        for s in items
                    ],
                    "color": colors[label],
                    "opacity": 0.7,
                },
            }
        )

    layout = {
        "xaxis": {"title": "Pass rate (%)", "range": [-2, 102]},
        "yaxis": {"title": "Prompt length (chars)"},
        "hovermode": "closest",
        "legend": {"orientation": "h", "y": -0.15},
        "margin": {"b": 80},
        "height": 500,
    }
    return (
        '<div id="chartDifficulty"></div>\n'
        f'<script>Plotly.newPlot("chartDifficulty", {json.dumps(traces)}, {json.dumps(layout)});</script>\n'
    )


def _generate_tasks_index(task_stats, base_names):
    """Write tasks.html."""
    rows = ""
    for stat in task_stats:
        tid = stat["task_id"]
        safe_id = sanitize_task_id(tid)
        rate_pct = stat["rate"] * 100
        sig = stat.get("difficulty")
        label_val = html.escape(sig.label) if sig and sig.label else "—"
        bar = _pass_bar(stat["passed"], stat["total"])
        rows += f"""<tr>
    <td><a href="task_{safe_id}.html">{html.escape(tid)}</a></td>
    <td data-sort-value="{stat["passed"]}">{stat["passed"]}/{stat["total"]}</td>
    <td data-sort-value="{stat["rate"]:.4f}">{rate_pct:.1f}%</td>
    <td>{label_val}</td>
    <td>{bar}</td>
</tr>\n"""

    chart_html = _chart_difficulty_scatter(task_stats)
    page = f"""<!doctype html>
<html>
<head>{HEAD_META}<title>Task Difficulty</title>{CSS_BLOCK}{SORT_SCRIPT}
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
{_nav_html("tasks")}
<div class="page-content">
  <h1>Task Difficulty</h1>
  <p>Pass rate vs prompt length. Marker size = iter-1 pass rate. Colour = task type.</p>
  {chart_html}
  <table style="width:100%">
    <thead><tr>
      <th class="sortable">Task ID</th>
      <th class="sortable">Passed</th>
      <th class="sortable">Pass Rate</th>
      <th class="sortable">Type</th>
      <th>Models</th>
    </tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</div>
</body>
</html>
"""
    (OUTPUT_DIR / "tasks.html").write_text(page, encoding="utf-8")


def _chart_pass_times(stat: dict, model_names: list) -> str:
    """Horizontal bar chart of total pass times, log-scaled, fastest at top."""
    passing = []
    for mname in model_names:
        task_data = stat["results"].get(mname)
        if task_data is None or task_data["final_result"]["exit_code"] != 0:
            continue
        total_ns = sum(
            it.get("total_duration", 0) or 0 for it in task_data["iterations"]
        )
        total_s = total_ns / 1_000_000_000.0
        if total_s > 0:
            passing.append((total_s, mname))

    if not passing:
        return ""

    passing.sort()  # fastest first → top of chart (Plotly h-bar: first = bottom, so reverse)
    passing.reverse()  # Plotly renders y-axis top-to-bottom reversed; reversing puts fastest at top

    labels = [p[1] for p in passing]
    values = [round(p[0], 3) for p in passing]
    hover = [f"{name}<br>{t:.2f}s" for t, name in zip(values, labels)]

    trace = {
        "type": "bar",
        "orientation": "h",
        "x": values,
        "y": labels,
        "text": [f"{v:.1f}s" for v in values],
        "textposition": "outside",
        "hovertext": hover,
        "hoverinfo": "text",
        "marker": {"color": "#16a34a"},
        "cliponaxis": False,
    }
    layout = {
        "xaxis": {
            "type": "log",
            "title": "Total time (s, log scale)",
            "tickvals": [0.5, 1, 2, 5, 10, 30, 60, 120],
            "ticktext": ["0.5", "1", "2", "5", "10", "30", "60", "120"],
        },
        "yaxis": {"automargin": True},
        "margin": {"l": 10, "r": 60, "t": 10, "b": 40},
        "height": max(100, len(passing) * 28 + 60),
        "showlegend": False,
        "plot_bgcolor": "#fff",
        "paper_bgcolor": "#fff",
    }
    cid = f"chartPassTimes_{sanitize_task_id(stat['task_id'])}"
    return (
        f'<div id="{cid}" style="width:100%;"></div>\n'
        f"<script>Plotly.newPlot('{cid}', [{json.dumps(trace)}], {json.dumps(layout)}, {{responsive:true}});</script>\n"
    )


def _generate_task_detail(
    stat, model_names, task_input, model_html_files: dict | None = None
):
    """Write a per-task detail page."""
    tid = stat["task_id"]
    safe_id = sanitize_task_id(tid)
    rate_pct = stat["rate"] * 100

    sig = stat.get("difficulty")
    if sig:
        label_str = sig.label if sig.label else "—"
        difficulty_html = (
            "<h2>Difficulty Signals</h2>\n"
            "<dl class='signals-dl'>\n"
            f"  <dt>Iter-1 assertion rate</dt><dd>{sig.iter1_assertion_rate:.0%}</dd>\n"
            f"  <dt>Prompt</dt><dd>{sig.prompt_chars} chars, {sig.n_examples} examples</dd>\n"
            f"  <dt>Type</dt><dd>{html.escape(label_str)}</dd>\n"
            "</dl>\n"
        )
    else:
        difficulty_html = ""

    prompt_html = html.escape(
        task_input.get("original_input_prompt") or task_input.get("prompt", "")
    )
    canonical_html = html.escape(task_input.get("canonical_solution", ""))
    test_html = html.escape(task_input.get("test", ""))

    # Cross-model results table — collect, sort (pass first, then quickest), then render
    result_entries = []
    for mname in model_names:
        task_data = stat["results"].get(mname)
        if task_data is None:
            continue
        passed = task_data["final_result"]["exit_code"] == 0
        iterations = task_data["iterations"]
        total_ns = sum(it.get("total_duration", 0) or 0 for it in iterations)
        total_s = total_ns / 1_000_000_000.0
        model_href = (model_html_files or {}).get(mname, "")
        model_cell = (
            f'<a href="{html.escape(model_href)}">{html.escape(mname)}</a>'
            if model_href
            else html.escape(mname)
        )
        result_entries.append((passed, total_s, model_cell, len(iterations)))

    result_entries.sort(key=lambda e: (not e[0], e[1]))

    result_rows = ""
    for passed, total_s, model_cell, n_iter in result_entries:
        result_label = (
            "<span class='pass'>Pass</span>"
            if passed
            else "<span class='fail'>Fail</span>"
        )
        result_rows += f"""<tr>
            <td>{model_cell}</td>
            <td data-sort-value="{0 if passed else 1}">{result_label}</td>
            <td>{n_iter}</td>
            <td data-sort-value="{total_s:.3f}">{total_s:.3f}</td>
        </tr>\n"""

    # Per-model iteration details
    details_html = ""
    for mname in model_names:
        task_data = stat["results"].get(mname)
        if task_data is None:
            continue
        iterations = task_data["iterations"]
        model_anchor = anchor_id(mname)
        iter_content = ""
        for i, it in enumerate(iterations):
            it_time = ns_to_seconds(it.get("total_duration"))
            prompt = expandable_code_html(it.get("prompt", ""))
            msg = it.get("message", {})
            code = expandable_code_html(msg.get("content", ""))
            thinking = msg.get("thinking")
            test_result = it.get("test_result")
            if test_result is None:
                test_result_html = "<p>N/A</p>"
            elif test_result.get("exit_code") == 0:
                test_result_html = "<p class='pass-bold'>Pass</p>"
            else:
                stderr = expandable_code_html(test_result.get("stderr", ""))
                test_result_html = f"<pre>{stderr}</pre>"

            iter_anchor = f"{model_anchor}_{i + 1}"
            iter_content += (
                f"<h4 id='{iter_anchor}'>Iteration {i + 1} ({it_time} s)</h4>\n"
            )
            iter_content += f"<details id='{iter_anchor}_prompt'><summary>Prompt</summary><pre>{prompt}</pre></details>\n"
            if thinking:
                iter_content += f"<details id='{iter_anchor}_thinking'><summary>Thinking</summary><pre>{expandable_code_html(thinking)}</pre></details>\n"
            iter_content += f"<details id='{iter_anchor}_code'><summary>Generated Code</summary><pre>{code}</pre></details>\n"
            iter_content += test_result_html + "\n"

        details_html += f"""<details id="{model_anchor}">
          <summary>{html.escape(mname)}</summary>
          {iter_content}
        </details>\n"""

    passed_count = stat["passed"]
    total_count = stat["total"]
    stat_card_color = (
        "accent-green"
        if rate_pct >= 50
        else "accent-blue"
        if rate_pct > 0
        else "accent-amber"
    )

    page = f"""<!doctype html>
<html>
<head>{HEAD_META}<title>Task: {html.escape(tid)}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
{CSS_BLOCK}{SORT_SCRIPT}{HLJS_BLOCK}{EXPAND_SCRIPT}</head>
<body>
{_nav_html("tasks")}
<div class="page-content">
  <div class="breadcrumb">
    <a href="index.html">Overview</a><span>/</span>
    <a href="tasks.html">Tasks</a><span>/</span>
    {html.escape(tid)}
  </div>
  <h1>{html.escape(tid)}</h1>
  <div class="stat-cards">
    <div class="stat-card {stat_card_color}">
      <div class="stat-label">Models Passed</div>
      <div class="stat-value">{passed_count} / {total_count}</div>
      <div class="stat-sub">{rate_pct:.1f}% pass rate</div>
    </div>
  </div>
  {_chart_pass_times(stat, model_names)}

  {difficulty_html}<h2 id="prompt">Task Prompt</h2>
  <pre><code>{prompt_html}</code></pre>

  <details id="canonical">
    <summary>Canonical Solution</summary>
    <pre><code>{canonical_html}</code></pre>
  </details>
  <details id="tests">
    <summary>Test Code</summary>
    <pre><code>{test_html}</code></pre>
  </details>

  <h2 id="results">Cross-Model Results</h2>
  <table>
    <thead><tr>
      <th class="sortable">Model</th>
      <th class="sortable">Result</th>
      <th class="sortable">Iterations</th>
      <th class="sortable">Time (s)</th>
    </tr></thead>
    <tbody>
{result_rows}
    </tbody>
  </table>

  <h2 id="iterations">Iteration Details</h2>
{details_html}
</div>
</body>
</html>
"""
    (OUTPUT_DIR / f"task_{safe_id}.html").write_text(page, encoding="utf-8")
