"""Standalone HTML report for context window usage and explosion analysis."""

import html as _html
import json

from ollama_codeeval.report._shared import CSS_BLOCK, OUTPUT_DIR, SORT_SCRIPT, _nav_html, sanitize_task_id


def _extract_context_stats(data: dict) -> dict:
    """Compute context window explosion statistics from a single model's summary dict."""
    explosions = 0
    total_iters = 0
    token_totals: list[int] = []
    tasks_exploded = 0
    tasks_stuck = 0
    task_details: list[dict] = []

    for task in data.get("tasks", []):
        task_had_explosion = False
        iter_rows: list[dict] = []

        for it in task.get("iterations", []):
            total_iters += 1
            eval_count = it.get("eval_count") or 0
            prompt_eval_count = it.get("prompt_eval_count") or 0
            tokens = eval_count + prompt_eval_count
            token_totals.append(tokens)
            done_reason = it.get("done_reason", "stop")
            exit_code = (it.get("test_result") or {}).get("exit_code", -1)

            iter_rows.append({
                "tokens": tokens,
                "done_reason": done_reason,
                "exit_code": exit_code,
            })

            if done_reason == "length":
                explosions += 1
                task_had_explosion = True

        if task_had_explosion:
            tasks_exploded += 1
            passed = task.get("final_result", {}).get("exit_code") == 0
            if not passed:
                tasks_stuck += 1  # had explosion AND failed (explosion may not be the cause)
            task_details.append({
                "task_id": task["input"]["task_id"],
                "iter_rows": iter_rows,
                "passed": passed,
            })

    return {
        "explosions": explosions,
        "total_iters": total_iters,
        "expl_pct": explosions / total_iters * 100 if total_iters else 0.0,
        "max_tokens": max(token_totals) if token_totals else 0,
        "avg_tokens": sum(token_totals) / len(token_totals) if token_totals else 0.0,
        "tasks_exploded": tasks_exploded,
        "tasks_stuck": tasks_stuck,
        "task_details": task_details,
        "token_totals": token_totals,
    }


def _build_summary_table(entries: list[dict]) -> str:
    """Build a sortable HTML summary table. Each entry must have a '_ctx' key from _extract_context_stats."""
    out = (
        '<table>\n'
        '<thead><tr>'
        '<th class="sortable">Model</th>'
        '<th class="sortable">Total Iters</th>'
        '<th class="sortable">Explosions</th>'
        '<th class="sortable">Expl%</th>'
        '<th class="sortable">Max Tokens</th>'
        '<th class="sortable">Avg Tokens</th>'
        '<th class="sortable">Tasks w/ Expl</th>'
        '<th class="sortable">Tasks Stuck</th>'
        '</tr></thead>\n<tbody>\n'
    )
    sorted_entries = sorted(entries, key=lambda e: -e["_ctx"]["expl_pct"])
    for e in sorted_entries:
        ctx = e["_ctx"]
        pct = ctx["expl_pct"]
        model_name = _html.escape(_model_label(e))
        html_file = _html.escape(e.get("html_file", ""))
        model_cell = f'<a href="{html_file}">{model_name}</a>' if html_file else model_name
        out += (
            f'<tr>'
            f'<td>{model_cell}</td>'
            f'<td data-sort-value="{ctx["total_iters"]}">{ctx["total_iters"]}</td>'
            f'<td data-sort-value="{ctx["explosions"]}">{ctx["explosions"]}</td>'
            f'<td data-sort-value="{pct:.2f}">{pct:.1f}%</td>'
            f'<td data-sort-value="{ctx["max_tokens"]}">{ctx["max_tokens"]:,}</td>'
            f'<td data-sort-value="{ctx["avg_tokens"]}">{ctx["avg_tokens"]:,.0f}</td>'
            f'<td data-sort-value="{ctx["tasks_exploded"]}">{ctx["tasks_exploded"]}</td>'
            f'<td data-sort-value="{ctx["tasks_stuck"]}">{ctx["tasks_stuck"]}</td>'
            f'</tr>\n'
        )
    return out + '</tbody></table>\n'


def _model_label(entry: dict) -> str:
    parts = []
    if entry["think"]:
        parts.append("Think")
    dataset = entry.get("dataset", "humaneval")
    if dataset != "humaneval":
        parts.append(f"Dataset: {dataset}")
    suffix = f" ({', '.join(parts)})" if parts else ""
    return f"{entry['model']}{suffix}"


def _chart_explosion_rate(entries: list[dict]) -> str:
    """Horizontal bar chart: explosion rate per model, sorted ascending (highest at top)."""
    if not entries:
        return ""
    sorted_e = sorted(entries, key=lambda e: e["_ctx"]["expl_pct"])
    labels = [_model_label(e) for e in sorted_e]
    values = [round(e["_ctx"]["expl_pct"], 1) for e in sorted_e]
    hover = [
        f"{_model_label(e)}<br>{e['_ctx']['explosions']} / {e['_ctx']['total_iters']} iters"
        for e in sorted_e
    ]
    max_val = max(values) if values else 100
    traces = [{
        "type": "bar",
        "orientation": "h",
        "x": values,
        "y": labels,
        "text": hover,
        "hoverinfo": "text",
        "marker": {
            "color": values,
            "colorscale": [[0, "#2ecc71"], [0.3, "#f1c40f"], [1, "#e74c3c"]],
            "cmin": 0,
            "cmax": max_val,
        },
    }]
    layout = {
        "xaxis": {"title": "Context explosion rate (%)", "range": [0, max_val * 1.1]},
        "margin": {"l": 220, "b": 60},
        "height": max(300, len(labels) * 28 + 80),
    }
    return (
        '<div id="chartExplosionRate" style="width:100%;"></div>\n'
        f"<script>Plotly.newPlot('chartExplosionRate', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


def _chart_token_usage(entries: list[dict]) -> str:
    """Horizontal strip plot: each dot is one iteration, each row is one model."""
    if not entries:
        return ""
    active = [e for e in entries if e["_ctx"]["token_totals"]]
    if not active:
        return ""
    # Sort ascending by explosion rate so worst offenders are at the top
    active_sorted = sorted(active, key=lambda e: e["_ctx"]["expl_pct"])
    labels = [_model_label(e) for e in active_sorted]
    traces = []
    for e in active_sorted:
        label = _model_label(e)
        totals = e["_ctx"]["token_totals"]
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "name": label,
            "x": totals,
            "y": [label] * len(totals),
            "marker": {"size": 5, "opacity": 0.5},
            "showlegend": False,
        })
    # Reference line at the autocontext cap
    traces.append({
        "type": "scatter",
        "mode": "lines",
        "name": "Context cap (16 384)",
        "x": [16384, 16384],
        "y": [labels[0], labels[-1]],
        "line": {"color": "red", "dash": "dash", "width": 1},
        "showlegend": True,
    })
    layout = {
        "xaxis": {"title": "Tokens per iteration (prompt + completion)", "range": [0, 20000]},
        "margin": {"l": 220, "b": 60},
        "height": max(300, len(labels) * 28 + 80),
        "showlegend": True,
    }
    return (
        '<div id="chartTokenUsage" style="width:100%;"></div>\n'
        f"<script>Plotly.newPlot('chartTokenUsage', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


def _build_detail_table(all_entries: list[dict]) -> str:
    """Table of per-task explosion sequences for all tasks with ≥1 explosion."""
    rows = ""
    group_idx = 0
    prev_model = None

    for entry in sorted(all_entries, key=lambda e: e["model"]):
        ctx = entry["_ctx"]
        if not ctx["task_details"]:
            continue
        model_label = _model_label(entry)
        if prev_model is not None and model_label != prev_model:
            group_idx += 1
        prev_model = model_label
        grp = "group-even" if group_idx % 2 == 0 else "group-odd"
        html_file = _html.escape(entry.get("html_file", ""))
        model_cell = f'<a href="{html_file}">{_html.escape(model_label)}</a>' if html_file else _html.escape(model_label)

        for detail in ctx["task_details"]:
            raw_task_id = detail["task_id"]
            task_id = _html.escape(raw_task_id)
            task_cell = f'<a href="task_{sanitize_task_id(raw_task_id)}.html">{task_id}</a>'
            outcome = "pass" if detail["passed"] else "FAIL"
            outcome_style = "color:green" if detail["passed"] else "color:red;font-weight:bold"
            visible_rows = [(i, row) for i, row in enumerate(detail["iter_rows"]) if row["done_reason"] != "stop"]
            for j, (i, row) in enumerate(visible_rows):
                dr = row["done_reason"]
                dr_style = "color:red;font-weight:bold" if dr == "length" else ""
                outcome_text = outcome if j == len(visible_rows) - 1 else ""
                cell_style = outcome_style if outcome_text else ""
                rows += (
                    f'<tr class="{grp}">'
                    f'<td>{model_cell}</td>'
                    f'<td>{task_cell}</td>'
                    f'<td>{i + 1}</td>'
                    f'<td data-sort-value="{row["tokens"]}">{row["tokens"]:,}</td>'
                    f'<td style="{dr_style}">{dr}</td>'
                    f'<td style="{cell_style}">{outcome_text}</td>'
                    f'</tr>\n'
                )

    if not rows:
        return "<p>No context explosions detected.</p>"

    return (
        '<table>\n'
        '<thead><tr>'
        '<th class="sortable">Model</th>'
        '<th class="sortable">Task</th>'
        '<th class="sortable">Iter</th>'
        '<th class="sortable">Tokens Used</th>'
        '<th class="sortable">done_reason</th>'
        '<th>Outcome</th>'
        '</tr></thead>\n'
        f'<tbody>\n{rows}</tbody></table>\n'
    )


def generate_context_html(all_data: list[dict]) -> str:
    """Generate context_report.html. Returns the relative output file path."""
    entries = []
    for data in all_data:
        if data.get("dataset", "humaneval") != "humaneval":
            continue
        entry = dict(data)
        entry["_ctx"] = _extract_context_stats(data)
        if entry["_ctx"]["explosions"] == 0:
            continue
        entries.append(entry)

    summary_table = _build_summary_table(entries)
    bar_chart = _chart_explosion_rate(entries)
    box_chart = _chart_token_usage(entries)
    detail_table = _build_detail_table(entries)

    out_path = OUTPUT_DIR / "context_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"""\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Context Window Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
{CSS_BLOCK}
{SORT_SCRIPT}
</head>
<body>
{_nav_html("context")}
<div class="page-content">
<h1>Context Window Report</h1>
<p>
  Shows which models hit the autocontext cap (<code>done_reason: length</code>) during evaluation.
  Token usage is <code>eval_count + prompt_eval_count</code> &mdash; the actual context consumed per iteration.
  When a model stops due to <b>length</b>, it generated until the cap (max 16&nbsp;384 tokens) without finishing.
  AutoContext grows num_ctx by 1.5&times; on each explosion and never shrinks it, so one runaway task inflates
  inference time for all subsequent tasks in that run.
</p>

<p>
  Context explosions are generally seen in small models and thinking models. They are generally stuck in some loop of
  &ldquo;Oh, no wait, maybe&hellip;&rdquo; over and over &mdash; probably a symptom of being trained on deepseek-like
  thinking traces. Small models seem more prone to this.
</p>

<h2>Explosion Rate by Model</h2>
<p>Fraction of iterations that hit the context cap. Higher = more runaway outputs.</p>
{bar_chart}

<h2>Token Usage Distribution</h2>
<p>Each dot is one iteration. The red dashed line marks the autocontext cap (16&nbsp;384 tokens); dots clustered there are explosions.</p>
{box_chart}

<h2>Summary Table</h2>
<p>Sorted by explosion rate descending. <b>Tasks Stuck</b> = tasks that had &#8805;1 explosion AND failed overall.</p>
{summary_table}

<h2>Per-Task Explosion Detail</h2>
<p>Only tasks with at least one <code>done_reason: length</code> iteration. Shows token growth and outcome.</p>
{detail_table}
</div>
</body>
</html>
""")
    return str(out_path.relative_to(OUTPUT_DIR))
