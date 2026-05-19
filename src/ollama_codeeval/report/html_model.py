import html
import json
import re
from pathlib import Path

from ollama_codeeval.config import OUTPUT_BASE
from ollama_codeeval.kde import KDEEstimator
from ollama_codeeval.report._shared import (
    CSS_BLOCK,
    EXPAND_SCRIPT,
    HEAD_META,
    HLJS_BLOCK,
    OUTPUT_DIR,
    _nav_html,
    anchor_id,
    compute_yield,
    expandable_code_html,
    load_model_stats,
    ns_to_seconds,
    resolve_model_name,
    sanitize_task_id,
    tag_display_name,
)


def _full_kde_chart(filtered_data, median, x_min, x_max, color, unit="", bins=30) -> str:
    """Render a full-size inline SVG histogram matching the mini-graph aesthetic."""
    import numpy as np

    counts, edges = np.histogram(filtered_data, bins=bins)
    W, H = 600, 115
    PAD_L, PAD_R, PAD_T, PAD_B = 8, 8, 14, 22
    chart_w = W - PAD_L - PAD_R
    chart_h = H - PAD_T - PAD_B
    x_range = edges[-1] - edges[0] or 1
    y_max = counts.max() or 1

    def sx(x):
        return PAD_L + (x - edges[0]) / x_range * chart_w

    def sy(c):
        return PAD_T + chart_h - c / y_max * chart_h * 0.9

    rects = ""
    for i, count in enumerate(counts):
        x0 = sx(edges[i])
        x1 = sx(edges[i + 1])
        y0 = sy(count)
        bottom = PAD_T + chart_h
        rects += (
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{max(0, x1-x0-1):.1f}" height="{bottom-y0:.1f}"'
            f' fill="{color}" fill-opacity="0.2" stroke="{color}" stroke-opacity="0.5" stroke-width="1"/>'
        )

    vx = sx(median)
    bottom = PAD_T + chart_h
    #fmt = lambda v: f"{v:.1f}{unit}"
    def fmt(v): 
        return f"{v:.1f}{unit}"

    font = "font-size='11' font-family='system-ui,sans-serif'"

    return (
        f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:{H}px;display:block;">'
        f'{rects}'
        f'<line x1="{vx:.1f}" y1="{PAD_T}" x2="{vx:.1f}" y2="{bottom}" stroke="{color}" stroke-width="2"/>'
        f'<text x="{PAD_L}" y="{H-4}" {font} fill="#94a3b8">{html.escape(fmt(x_min))}</text>'
        f'<text x="{W-PAD_R}" y="{H-4}" {font} fill="#94a3b8" text-anchor="end">{html.escape(fmt(x_max))}</text>'
        f'<text x="{vx+4:.1f}" y="{PAD_T+10}" {font} fill="{color}" text-anchor="start">median {html.escape(fmt(median))}</text>'
        f'</svg>'
    )


def _mini_kde_chart(dist: dict, current_value: float, color: str) -> str:
    """Return an inline SVG KDE chart with a vertical marker for current_value."""
    kde_x = dist.get("kde_x", [])
    kde_y = dist.get("kde_y", [])
    if not kde_x:
        return ""

    W, H = 200, 55
    x_min, x_max = min(kde_x), max(kde_x)
    y_max = max(kde_y)
    x_range = x_max - x_min or 1

    def sx(x):
        return max(0.0, min(W, (x - x_min) / x_range * W))

    def sy(y):
        return H - y / y_max * H * 0.88

    points = [(sx(x), sy(y)) for x, y in zip(kde_x, kde_y)]
    path = f"M {points[0][0]:.1f} {H} L {points[0][0]:.1f} {points[0][1]:.1f}"
    path += " ".join(f"L {px:.1f} {py:.1f}" for px, py in points[1:])
    path += f" L {points[-1][0]:.1f} {H} Z"

    vx = sx(current_value)
    return (
        f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:55px;display:block;margin-top:8px;">'
        f'<path d="{path}" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-opacity="0.5" stroke-width="1.2"/>'
        f'<line x1="{vx:.1f}" y1="0" x2="{vx:.1f}" y2="{H}" stroke="{color}" stroke-width="2"/>'
        f"</svg>"
    )


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _natural_sort_key(task):
    task_id = task["input"]["task_id"]
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", task_id)]


def _load_vram_bench() -> dict[str, list[dict]] | None:
    """Load the latest vram_bench*.json, return {model_name: [entry, ...]} or None."""
    vram_files = sorted(OUTPUT_BASE.glob("vram_bench*.json"))
    if not vram_files:
        return None
    data = json.loads(vram_files[-1].read_text())
    by_model: dict[str, list[dict]] = {}
    for r in data.get("results", []):
        by_model.setdefault(r["model"], []).append(r)
    return by_model


_VRAM_BENCH_CACHE: dict[str, list[dict]] | None = None
_MODEL_STATS_CACHE: dict | None = None


def _badges_for_model(model_name: str) -> str:
    """Return HTML badge string for a single model from model_stats + vram_bench."""
    global _VRAM_BENCH_CACHE, _MODEL_STATS_CACHE

    if _MODEL_STATS_CACHE is None:
        _MODEL_STATS_CACHE = load_model_stats() or {}
    if _VRAM_BENCH_CACHE is None:
        _VRAM_BENCH_CACHE = _load_vram_bench() or {}

    stats = _MODEL_STATS_CACHE
    vram = _VRAM_BENCH_CACHE

    # Resolve the model name through aliases; fall back to original name
    resolved = resolve_model_name(model_name, stats)
    entry = stats.get(resolved) if stats else None
    if entry is None and resolved != model_name:
        entry = stats.get(model_name)

    badges: list[str] = []

    # Parameter size
    if entry and entry.get("parameter_size"):
        badges.append(f'<span class="info-badge" title="Parameter size">{html.escape(entry["parameter_size"])}</span>')

    # Quantization
    if entry and entry.get("quantization_level"):
        badges.append(f'<span class="info-badge" title="Quantization">{html.escape(entry["quantization_level"])}</span>')

    # Capabilities
    if entry:
        caps = entry.get("capabilities") or []
        if caps:
            badges.append(f'<span class="info-badge info-badge--caps" title="Capabilities">{html.escape(", ".join(caps))}</span>')

    # Abbreviated digest
    if entry and entry.get("digest"):
        digest = entry["digest"]
        short = digest[:8] + "…" + digest[-4:]
        badges.append(f'<span class="info-badge info-badge--mono" title="Digest: {html.escape(digest)}">sha256:{html.escape(short)}</span>')

    # VRAM and speed from vram bench
    vram_entries = vram.get(model_name, [])
    if vram_entries:
        # Use the smallest context size entry for representative VRAM
        min_ctx = min(vram_entries, key=lambda e: e.get("ctx", 999999))
        vram_mb = min_ctx.get("ollama_size_vram_mb")
        if vram_mb:
            badges.append(f'<span class="info-badge" title="Ollama-reported VRAM at ctx={min_ctx["ctx"]}">VRAM: {vram_mb / 1024:.2f} GB</span>')

        # Average prefill and decode across all context sizes
        prefill_vals = [e["prefill_tokens_per_sec"] for e in vram_entries if e.get("prefill_tokens_per_sec")]
        decode_vals = [e["decode_tokens_per_sec"] for e in vram_entries if e.get("decode_tokens_per_sec")]
        if prefill_vals:
            avg_pre = sum(prefill_vals) / len(prefill_vals)
            badges.append(f'<span class="info-badge" title="Avg prefill speed">Prefill: {avg_pre:.0f} tok/s</span>')
        if decode_vals:
            avg_dec = sum(decode_vals) / len(decode_vals)
            badges.append(f'<span class="info-badge" title="Avg decode speed">Decode: {avg_dec:.1f} tok/s</span>')

    return " ".join(badges)


def _cascade_models_from_jsonl(jsonl_file: str | Path) -> list[str]:
    """Read the JSONL file and return unique model names from iterations, in first-seen order."""
    seen: dict[str, None] = {}
    ordered: list[str] = []
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            task = json.loads(line)
            for it in task.get("iterations", []):
                m = it.get("model")
                if m and m not in seen:
                    seen[m] = None
                    ordered.append(m)
    return ordered


def _model_info_header(summary_data: dict, jsonl_file: str | Path) -> str:
    """Return an HTML summary header for a model.

    For cascade/champion runs, shows one line per cascade model.
    For regular runs, shows a single line.
    """
    _tag = summary_data.get("tag") or ""
    is_cascade = _tag.startswith("cascade") or _tag.startswith("champion")

    if is_cascade:
        models = _cascade_models_from_jsonl(jsonl_file)
        if not models:
            return ""
        lines: list[str] = []
        for i, m in enumerate(models):
            badges = _badges_for_model(m)
            if not badges:
                badge_html = f'<span class="info-badge">no data for {html.escape(m)}</span>'
            else:
                badge_html = badges
            tier_label = f"Tier {i + 1}: {html.escape(m)}" if len(models) > 1 else ""
            label_html = f'<span class="info-model-name">{tier_label}</span>' if tier_label else ""
            lines.append(f'<div class="model-info">{label_html} {badge_html}</div>')
        return "\n".join(lines)
    else:
        badges = _badges_for_model(summary_data["model"])
        if not badges:
            return ""
        return '<div class="model-info">' + badges + "</div>"


def generate_task_html(task):
    raw_task_id = task["input"]["task_id"]
    task_id = html.escape(raw_task_id)
    task_link = f'<a href="task_{sanitize_task_id(raw_task_id)}.html">{task_id}</a>'
    generated_prompt = html.escape(task["generated_prompt"])
    # New flow: original_input_prompt in input dict; old flow: rewritten_prompt at top level
    rewritten_prompt = (
        task["input"]["prompt"] if task["input"].get("original_input_prompt")
        else task.get("rewritten_prompt")
    )
    input_canonical_solution = html.escape(task["input"]["canonical_solution"])
    input_test = html.escape(task["input"]["test"])
    iterations = task["iterations"]
    final_result = task["final_result"]
    total_time = (
        sum(
            iteration.get("total_duration", 0)
            for iteration in iterations
            if iteration.get("total_duration", 0) is not None
        )
        / 1000000000.0
    )
    overall_result = (
        "<span style='color: green;'>Pass</span>"
        if final_result["exit_code"] == 0
        else "<span style='color: red;'>Fail</span>"
    )
    summary_rows = ""
    for i, iteration in enumerate(iterations):
        iteration_time = ns_to_seconds(iteration.get("total_duration"))
        test_res = iteration.get("test_result")
        if test_res is None:
            iteration_result = "N/A"
        elif test_res["exit_code"] == 0:
            iteration_result = "Pass"
        else:
            iteration_result = "Fail"
        iter_model = html.escape(iteration.get("model", ""))
        summary_rows += f"<tr><td>{i + 1}</td><td>{iteration_result}</td><td>{iteration_time}</td><td>{iter_model}</td></tr>"

    iteration_details = ""
    for i, iteration in enumerate(iterations):
        iteration_time = ns_to_seconds(iteration.get("total_duration"))
        prompt = expandable_code_html(iteration.get("prompt", ""))
        msg = iteration.get("message", {})
        code = expandable_code_html(msg.get("content", ""))
        thinking = msg.get("thinking")
        test = expandable_code_html(iteration.get("test", ""))
        raw_test_result = iteration.get("test_result")
        if raw_test_result is None:
            test_result_html = "<p>N/A</p>"
        elif raw_test_result.get("exit_code") == 0:
            test_result_html = "<p style='color: green; font-weight: bold;'>Pass</p>"
        else:
            stderr = expandable_code_html(raw_test_result.get("stderr", ""))
            test_result_html = f"<pre>{stderr}</pre>"

        iter_anchor = anchor_id(f"{raw_task_id}_{i + 1}")
        iteration_details += f"""
        <h4 id="{iter_anchor}">Iteration {i + 1} ({iteration_time} s)</h4>
        <details id="{iter_anchor}_prompt">
          <summary>Prompt</summary>
          <pre>{prompt}</pre>
        </details>
        """
        if thinking:
            thinking = expandable_code_html(thinking)
            iteration_details += f"""
        <details id="{iter_anchor}_thinking">
          <summary>Thinking</summary>
          <pre>{thinking}</pre>
        </details>
        """
        iteration_details += f"""
        <details id="{iter_anchor}_code">
          <summary>Generated Code</summary>
          <pre>{code}</pre>
        </details>
        <details id="{iter_anchor}_test">
          <summary>Generated Test</summary>
          <pre>{test}</pre>
        </details>
        {test_result_html}
        """

    rewrite_html = ""
    if rewritten_prompt:
        escaped_rewrite = html.escape(rewritten_prompt)
        dataset_name = html.escape(task.get("dataset", ""))
        rewrite_html = f"""
        <details>
          <summary>Rewritten Prompt{f' ({dataset_name})' if dataset_name else ''}</summary>
          <pre><code>{escaped_rewrite}</code></pre>
        </details>"""
    rewrite_badge = (
        " <span style='color: #8e44ad; font-size: 0.8em;'>[rewritten]</span>"
        if rewritten_prompt
        else ""
    )
    return f"""
    <details id="{anchor_id(raw_task_id)}">
      <summary>{task_link}{rewrite_badge} {overall_result} at {len(iterations)} iterations in {total_time:0.2f} seconds.</summary>
      <table>
        <tr><th>Iteration</th><th>Result</th><th>Time (s)</th><th>Model</th></tr>
        {summary_rows}
      </table>
      <div>
        <details>
          <summary>Input Prompt</summary>
          <pre><code class="language-python">{generated_prompt}</code></pre>
        </details>
        {rewrite_html}
        <details>
          <summary>Input Test</summary>
          <pre><code class="language-python">{input_test}</code></pre>
        </details>
        <details>
          <summary>Input Canonical Solution</summary>
          <pre><code class="language-python">{input_canonical_solution}</code></pre>
        </details>
        {iteration_details}
      </div>
    </details>
    """


def generate_individual_html(jsonl_file, summary_data, no_cache=False, distributions=None):
    html_file_name = Path(jsonl_file).with_suffix(".html").name
    if not no_cache:
        html_path = OUTPUT_DIR / html_file_name
        if html_path.exists() and html_path.stat().st_mtime > Path(jsonl_file).stat().st_mtime:
            return html_file_name

    tasks = summary_data["tasks"]
    model = tag_display_name(summary_data)
    think = summary_data["think"]
    dataset = summary_data.get("dataset", "human-eval-enhanced-202307")
    alt_dataset = dataset if dataset != "human-eval-enhanced-202307" else None

    durations = []
    lengths: list[int] = []
    for task in tasks:
        for iteration in task.get("iterations", []):
            duration = iteration.get("total_duration", 0)
            length = iteration.get("eval_count", 0)
            if duration is not None and duration > 0:
                durations.append(duration / 1_000_000_000.0)
            if isinstance(length, int) and length > 0:
                lengths.append(length)
    times_kde_html = ""
    lengths_kde_html = ""
    if durations:
        estimator = KDEEstimator(durations)
        estimator.filter_outliers()
        times_kde_html = _full_kde_chart(
            estimator.filtered_data, estimator.raw_median,
            min(durations), max(durations), "#d97706", unit="s",
        )
    if lengths:
        estimator2 = KDEEstimator(lengths)
        estimator2.filter_outliers()
        lengths_kde_html = _full_kde_chart(
            estimator2.filtered_data, estimator2.raw_median,
            min(lengths), max(lengths), "#2563eb", unit=" tok",
        )

    error_counts = summary_data.get("error_counts", {})
    filtered_errors = {k: v for k, v in error_counts.items() if v > 0}
    error_chart_html = ""
    if filtered_errors:
        sorted_errors = sorted(filtered_errors.items(), key=lambda item: item[1], reverse=True)
        labels = [item[0] for item in sorted_errors]
        values = [item[1] for item in sorted_errors]
        chart_height = max(180, len(labels) * 28 + 40)
        error_chart_html = f"""
        <div style="background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.12);padding:1rem 1.5rem;margin:1rem 0;">
          <div class="stat-label" style="margin-bottom:0.5rem;">Error Breakdown</div>
          <div id="errorChart" style="width:100%;max-width:760px;height:{chart_height}px;"></div>
        </div>
        <script>
            Plotly.newPlot('errorChart', [{{
                x: {json.dumps(values)},
                y: {json.dumps(labels)},
                type: 'bar',
                orientation: 'h',
                marker: {{ color: '#2563eb', opacity: 0.7 }},
                text: {json.dumps(values)}.map(String),
                textposition: 'outside',
                cliponaxis: false
            }}], {{
                margin: {{ l: 180, r: 40, t: 4, b: 30 }},
                yaxis: {{ autorange: 'reversed', tickfont: {{ size: 12 }}, gridcolor: '#f1f5f9' }},
                xaxis: {{ gridcolor: '#f1f5f9', zeroline: false }},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)',
                showlegend: false
            }}, {{displayModeBar: false}});
        </script>
        """

    pass_pct = summary_data["passed_tests"] / summary_data["total_tests"] * 100
    lta = compute_yield(tasks)
    avg_time = summary_data["average_time_per_iteration"]
    spm = summary_data.get("success_per_minute", 0)

    def _chart(key, value, color):
        if not distributions or key not in distributions:
            return ""
        return _mini_kde_chart(distributions[key], value, color)

    stat_cards_html = f"""<div class="stat-cards">
  <div class="stat-card accent-blue">
    <div class="stat-label">Pass Rate</div>
    <div class="stat-value">{pass_pct:.1f}%</div>
    <div class="stat-sub">{summary_data["passed_tests"]} / {summary_data["total_tests"]} tasks</div>
    {_chart("pass_rate", pass_pct, "#2563eb")}
  </div>
  <div class="stat-card accent-amber">
    <div class="stat-label">Avg Time / Iter</div>
    <div class="stat-value">{avg_time:.1f}s</div>
    <div class="stat-sub">total {_fmt_duration(summary_data["total_time"])}</div>
    {_chart("avg_time", avg_time, "#d97706")}
  </div>
  <div class="stat-card accent-green">
    <div class="stat-label">Pass / Min</div>
    <div class="stat-value">{spm:.3f}</div>
    <div class="stat-sub">{summary_data.get("success_per_1k_tokens", 0):.3f} / 1K tok</div>
    {_chart("pass_per_min", spm, "#16a34a")}
  </div>
  <div class="stat-card accent-green">
    <div class="stat-label">Yield Score T=10</div>
    <div class="stat-value">{lta * 100:.1f}%</div>
    {_chart("yield", lta * 100, "#16a34a")}
  </div>
</div>"""

    html_file = OUTPUT_DIR / html_file_name
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html>
<head>
  {HEAD_META}
  <title>Model: {model}, Think: {think}{f", Dataset: {alt_dataset}" if alt_dataset else ""}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  {HLJS_BLOCK}
  {CSS_BLOCK}
  {EXPAND_SCRIPT}
</head>
<body>
{_nav_html("model")}
<div class="page-content">
  <h1>Model: {html.escape(model)}, Think: {think}</h1>
  {_model_info_header(summary_data, jsonl_file)}
  {f'<h2 style="color: #8e44ad;">Dataset: {html.escape(alt_dataset)}</h2>' if alt_dataset else ""}
  {stat_cards_html}
""")
        f.write(error_chart_html)
        if times_kde_html or lengths_kde_html:
            f.write('<div style="display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0;">')
            if times_kde_html:
                f.write(
                    f'<div style="flex:1;min-width:280px;background:#fff;border-radius:8px;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,.12);padding:1rem 1.5rem;">'
                    f'<div class="stat-label">Response Time Distribution</div>'
                    f'{times_kde_html}</div>'
                )
            if lengths_kde_html:
                f.write(
                    f'<div style="flex:1;min-width:280px;background:#fff;border-radius:8px;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,.12);padding:1rem 1.5rem;">'
                    f'<div class="stat-label">Token Count Distribution</div>'
                    f'{lengths_kde_html}</div>'
                )
            f.write('</div>')
        for task in sorted(tasks, key=_natural_sort_key):
            f.write(generate_task_html(task))
        f.write("\n</div>\n</body>\n</html>\n")
    return html_file_name
