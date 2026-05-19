"""Generate rewrite analysis page comparing model performance on rewritten vs baseline questions."""

import html

from ollama_codeeval.report._shared import CSS_BLOCK, OUTPUT_DIR, SORT_SCRIPT, _nav_html, compute_yield
from ollama_codeeval.report.html_violin import chart_combined_progression

_REWRITE_PREFIX = "humaneval-rewritten-"


def _pass_rate(d: dict) -> float:
    return d["passed_tests"] / d["total_tests"] * 100


def _extract_rewriter(dataset: str) -> str | None:
    if dataset.startswith(_REWRITE_PREFIX):
        return dataset[len(_REWRITE_PREFIX):]
    return None


def _delta_cell(rw_pct: float, base_pct: float | None) -> str:
    val = f"{rw_pct:.1f}%"
    if base_pct is None:
        return f'<td data-sort-value="{rw_pct:.2f}">{val}</td>'
    delta = rw_pct - base_pct
    sign = "+" if delta >= 0 else ""
    color = "#27ae60" if delta >= 0 else "#e74c3c"
    badge = f'<span style="color:{color};font-size:0.85em;margin-left:4px">({sign}{delta:.1f})</span>'
    return f'<td data-sort-value="{rw_pct:.2f}">{val}{badge}</td>'


def _build_matrix(baseline: dict, rewrite: dict, sorted_rewriters: list[str]) -> str:
    models = sorted(
        {model for (model, _) in rewrite},
        key=lambda m: -_pass_rate(baseline[m]) if m in baseline else 0,
    )

    header = (
        '<th class="sortable">Model</th>'
        '<th class="sortable">Baseline</th>'
        + "".join(f'<th class="sortable">{html.escape(rw)}</th>' for rw in sorted_rewriters)
    )

    rows = []
    for model in models:
        base_d = baseline.get(model)
        base_pct = _pass_rate(base_d) if base_d else None

        if base_pct is not None:
            base_cell = f'<td data-sort-value="{base_pct:.2f}">{base_pct:.1f}%</td>'
        else:
            base_cell = "<td>—</td>"

        rw_cells = "".join(
            _delta_cell(_pass_rate(rewrite[(model, rw)]), base_pct)
            if (model, rw) in rewrite else "<td>—</td>"
            for rw in sorted_rewriters
        )
        rows.append(f"<tr><td>{html.escape(model)}</td>{base_cell}{rw_cells}</tr>")

    return f"""<h2>Pass Rate by Rewriter</h2>
<p>Delta shown relative to baseline (HumanEval). Green = improvement, red = regression.</p>
<table>
  <thead><tr>{header}</tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>"""


def _build_run_table(rewrite_data: list[dict]) -> str:
    header = """<tr>
      <th class="sortable">Model</th>
      <th class="sortable">Rewriter</th>
      <th class="sortable">Passed</th>
      <th class="sortable">Avg Time/IT (s)</th>
      <th class="sortable">Yield</th>
      <th class="sortable">Pass@1</th>
      <th class="sortable">Pass@5</th>
    </tr>"""

    sorted_runs = sorted(rewrite_data, key=lambda d: -_pass_rate(d))
    rows = []
    for d in sorted_runs:
        rewriter = _extract_rewriter(d.get("dataset", "")) or "?"
        passed_pct = _pass_rate(d)
        lta = d.get("lta", 0) * 100
        cum = d["cum_pass_per_iteration"]
        pass1 = cum[0] if len(cum) > 0 else 0
        pass5 = cum[4] if len(cum) > 4 else 0
        rows.append(f"""<tr>
      <td><a href="{d["html_file"]}">{html.escape(d["model"])}</a></td>
      <td>{html.escape(rewriter)}</td>
      <td data-sort-value="{passed_pct:.2f}">{d["passed_tests"]} ({passed_pct:.1f}%)</td>
      <td data-sort-value="{d["average_time_per_iteration"]:.3f}">{d["average_time_per_iteration"]:.3f}</td>
      <td data-sort-value="{lta:.2f}">{lta:.1f}%</td>
      <td data-sort-value="{pass1:.2f}">{pass1:.2f}%</td>
      <td data-sort-value="{pass5:.2f}">{pass5:.2f}%</td>
    </tr>""")

    return f"""<h2>Rewrite Run Summary</h2>
<table>
  <thead>{header}</thead>
  <tbody>{"".join(rows)}</tbody>
</table>"""


def generate_rewrite_html(all_data: list) -> None:
    baseline: dict[str, dict] = {
        d["model"]: d
        for d in all_data
        if d.get("dataset", "humaneval") == "humaneval" and not d.get("tag")
    }

    rewrite: dict[tuple[str, str], dict] = {}
    for d in all_data:
        rewriter = _extract_rewriter(d.get("dataset", ""))
        if rewriter:
            rewrite[(d["model"], rewriter)] = d
            d["lta"] = compute_yield(d["tasks"])

    if not rewrite:
        return

    sorted_rewriters = sorted({rw for (_, rw) in rewrite})
    rewrite_data = list(rewrite.values())

    matrix_html = _build_matrix(baseline, rewrite, sorted_rewriters)
    run_table_html = _build_run_table(rewrite_data)
    progression_html = chart_combined_progression(all_data)

    n_models = len({m for (m, _) in rewrite})
    n_rewriters = len(sorted_rewriters)

    progression_section = ""
    if progression_html:
        progression_section = (
            "<h2>Full Progression: Attempt → Iteration → Rewrite</h2>\n"
            "<p>Each row shows how pass rate improves from first attempt, through iteration, to prompt rewriting.</p>\n"
            + progression_html
        )

    with open(OUTPUT_DIR / "rewrite.html", "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>LLM Benchmark — Rewrite Analysis</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
{CSS_BLOCK}{SORT_SCRIPT}</head>
<body>
{_nav_html("rewrite")}
<div class="page-content">
  <h1>Rewrite Analysis</h1>
  <p>{n_models} models evaluated on {n_rewriters} rewritten question sets.</p>
  {progression_section}
  {matrix_html}
  {run_table_html}
</div>
</body>
</html>
""")
