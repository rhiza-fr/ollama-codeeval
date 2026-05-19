"""Yield Explorer page — interactive τ (half-credit time) sensitivity chart.



┌─────────────────────────────────────┬──────────┬──────────────────────────────────────────────────────┐
│              Use case               │    τ     │                      Reasoning                       │
├─────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤
│ Inline autocomplete / Copilot-style │ 1-3 s    │ User pauses ~1s; anything beyond 3s breaks flow      │
├─────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤
│ Interactive REPL / chat iteration   │ 10-30 s  │ A 30s wait is annoying but acceptable once per query │
├─────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤
│ CLI tool / pre-commit hook          │ 30-120 s │ You switched context; 1-2 min is tolerable           │
├─────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤
│ CI pipeline step                    │ 2-10 min │ Batched anyway; you're not watching                  │
├─────────────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤
│ Overnight batch job                 │ τ → ∞    │ Speed irrelevant; yield → raw accuracy               │
└─────────────────────────────────────┴──────────┴──────────────────────────────────────────────────────┘

"""

import json
import math

import numpy as np

from ollama_codeeval.report._shared import (
    CANONICAL_TAU,
    CSS_BLOCK,
    HEAD_META,
    OUTPUT_DIR,
    SORT_SCRIPT,
    _nav_html,
    tag_display_name,
)

_DEFAULT_TAU = CANONICAL_TAU  # keep slider default in sync with reports
_TAU_MIN = 0.5
_TAU_MAX = 600.0
_TAU_STEPS = 50


def _model_yield(tasks: list, tau: float) -> float:
    """Hyperbolic-discounted accuracy: mean of 1/(1 + t/τ) over all tasks (0-100)."""
    N = len(tasks)
    if N == 0:
        return 0.0
    total = 0.0
    for t in tasks:
        elapsed = (
            sum(it.get("total_duration") or 0 for it in t.get("iterations", [])) / 1e9
        )
        if t["final_result"]["exit_code"] == 0:
            total += 1.0 / (1.0 + elapsed / tau)
    return total / N * 100


def _tau_label(tau: float) -> str:
    if tau < 60:
        return f"{tau:.1f}s"
    return f"{tau / 60:.1f}m"


def generate_yield_html(all_data: list) -> None:
    models = [d for d in all_data if d.get("dataset", "humaneval") == "humaneval"]
    labels = [
        tag_display_name(d) + (" (think)" if d.get("think") else "") for d in models
    ]

    tau_values: list[float] = np.logspace(
        math.log10(_TAU_MIN), math.log10(_TAU_MAX), _TAU_STEPS
    ).tolist()

    # yields_matrix[tau_idx][model_idx]
    yields_matrix = [
        [_model_yield(m["tasks"], tau) for m in models] for tau in tau_values
    ]

    # Fix model order by yield at default τ, descending
    default_idx = min(
        range(len(tau_values)), key=lambda i: abs(tau_values[i] - _DEFAULT_TAU)
    )
    order = sorted(
        range(len(models)), key=lambda i: yields_matrix[default_idx][i], reverse=True
    )
    sorted_labels = [labels[i] for i in order]
    sorted_matrix = [[row[i] for i in order] for row in yields_matrix]

    # Raw pass rate (%) per model, in sorted order
    def _accuracy(m: dict) -> float:
        tasks = m["tasks"]
        if not tasks:
            return 0.0
        return (
            sum(1 for t in tasks if t["final_result"]["exit_code"] == 0)
            / len(tasks)
            * 100
        )

    sorted_accuracy = [_accuracy(models[i]) for i in order]

    def _fmt_time(s: float | None) -> str:
        if not s:
            return "N/A"
        if s < 60:
            return f"{s:.1f}s"
        return f"{s / 60:.1f}m"

    sorted_median_time = [models[i].get("average_time_per_iteration") for i in order]

    # Bar chart slider steps (restyle y + text for trace 0)
    slider_steps = [
        {
            "method": "restyle",
            "label": _tau_label(tau),
            "args": [
                {
                    "y": [sorted_matrix[t_idx]],
                    "text": [[f"{v:.1f}%" for v in sorted_matrix[t_idx]]],
                },
                [0],
            ],
        }
        for t_idx, tau in enumerate(tau_values)
    ]

    bar_y_init = sorted_matrix[default_idx]
    bar_text_init = [f"{v:.1f}%" for v in bar_y_init]

    # Leaderboard tables at fixed tau values
    _LEADERBOARD_TAUS = [3.0, _DEFAULT_TAU, 30.0, 240.0]

    def _leaderboard_table(tau_target: float) -> str:
        t_idx = min(
            range(len(tau_values)), key=lambda i: abs(tau_values[i] - tau_target)
        )
        rows = sorted(
            zip(
                sorted_labels, sorted_matrix[t_idx], sorted_accuracy, sorted_median_time
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        label = _tau_label(tau_target)
        rows_html = "\n".join(
            f"<tr><td>{rank}</td><td>{name}</td><td>{yield_:.1f}%</td><td>{acc:.1f}%</td><td>{_fmt_time(med)}</td></tr>"
            for rank, (name, yield_, acc, med) in enumerate(rows, 1)
        )
        return (
            f"<div><h2>Leaderboard at τ = {label}</h2>"
            "<table><thead><tr><th>#</th><th>Model</th><th>Yield</th><th>Pass rate</th><th>Median time/iter</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table></div>"
        )

    tables = "\n".join(_leaderboard_table(t) for t in _LEADERBOARD_TAUS)
    leaderboard_html = f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem 2rem">{tables}</div>'

    with open(OUTPUT_DIR / "yield.html", "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html>
<head>{HEAD_META}<title>Yield Explorer</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
{CSS_BLOCK}{SORT_SCRIPT}</head>
<body>
{_nav_html("yield")}
<div class="page-content">
  <h1>Yield</h1>
  <p>
    <b>Yield</b> is speed-discounted accuracy. Each correct answer at time <i>t</i> contributes
    <code>1 / (1 + t/τ)</code> to the score, averaged over all tasks.
    The parameter <b>τ (tau)</b> is the <em>half-credit time</em>: the latency at which a correct
    answer counts for half of what an instant answer would.
  </p>
  <ul>
    <li><b>Low τ</b> (e.g. 2 s) — interactive use; speed matters a lot</li>
    <li><b>High τ</b> (e.g. 5 m) — batch use; accuracy dominates</li>
  </ul>

  <h2>Model ranking at selected τ</h2>
  <p>Drag the slider to choose τ and compare models side by side.</p>
  <div id="yield-bar" style="width:100%;height:560px"></div>

  {leaderboard_html}
</div>

<script>
(function() {{
  var barTrace = {{
    x: {json.dumps(sorted_labels)},
    y: {json.dumps(bar_y_init)},
    text: {json.dumps(bar_text_init)},
    textposition: 'outside',
    cliponaxis: false,
    type: 'bar',
    marker: {{ color: '#16a34a' }},
    hovertemplate: '<b>%{{x}}</b><br>Yield = %{{y:.1f}}%<extra></extra>',
  }};
  Plotly.newPlot('yield-bar', [barTrace], {{
    yaxis: {{ title: 'Yield (%)', range: [0, 108] }},
    xaxis: {{ tickangle: -40 }},
    margin: {{ t: 20, b: 280, l: 60 }},
    sliders: [{{
      active: {default_idx},
      currentvalue: {{ prefix: 'τ = ', font: {{ size: 14 }} }},
      pad: {{ t: 80 }},
      steps: {json.dumps(slider_steps)},
    }}],
  }}, {{responsive: true}});
}})();
</script>
</body>
</html>
""")
