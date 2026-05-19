"""Cascade report page: visual flow diagram and tier-level statistics."""

import html
from statistics import median

from ollama_codeeval.report._shared import (
    CSS_BLOCK,
    HEAD_META,
    OUTPUT_DIR,
    SORT_SCRIPT,
    _nav_html,
    tag_display_name,
)


def _display_name(d):
    return tag_display_name(d)


def _analyse_cascade(all_data: list) -> dict:
    """Analyse champion-tagged data to produce cascade-specific statistics.

    Returns a dict with:
      - champion_data: list of champion summary dicts
      - tiers: list of model names in order from the first champion run
      - per_tier: {model: {attempts, successes, total_time_s}}
      - tasks: list of {task_id, total_iters, solving_tier, solving_iter, tier_sequence}
    """
    champion_data = [
        d for d in all_data
        if d.get("dataset", "humaneval") == "humaneval"
        and ((d.get("tag") or "").startswith("cascade") or (d.get("tag") or "").startswith("champion"))
    ]
    if not champion_data:
        return {"champion_data": [], "tiers": [], "per_tier": {}, "tasks": []}

    # Pick the first champion entry to derive tiers from iteration model labels
    first = champion_data[0]
    tier_names = []
    seen_tiers = set()
    for task in first["tasks"]:
        for it in task.get("iterations", []):
            m = it.get("model", "")
            if m and m not in seen_tiers:
                seen_tiers.add(m)
                tier_names.append(m)

    per_tier: dict = {}
    for name in tier_names:
        per_tier[name] = {"attempts": 0, "successes": 0, "durations": []}

    tasks_analysis = []
    for d in champion_data:
        for task in d["tasks"]:
            tid = task["input"]["task_id"]
            iterations = task.get("iterations", [])
            total_iters = len(iterations)
            passed = task["final_result"]["exit_code"] == 0

            solving_tier = None
            solving_iter = None
            tier_sequence = []

            for i, it in enumerate(iterations):
                m = it.get("model", "?")
                tr = it.get("test_result", {})
                duration = it.get("total_duration") or 0

                if m not in tier_sequence:
                    tier_sequence.append(m)

                if m in per_tier:
                    per_tier[m]["attempts"] += 1
                    if duration > 0:
                        per_tier[m]["durations"].append(duration)

                if tr.get("exit_code") == 0:
                    if m in per_tier:
                        per_tier[m]["successes"] += 1
                    if solving_tier is None:
                        solving_tier = m
                        solving_iter = i + 1  # 1-indexed

            tasks_analysis.append({
                "task_id": tid,
                "total_iters": total_iters,
                "passed": passed,
                "solving_tier": solving_tier or "\u2014",
                "solving_iter": solving_iter or 0 if passed else "\u2014",
                "tier_sequence": " \u2192 ".join(tier_sequence),
            })

    tasks_analysis.sort(key=lambda t: (not t["passed"], t["total_iters"]))

    for d in champion_data:
        durations = [
            it.get("total_duration") or 0
            for task in d["tasks"]
            for it in task.get("iterations", [])
        ]
        valid = [ns for ns in durations if ns > 0]
        d["_median_time_per_iter"] = median(valid) / 1e9 if valid else 0

    return {
        "champion_data": champion_data,
        "tiers": tier_names,
        "per_tier": per_tier,
        "tasks": tasks_analysis,
    }


def generate_cascade_html(all_data: list) -> None:
    """Write the cascade analysis page."""
    cascade = _analyse_cascade(all_data)
    champion_data = cascade["champion_data"]
    tiers = cascade["tiers"]
    per_tier = cascade["per_tier"]
    tasks_analysis = cascade["tasks"]

    # ── config section ──
    from ollama_codeeval.cascade_agent import DEFAULT_CASCADE
    config_rows = ""
    for i, (model, max_iters) in enumerate(DEFAULT_CASCADE, start=1):
        config_rows += (
            f"<tr><td><strong>Tier {i}</strong></td>"
            f"<td>{html.escape(model)}</td>"
            f"<td>{max_iters}</td></tr>\n"
        )

    # ── tier stats table ──
    tier_rows = ""
    for i, name in enumerate(tiers, start=1):
        s = per_tier[name]
        att = s["attempts"]
        suc = s["successes"]
        pct = suc / att * 100 if att > 0 else 0
        med_time = median(s["durations"]) / 1e9 if s["durations"] else 0
        tier_rows += (
            f"<tr><td>Tier {i} \u2014 {html.escape(name)}</td>"
            f'<td data-sort-value="{att}">{att}</td>'
            f'<td data-sort-value="{suc}">{suc}</td>'
            f'<td data-sort-value="{pct:.2f}">{pct:.1f}%</td>'
            f'<td data-sort-value="{med_time:.3f}">{med_time:.3f}</td>'
            f"</tr>\n"
        )

    # ── task drill-down ──
    task_rows = ""
    for t in tasks_analysis:
        result_cls = "pass" if t["passed"] else "fail"
        result_label = "Pass" if t["passed"] else "Fail"
        si = t["solving_iter"]
        si_display = str(si) if isinstance(si, int) else si
        task_rows += (
            f'<tr class="{result_cls}">'
            f"<td>{html.escape(t['task_id'])}</td>"
            f'<td data-sort-value="{1 if t["passed"] else 0}"><span class="{result_cls}">{result_label}</span></td>'
            f'<td data-sort-value="{t["total_iters"]}">{t["total_iters"]}</td>'
            f"<td>{html.escape(t['solving_tier'])}</td>"
            f'<td data-sort-value="{si_display}">{si_display}</td>'
            f"<td>{html.escape(t['tier_sequence'])}</td>"
            f"</tr>\n"
        )

    # ── overview stats ──
    champion_overview = ""
    for d in champion_data:
        name = _display_name(d)
        pct = d["passed_tests"] / d["total_tests"] * 100
        champion_overview += (
            f'<tr><td><a href="{d["html_file"]}">{html.escape(name)}</a></td>'
            f'<td data-sort-value="{d["passed_tests"]}">{d["passed_tests"]}/{d["total_tests"]} ({pct:.1f}%)</td>'
            f'<td data-sort-value="{d["_median_time_per_iter"]:.3f}">{d["_median_time_per_iter"]:.3f}</td>'
            f'<td data-sort-value="{d.get("success_per_minute", 0):.3f}">{d.get("success_per_minute", 0):.3f}</td>'
            f'<td data-sort-value="{d.get("lta", 0):.4f}">{d.get("lta", 0) * 100:.1f}%</td>'
            f"</tr>\n"
        )

    # ── flow diagram (SVG) ──
    flow_svg = """<svg viewBox="0 0 800 460" class="cascade-diagram" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
    </marker>
    <marker id="arrowhead-green" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#16a34a"/>
    </marker>
    <marker id="arrowhead-red" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#dc2626"/>
    </marker>
  </defs>
  <!-- Nodes -->
  <rect x="330" y="10" width="140" height="40" rx="6" fill="#bfdbfe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="400" y="35" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#1e3a5f">Format Prompt</text>

  <rect x="330" y="85" width="140" height="40" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="400" y="110" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#1e3a5f">Generate</text>

  <rect x="330" y="155" width="140" height="40" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="400" y="175" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#78350f">Ruff Fix</text>
  <text x="400" y="190" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#92400e">format + check --fix</text>

  <rect x="330" y="225" width="140" height="40" rx="6" fill="#d1fae5" stroke="#10b981" stroke-width="1.5"/>
  <text x="400" y="250" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#064e3b">Execute (Docker)</text>

  <rect x="530" y="225" width="140" height="40" rx="6" fill="#fce7f3" stroke="#ec4899" stroke-width="1.5"/>
  <text x="600" y="247" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#831843">Fix (error feedback)</text>
  <text x="600" y="260" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#9d174d">temperature escalation</text>

  <rect x="530" y="305" width="140" height="40" rx="6" fill="#ede9fe" stroke="#8b5cf6" stroke-width="1.5"/>
  <text x="600" y="330" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#4c1d95">Escalate</text>

  <rect x="330" y="385" width="140" height="40" rx="6" fill="#fee2e2" stroke="#ef4444" stroke-width="1.5"/>
  <text x="400" y="410" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#7f1d1d">Respond (done)</text>

  <!-- Tier boxes on left -->
  <rect x="15" y="85" width="120" height="34" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1"/>
  <text x="75" y="106" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#312e81">Tier 1: qwen3:4b</text>

  <rect x="15" y="130" width="120" height="34" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1"/>
  <text x="75" y="151" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#312e81">Tier 2: qwen2.5-coder</text>

  <rect x="15" y="176" width="120" height="34" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1"/>
  <text x="75" y="197" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#312e81">Tier 3: gemma4:26b</text>

  <!-- Arrows -->
  <line x1="400" y1="50" x2="400" y2="85" stroke="#555" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="125" x2="400" y2="155" stroke="#555" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <line x1="400" y1="195" x2="400" y2="225" stroke="#555" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Execute → Fix (error) -->
  <line x1="470" y1="245" x2="530" y2="245" stroke="#ea580c" stroke-width="1.5" marker-end="url(#arrowhead-red)"/>
  <text x="500" y="240" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#ea580c">error</text>

  <!-- Fix → Escalate (stuck) -->
  <path d="M 600 265 L 600 305" fill="none" stroke="#ea580c" stroke-width="1.5" marker-end="url(#arrowhead-red)"/>
  <text x="610" y="290" font-family="sans-serif" font-size="9" fill="#ea580c">stuck</text>

  <!-- Fix → Ruff Fix (default) -->
  <path d="M 600 225 L 600 175 L 470 175" fill="none" stroke="#555" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <text x="535" y="170" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#555">default</text>

  <!-- Escalate → Generate (right side, around edge) -->
  <path d="M 670 325 L 770 325 L 770 105 L 470 105" fill="none" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#arrowhead)"/>
  <text x="735" y="215" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#8b5cf6">next tier</text>

  <!-- Execute → Respond (ok) -->
  <line x1="380" y1="265" x2="380" y2="385" stroke="#16a34a" stroke-width="2" marker-end="url(#arrowhead-green)"/>
  <text x="370" y="330" text-anchor="end" font-family="sans-serif" font-size="9" fill="#16a34a">ok</text>

  <!-- Fail paths to Respond -->
  <line x1="420" y1="265" x2="420" y2="385" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrowhead-red)"/>
  <text x="430" y="330" font-family="sans-serif" font-size="8" fill="#dc2626">fail</text>

  <path d="M 600 345 L 600 405 L 470 405" fill="none" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrowhead-red)"/>
  <text x="535" y="403" text-anchor="middle" font-family="sans-serif" font-size="8" fill="#dc2626">no tiers left</text>

  <!-- Generate fail → Respond -->
  <path d="M 330 105 L 260 105 L 260 405 L 330 405" fill="none" stroke="#dc2626" stroke-width="1.2" stroke-dasharray="3,2" marker-end="url(#arrowhead-red)"/>
  <text x="247" y="250" font-family="sans-serif" font-size="8" fill="#dc2626">fail</text>
</svg>"""

    no_champ = not champion_data

    page = f"""<!doctype html>
<html>
<head>{HEAD_META}<title>Humaneval LLM Benchmark — Cascade</title>
{CSS_BLOCK}{SORT_SCRIPT}</head>
<body>
{_nav_html("cascade")}
<div class="page-content">
  <h1>Cascade</h1>
  <p>
    The cascade routes each problem through a sequence of models,
    escalating to the next tier only when the current one exhausts its budget.
    Each tier has a fixed number of fix+retry attempts before escalation.
  </p>

  <p>The insight is: if you have tests, and fast models: run a fast model first. If it fails, escalate to a larger model
    if you expect better results from a new viewpoint, than from continued iteration with a fast model.
  </p>

  <p>Note: this is highly coupled to this dataset and succeeds through the joy of hindsight. It *may* generalize, but there are no guarantees.</p>

  <h2>Harness</h2>
  {flow_svg}

  <h2>Cascade Configuration</h2>
  <p>This cascade was optimized for 28Gb of VRAM, so the third model chosen is the one that can get the highest % success within the availble VRAM. In this case gemma4:26b using 18.7 GB of VRAM. Ollama will swap these in and out of VRAM, so ~19Gb of VRAM is enough, at the cost of swapping.</p>
  <p>The first two models only consume ~4Gb and ~5Gb. You could skip or swap, accoring to your VRAM.
  <table class="cascade-config">
    <thead><tr><th>Tier</th><th>Model</th><th>Max Attempts</th></tr></thead>
    <tbody>{config_rows}</tbody>
  </table>

  <h2>Cascade Runs</h2>
  {"<p class='note'>No cascade-tagged data found in the evaluation results. Run with <code>--cascade</code> to populate this page.</p>" if no_champ else ""}
  {"" if no_champ else f'''
  <table>
    <thead><tr>
      <th class="sortable">Run</th>
      <th class="sortable">Passed</th>
      <th class="sortable">Avg Time/IT (s)</th>
      <th class="sortable">Success/m</th>
      <th class="sortable">Yield</th>
    </tr></thead>
    <tbody>{champion_overview}</tbody>
  </table>
  '''}

  <h2>Per-Tier Breakdown</h2>
  <p>Fail fast, then escalate.</p>
  {"<p class='note'>No cascade data to display.</p>" if not tiers else f"""
  <table>
    <thead><tr>
      <th class="sortable">Tier</th>
      <th class="sortable">Total Attempts</th>
      <th class="sortable">Successes</th>
      <th class="sortable">Success Rate</th>
      <th class="sortable">Avg Time/IT (s)</th>
    </tr></thead>
    <tbody>{tier_rows}</tbody>
  </table>
  """}

  <h2>Task Details</h2>
  <p>
    Each task's journey through the cascade.
  </p>
  {"<p class='note'>No tasks to display.</p>" if not tasks_analysis else f"""
  <table>
    <thead><tr>
      <th class="sortable">Task ID</th>
      <th class="sortable">Result</th>
      <th class="sortable">Total Iterations</th>
      <th class="sortable">Solving Tier</th>
      <th class="sortable">Solving Iter</th>
      <th>Tier Sequence</th>
    </tr></thead>
    <tbody>{task_rows}</tbody>
  </table>
  """}

</div>
</body>
</html>
"""

    with open(OUTPUT_DIR / "cascade.html", "w", encoding="utf-8") as f:
        f.write(page)
