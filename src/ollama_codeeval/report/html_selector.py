"""Model Selector page — filter by VRAM and urgency (τ), ranked by yield."""

import json
import math
from pathlib import Path

from ollama_codeeval.config import OUTPUT_BASE
from ollama_codeeval.report._shared import CSS_BLOCK, HEAD_META, OUTPUT_DIR, _nav_html, load_model_stats, resolve_model_name, tag_display_name

_VRAM_BENCH_GLOB = "vram_bench_*.json"
_TAU_MIN = 0.5
_TAU_MAX = 600.0
_DEFAULT_TAU = 30.0


def _load_vram_data(model_stats: dict | None) -> dict[str, float]:
    """Return {resolved_model_name: total_vram_mb} using best non-error entry per model."""
    vram_files = sorted(OUTPUT_BASE.glob(_VRAM_BENCH_GLOB))
    if not vram_files:
        return {}
    with open(vram_files[-1]) as f:
        bench = json.load(f)

    by_model: dict[str, list] = {}
    for r in bench["results"]:
        by_model.setdefault(r["model"], []).append(r)

    result: dict[str, float] = {}
    for raw_name, entries in by_model.items():
        non_error = [e for e in entries if e.get("error") is None]
        pool = non_error if non_error else entries
        # Prefer ctx closest to 8192 among valid entries
        best = sorted(pool, key=lambda e: abs(e["ctx"] - 8192))[0]
        total_vram = sum(best["vram_delta_mb"].values())
        resolved = resolve_model_name(raw_name, model_stats)
        result[resolved] = total_vram
    return result


def _join_vram(model_name: str, vram_data: dict[str, float]) -> float | None:
    if model_name in vram_data:
        return vram_data[model_name]
    # Fallback: match on base name (before the colon)
    base = model_name.split(":")[0]
    candidates = [(k, v) for k, v in vram_data.items() if k.split(":")[0] == base]
    if len(candidates) == 1:
        return candidates[0][1]
    return None


def _task_solve_times(tasks: list) -> list[float]:
    """Per-task total elapsed seconds for successful tasks only."""
    times = []
    for t in tasks:
        elapsed = sum(it.get("total_duration") or 0 for it in t.get("iterations", [])) / 1e9
        if t["final_result"]["exit_code"] == 0:
            times.append(elapsed)
    return times


def generate_selector_html(all_data: list) -> None:
    model_stats = load_model_stats()
    vram_data = _load_vram_data(model_stats)

    # Build resolved_name -> (parameter_size, tools_support) from model_stats
    params_by_resolved: dict[str, str] = {}
    tools_by_resolved: dict[str, bool] = {}
    if model_stats:
        for raw, info in model_stats.items():
            resolved = resolve_model_name(raw, model_stats)
            if "parameter_size" in info:
                params_by_resolved[resolved] = info["parameter_size"]
            caps = info.get("capabilities") or []
            tools_by_resolved[resolved] = "tools" in caps

    rows = []
    for d in all_data:
        if d.get("dataset", "humaneval") != "humaneval":
            continue
        if (d.get("tag") or "").startswith(("cascade", "champion")):
            continue
        think = d.get("think", False)
        vram_mb = _join_vram(d["model"], vram_data)
        times = _task_solve_times(d["tasks"])
        total = d["total_tests"]
        pass_rate = d["passed_tests"] / total * 100
        display = tag_display_name(d) + (" (think)" if think else "")
        html_file = Path(d["file"]).with_suffix(".html").name
        rows.append({
            "name": display,
            "html_file": html_file,
            "params": params_by_resolved.get(d["model"]),
            "tools": tools_by_resolved.get(d["model"], False),
            "pass_rate": round(pass_rate, 2),
            "avg_time_s": round(d.get("average_time_per_iteration", 0), 3),
            "vram_mb": round(vram_mb) if vram_mb is not None else None,
            "times": [round(t, 3) for t in times],
            "total": total,
        })

    max_vram_gb = math.ceil(
        max((m["vram_mb"] / 1024 for m in rows if m["vram_mb"] is not None), default=28) / 2
    ) * 2
    default_tau_slider = round(
        100 * math.log(_DEFAULT_TAU / _TAU_MIN) / math.log(_TAU_MAX / _TAU_MIN)
    )

    payload = json.dumps(rows, separators=(",", ":"))

    with open(OUTPUT_DIR / "selector.html", "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html>
<html>
<head>
{HEAD_META}
<title>Model Selector — Humaneval LLM Benchmark</title>
{CSS_BLOCK}
<style>
.selector-controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 2rem;
  align-items: flex-start;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,.08);
  padding: 1.25rem 1.5rem;
  margin: 1rem 0 1.5rem;
}}
.control-group {{
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-width: 260px;
  flex: 1;
}}
.control-group label {{
  font-size: 0.82rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #64748b;
}}
.control-group label strong {{
  color: #0f172a;
  font-size: 1rem;
  text-transform: none;
  letter-spacing: 0;
}}
input[type=range] {{
  width: 100%;
  accent-color: #2563eb;
}}
.tau-presets {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}}
.tau-preset {{
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  border-radius: 5px;
  padding: 0.25rem 0.65rem;
  font-size: 0.8rem;
  cursor: pointer;
  color: #475569;
  transition: background 0.1s;
}}
.tau-preset:hover {{ background: #e2e8f0; }}
.tau-preset.active {{
  background: #2563eb;
  border-color: #2563eb;
  color: #fff;
}}
.selector-summary {{
  font-size: 0.9rem;
  color: #475569;
  margin-bottom: 0.75rem;
}}
.selector-summary strong {{ color: #0f172a; }}
.yield-cell {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 120px;
}}
.yield-bar-wrap {{
  flex: 1;
  background: #f1f5f9;
  border-radius: 3px;
  height: 10px;
  overflow: hidden;
}}
.yield-bar {{
  height: 100%;
  background: linear-gradient(90deg, #86efac, #16a34a);
  border-radius: 3px;
  transition: width 0.15s;
}}
.yield-val {{
  font-size: 0.85rem;
  font-weight: 600;
  color: #0f172a;
  min-width: 42px;
  text-align: right;
}}
.row-excluded td {{
  opacity: 0.35;
}}
.row-excluded .vram-badge {{
  color: #dc2626;
  font-weight: 600;
}}
.rank-cell {{
  color: #94a3b8;
  font-size: 0.8rem;
  width: 2rem;
  text-align: center;
}}
.rank-cell.top {{ color: #0f172a; font-weight: 700; }}
.toggle-excluded {{
  font-size: 0.82rem;
  color: #2563eb;
  cursor: pointer;
  background: none;
  border: none;
  padding: 0;
  text-decoration: underline;
  margin-left: 0.75rem;
}}
</style>
</head>
<body>
{_nav_html("selector")}
<div class="page-content">
  <h1>Model Selector</h1>
  <p>Adjust your available VRAM and urgency to find the best model for your setup.
  <b>Yield</b> is speed-discounted accuracy: a correct answer at time <i>t</i> scores
  <code>1/(1 + t/τ)</code>. At low τ, fast models win even if they miss more; at high τ,
  raw accuracy dominates.</p>

  <div class="selector-controls">
    <div class="control-group">
      <label for="vram-slider">Available VRAM &nbsp;<strong id="vram-label">16 GB</strong></label>
      <input type="range" id="vram-slider" min="2" max="{max_vram_gb}" step="1" value="16">
    </div>
    <div class="control-group">
      <label>Urgency (τ = half-credit time) &nbsp;<strong id="tau-label">{_DEFAULT_TAU:.0f}s</strong></label>
      <div class="tau-presets">
        <button class="tau-preset" data-tau="3">Inline (3s)</button>
        <button class="tau-preset active" data-tau="30">Interactive (30s)</button>
        <button class="tau-preset" data-tau="120">CLI / pre-commit (2m)</button>
        <button class="tau-preset" data-tau="600">Batch (10m)</button>
      </div>
      <input type="range" id="tau-slider" min="0" max="100" step="1" value="{default_tau_slider}">
    </div>
  </div>

  <p class="selector-summary" id="selector-summary"></p>

  <table>
    <thead><tr>
      <th>#</th>
      <th>Model</th>
      <th>#Params</th>
      <th>VRAM</th>
      <th>Pass Rate</th>
      <th>Avg Time / Iter</th>
      <th>Yield at τ</th>
    </tr></thead>
    <tbody id="selector-tbody"></tbody>
  </table>
</div>

<script>
(function() {{
  const MODELS = {payload};
  const TAU_MIN = {_TAU_MIN}, TAU_MAX = {_TAU_MAX};

  function sliderToTau(v) {{
    return TAU_MIN * Math.pow(TAU_MAX / TAU_MIN, v / 100);
  }}
  function tauToSlider(tau) {{
    return 100 * Math.log(tau / TAU_MIN) / Math.log(TAU_MAX / TAU_MIN);
  }}
  function fmtTau(tau) {{
    if (tau < 60) return tau.toFixed(1) + 's';
    return (tau / 60).toFixed(1) + 'm';
  }}
  function computeYield(times, total, tau) {{
    let sum = 0;
    for (const t of times) sum += 1 / (1 + t / tau);
    return sum / total * 100;
  }}

  const vramSlider = document.getElementById('vram-slider');
  const tauSlider  = document.getElementById('tau-slider');
  const vramLabel  = document.getElementById('vram-label');
  const tauLabel   = document.getElementById('tau-label');
  const summary    = document.getElementById('selector-summary');
  const tbody      = document.getElementById('selector-tbody');
  const presets    = document.querySelectorAll('.tau-preset');

  let showExcluded = true;

  function update() {{
    const vramGb = +vramSlider.value;
    const tau    = sliderToTau(+tauSlider.value);

    vramLabel.textContent = vramGb + ' GB';
    tauLabel.textContent  = fmtTau(tau);

    // Sync active preset button
    presets.forEach(b => {{
      const t = +b.dataset.tau;
      b.classList.toggle('active', Math.abs(t - tau) / tau < 0.05);
    }});

    const scored = MODELS.map(m => ({{
      ...m,
      yld: computeYield(m.times, m.total, tau),
      fits: m.vram_mb === null || m.vram_mb / 1024 <= vramGb,
    }}));
    scored.sort((a, b) => b.yld - a.yld);

    const fitsCount = scored.filter(m => m.fits).length;
    const excCount  = scored.length - fitsCount;

    let toggleBtn = '';
    if (excCount > 0) {{
      toggleBtn = `<button class="toggle-excluded" id="toggle-exc">
        ${{showExcluded ? 'hide' : 'show'}} ${{excCount}} excluded</button>`;
    }}
    summary.innerHTML =
      `<strong>${{fitsCount}}</strong> model${{fitsCount !== 1 ? 's' : ''}} fit${{fitsCount === 1 ? 's' : ''}} your VRAM.` +
      (excCount > 0 ? ` <span style="color:#94a3b8">(${{excCount}} exceed it)</span>` : '') +
      toggleBtn;

    document.getElementById('toggle-exc')?.addEventListener('click', () => {{
      showExcluded = !showExcluded;
      update();
    }});

    tbody.innerHTML = '';
    let rank = 0;
    for (const m of scored) {{
      if (!m.fits && !showExcluded) continue;
      if (m.fits) rank++;

      const vramText = m.vram_mb !== null
        ? `<span class="${{m.fits ? '' : 'vram-badge'}}">${{(m.vram_mb / 1024).toFixed(1)}} GB</span>`
        : '<span style="color:#94a3b8">?</span>';
      const toolsBadge = m.tools ? ' <span title="Supports tool use" style="font-size:0.85em">🔧</span>' : '';
      const paramsText = (m.params ?? '<span style="color:#94a3b8">?</span>') + toolsBadge;

      const row = document.createElement('tr');
      if (!m.fits) row.classList.add('row-excluded');
      row.innerHTML = `
        <td class="rank-cell ${{m.fits && rank <= 3 ? 'top' : ''}}">${{m.fits ? rank : '—'}}</td>
        <td><a href="${{m.html_file}}">${{m.name}}</a></td>
        <td>${{paramsText}}</td>
        <td>${{vramText}}</td>
        <td>${{m.pass_rate.toFixed(1)}}%</td>
        <td>${{m.avg_time_s.toFixed(1)}}s</td>
        <td>
          <div class="yield-cell">
            <div class="yield-bar-wrap">
              <div class="yield-bar" style="width:${{m.yld.toFixed(1)}}%"></div>
            </div>
            <span class="yield-val">${{m.yld.toFixed(1)}}%</span>
          </div>
        </td>`;
      tbody.appendChild(row);
    }}
  }}

  vramSlider.addEventListener('input', update);
  tauSlider.addEventListener('input', update);

  presets.forEach(btn => {{
    btn.addEventListener('click', () => {{
      const tau = +btn.dataset.tau;
      tauSlider.value = tauToSlider(tau).toFixed(1);
      update();
    }});
  }});

  update();
}})();
</script>
</body>
</html>
""")
