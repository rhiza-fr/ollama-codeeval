"""Setup report page: agent flow diagram, configuration, and machine details."""

import html
import json

from ollama_codeeval.data import DATA_FILENAME
from ollama_codeeval.report._shared import (
    CSS_BLOCK,
    HEAD_META,
    OUTPUT_DIR,
    SORT_SCRIPT,
    _nav_html,
)


def _load_system_profile() -> dict | None:
    """Load output/system_profile.json if it exists."""
    from ollama_codeeval.config import OUTPUT_BASE

    path = OUTPUT_BASE / "system_profile.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_model_stats() -> dict | None:
    """Load output/model_stats.json if it exists."""
    from ollama_codeeval.config import OUTPUT_BASE

    path = OUTPUT_BASE / "model_stats.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _format_bytes(size: float) -> str:
    """Pretty-print a byte count."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _section(title: str, body: str) -> str:
    return f"<h2>{html.escape(title)}</h2>\n{body}"


def _kv_table(rows: list[tuple[str, str]], first_col_width: str = "280px") -> str:
    """Build a key-value table. Each row is (key, value-or-html)."""
    if not rows:
        return '<p class="note">No data available.</p>'
    out = "<table><tbody>\n"
    for k, v in rows:
        out += f'<tr><td style="font-weight:600;white-space:nowrap;width:{first_col_width}">{html.escape(k)}</td><td>{v}</td></tr>\n'
    out += "</tbody></table>\n"
    return out


# ---------------------------------------------------------------------------
# Flow diagram (SVG) — matches agent.py create_flow()
# ---------------------------------------------------------------------------

_FLOW_SVG = """<?xml version="1.0"?>
<svg width="1000" height="600" xmlns="http://www.w3.org/2000/svg" xmlns:svg="http://www.w3.org/2000/svg" class="cascade-diagram">
 <defs>
  <marker id="arr" markerHeight="7" markerWidth="10" orient="auto" refX="10" refY="3.5">
   <polygon fill="#555" id="svg_1" points="0 0, 10 3.5, 0 7"/>
  </marker>
  <marker id="arr-green" markerHeight="7" markerWidth="10" orient="auto" refX="10" refY="3.5">
   <polygon fill="#16a34a" id="svg_2" points="0 0, 10 3.5, 0 7"/>
  </marker>
  <marker id="arr-orange" markerHeight="7" markerWidth="10" orient="auto" refX="10" refY="3.5">
   <polygon fill="#ea580c" id="svg_4" points="0 0, 10 3.5, 0 7"/>
  </marker>
  <marker id="arr-purple" markerHeight="7" markerWidth="10" orient="auto" refX="10" refY="3.5">
   <polygon fill="#8b5cf6" id="svg_5" points="0 0, 10 3.5, 0 7"/>
  </marker>
  <marker id="mkr_end_svg_48" markerHeight="5" markerUnits="strokeWidth" markerWidth="5" orient="auto" refX="50" refY="50" se_type="rightarrow" viewBox="0 0 100 100">
   <path d="m100,50l-100,40l30,-40l-30,-40z" fill="#dc2626" stroke="#dc2626" stroke-width="10"/>
  </marker>
  <marker id="mkr_end_svg_42" markerHeight="5" markerUnits="strokeWidth" markerWidth="5" orient="auto" refX="50" refY="50" se_type="rightarrow" viewBox="0 0 100 100">
   <path d="m100,50l-100,40l30,-40l-30,-40z" fill="#dc2626" stroke="#dc2626" stroke-width="10"/>
  </marker>
 </defs>
 <!-- Nodes: main column -->
 <!-- Right-side fix nodes (aligned with Execute row) -->
 <!-- Left-side lint node -->
 <!-- Straight vertical arrows (main column) -->
 <!-- Generate fail → Respond (left path) -->
 <!-- RuffFix → LintFix (left arrow, upper half) -->
 <!-- LintFix → RuffFix (straight horizontal return, just below LintFix center) -->
 <!-- RuffFix fail → Respond (down) -->
 <!-- Execute → Fix (right arrow, error) — straight horizontal since both at y=245..287 -->
 <!-- Execute → Respond (ok, center path) -->
 <!-- Execute → Respond (fail, right of ok) -->
 <!-- Fix → RuffFix — exits right, up right-side trunk -->
 <!-- Fix → FixHarder (stuck) -->
 <!-- FixHarder → RuffFix — exits right, up right-side trunk, same backbone -->
 <!-- FixHarder → Respond (verystuck) -->
 <!-- Legend -->
 <!-- Flow label -->
 <g class="layer">
  <title>Layer 1</title>
  <rect fill="#bfdbfe" height="42" id="svg_6" rx="6" stroke="#3b82f6" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" width="150" x="350" y="10"/>
  <text fill="#1e3a5f" font-family="sans-serif" font-size="13" id="svg_7" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="425" y="36">FormatOriginalQuestion</text>
  <rect fill="#dbeafe" height="42" id="svg_8" rx="6" stroke="#3b82f6" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" width="150" x="350" y="85"/>
  <text fill="#1e3a5f" font-family="sans-serif" font-size="13" id="svg_9" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="425" y="111">Generate</text>
  <rect fill="#fef3c7" height="42" id="svg_10" rx="6" stroke="#f59e0b" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" width="150" x="350" y="165"/>
  <text fill="#78350f" font-family="sans-serif" font-size="12" id="svg_11" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="425" y="186">RuffFix</text>
  <text fill="#92400e" font-family="sans-serif" font-size="9" id="svg_12" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="425" y="200">format + check --fix</text>
  <rect fill="#d1fae5" height="42" id="svg_13" rx="6" stroke="#10b981" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" width="150" x="350" y="245"/>
  <text fill="#064e3b" font-family="sans-serif" font-size="12" id="svg_14" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="425" y="266">Execute</text>
  <text fill="#047857" font-family="sans-serif" font-size="9" id="svg_15" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="425" y="280">Docker sandbox</text>
  <rect fill="#fee2e2" height="42" id="svg_16" rx="6" stroke="#ef4444" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" width="150" x="350" y="420"/>
  <text fill="#7f1d1d" font-family="sans-serif" font-size="13" id="svg_17" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="425" y="446">Respond (done)</text>
  <rect fill="#fce7f3" height="42" id="svg_18" rx="6" stroke="#ec4899" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" width="150" x="580" y="245"/>
  <text fill="#831843" font-family="sans-serif" font-size="12" id="svg_19" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="655" y="263">Fix</text>
  <text fill="#9d174d" font-family="sans-serif" font-size="9" id="svg_20" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="655" y="278">error feedback + temp escalate</text>
  <rect fill="#ede9fe" height="42" id="svg_21" rx="6" stroke="#8b5cf6" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" width="150" x="580" y="325"/>
  <text fill="#4c1d95" font-family="sans-serif" font-size="12" id="svg_22" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="655" y="343">FixHarder</text>
  <text fill="#6d28d9" font-family="sans-serif" font-size="9" id="svg_23" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="655" y="358">fresh restart, high temp</text>
  <rect fill="#fef9c3" height="42" id="svg_24" rx="6" stroke="#eab308" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" width="150" x="95" y="165"/>
  <text fill="#854d0e" font-family="sans-serif" font-size="12" id="svg_25" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="170" y="186">LintFix</text>
  <text fill="#a16207" font-family="sans-serif" font-size="9" id="svg_26" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="170" y="200">LLM fixes ruff errors</text>
  <line id="svg_27" marker-end="url(#arr)" stroke="#555" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" x1="425" x2="425" y1="52" y2="85"/>
  <line id="svg_28" marker-end="url(#arr)" stroke="#555" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" x1="425" x2="425" y1="127" y2="165"/>
  <line id="svg_29" marker-end="url(#arr)" stroke="#555" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" x1="425" x2="425" y1="207" y2="245"/>
  <text fill="#dc2626" font-family="sans-serif" font-size="9" id="svg_31" transform="matrix(1 0 0 1 -47 11)" x="285" y="309">fail</text>
  <line id="svg_32" marker-end="url(#arr)" stroke="#555" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" x1="350" x2="245" y1="178" y2="178"/>
  <text fill="#555" font-family="sans-serif" font-size="9" id="svg_33" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="297" y="170">lint_error</text>
  <line id="svg_34" marker-end="url(#arr)" stroke="#eab308" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" x1="245" x2="350" y1="197" y2="197"/>
  <text fill="#92400e" font-family="sans-serif" font-size="9" id="svg_35" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="297" y="193">retry</text>
  <path d="m350,207.9l-22.54,24.47l-0.46,208.73l23,0" fill="none" id="svg_36" marker-end="url(#arr-red)" stroke="#dc2626" stroke-dasharray="4,3" stroke-width="1.2" transform="matrix(1 0 0 1 -68 7)"/>
  <line id="svg_38" marker-end="url(#arr-orange)" stroke="#ea580c" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" x1="500" x2="580" y1="266" y2="266"/>
  <text fill="#ea580c" font-family="sans-serif" font-size="9" id="svg_39" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="535" y="259">error</text>
  <line id="svg_40" marker-end="url(#arr-green)" stroke="#16a34a" stroke-width="2" transform="matrix(1 0 0 1 -68 7)" x1="400" x2="400" y1="287" y2="420"/>
  <text fill="#16a34a" font-family="sans-serif" font-size="10" id="svg_41" text-anchor="end" transform="matrix(1 0 0 1 -68 7)" x="390" y="360">ok</text>
  <line id="svg_42" marker-end="url(#mkr_end_svg_42)" stroke="#dc2626" stroke-dasharray="4,3" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" x1="445" x2="445" y1="287" y2="417"/>
  <text fill="#dc2626" font-family="sans-serif" font-size="9" id="svg_43" transform="matrix(1 0 0 1 -68 7)" x="454" y="360">fail</text>
  <path d="m730,266l30,0l0,-80l-260,0" fill="none" id="svg_44" marker-end="url(#arr)" stroke="#555" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)"/>
  <line id="svg_45" marker-end="url(#arr-orange)" stroke="#ea580c" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)" x1="655" x2="655" y1="287" y2="325"/>
  <text fill="#ea580c" font-family="sans-serif" font-size="9" id="svg_46" transform="matrix(1 0 0 1 -68 7)" x="665" y="310">stuck</text>
  <path d="m730,346l30,0l0,-160l-260,0" fill="none" id="svg_47" marker-end="url(#arr-purple)" stroke="#8b5cf6" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)"/>
  <path d="m655.03,367l0,74l-150.07,0" fill="none" id="svg_48" marker-end="url(#mkr_end_svg_48)" stroke="#dc2626" stroke-dasharray="4,3" stroke-width="1.5" transform="matrix(1 0 0 1 -68 7)"/>
  <text fill="#dc2626" font-family="sans-serif" font-size="9" id="svg_49" text-anchor="middle" transform="matrix(1 0 0 1 -68 7)" x="578" y="436">verystuck</text>
  <text fill="#94a3b8" font-family="sans-serif" font-size="9" id="svg_64" text-anchor="middle" transform="matrix(1 0 0 1 -55 -307)" x="655" y="490">Loops up to MAX_ITERATIONS</text>
  <line fill="none" id="svg_70" stroke="#dc2626" stroke-dasharray="4,3" stroke-width="1.2" transform="matrix(1 0 0 1 -68 7)" x1="348.86" x2="328.86" y1="103.29" y2="103.29"/>
  <line fill="none" id="svg_71" stroke="#dc2626" stroke-dasharray="4,3" stroke-width="1.2" transform="matrix(1 0 0 1 -68 7)" x1="327.86" x2="327.86" y1="101.29" y2="231.29"/>
 </g>
</svg>"""

# ---------------------------------------------------------------------------
# Config defaults table
# ---------------------------------------------------------------------------


def _config_rows() -> str:
    from ollama_codeeval import config

    rows = [
        ("OLLAMA_HOST", str(config.OLLAMA_HOST)),
        ("MODEL_TEMPERATURE", str(config.MODEL_TEMPERATURE)),
        ("MAX_ITERATIONS", str(config.MAX_ITERATIONS)),
        ("SANDBOX_LANG", config.SANDBOX_LANG),
        ("SANDBOX_IMAGE", config.SANDBOX_IMAGE),
        ("SANDBOX_TIMEOUT", f"{config.SANDBOX_TIMEOUT}s"),
        ("EXECUTION_CACHE_DIR", config.EXECUTION_CACHE_DIR),
        ("OUTPUT_BASE", str(config.OUTPUT_BASE)),
        ("OUTPUT_HTML", str(config.OUTPUT_HTML)),
    ]
    return _kv_table(rows)


# ---------------------------------------------------------------------------
# AutoContext defaults
# ---------------------------------------------------------------------------


def _autocontext_rows() -> str:
    return _kv_table(
        [
            ("min_num_predict", "512"),
            ("max_num_predict", "16,384"),
            ("num_predict_growth", "1.5x"),
            ("num_predict_shrinkage", "1.0x (disabled)"),
            ("num_predict_chunk", "64"),
            ("min_num_ctx", "4,096"),
            ("max_num_ctx", "16,384"),
            ("num_ctx_growth", "1.5x"),
            ("num_ctx_shrinkage", "1.0x (disabled)"),
            ("num_ctx_chunk", "256"),
            ("max retries", "10"),
            ("context cap check", "90% of num_ctx"),
            ("prediction cap check", "90% of num_predict"),
            ("initial ctx estimation", "4 chars/token heuristic"),
        ]
    )


# ---------------------------------------------------------------------------
# Machine details
# ---------------------------------------------------------------------------


def _cpu_section(profile: dict) -> str:
    cpu = profile.get("cpu", {})
    rows = []
    if cpu.get("system"):
        rows.append(
            ("OS", f"{cpu['system']} {cpu.get('release', '')} {cpu.get('version', '')}")
        )
    if cpu.get("processor"):
        rows.append(("Processor", cpu["processor"]))
    if cpu.get("machine"):
        rows.append(("Architecture", cpu["machine"]))
    if cpu.get("cpu_count_logical"):
        rows.append(("Logical CPUs", str(cpu["cpu_count_logical"])))

    # Parse wmic output for Windows
    wmic = cpu.get("wmic", "")
    if wmic:
        rows.append(
            ("WMIC", f'<pre style="font-size:0.8em;margin:0">{html.escape(wmic)}</pre>')
        )
    model_names = cpu.get("model_names", [])
    for mn in model_names:
        rows.append(("Model", html.escape(mn.split(":")[-1].strip())))

    if not rows:
        return '<p class="note">CPU info not available.</p>'
    return _kv_table(rows)


def _gpu_section(profile: dict) -> str:
    gpu = profile.get("gpu", {})
    if not gpu.get("available"):
        return '<p class="note">GPU info not available — nvidia-smi not found or no NVIDIA GPU.</p>'
    rows = []
    for g in gpu.get("gpus", []):
        rows.append(
            (
                f"GPU {g['index']} — {g['name']}",
                f"Memory: {g['memory']}, Driver: {g['driver']}",
            )
        )
    return _kv_table(rows)


def _docker_section(profile: dict) -> str:
    d = profile.get("docker", {})
    rows = [
        ("Version", d.get("version", "not available")),
        ("Server Info", d.get("info", "not available")),
        ("Sandbox Image (python-sandbox)", d.get("sandbox_image", "unknown")),
    ]
    return _kv_table(rows)


def _ruff_section(profile: dict) -> str:
    r = profile.get("ruff", {})
    rows = [
        ("Version", r.get("version", "not available")),
    ]
    return _kv_table(rows)


def _ollama_section(profile: dict) -> str:
    o = profile.get("ollama", {})
    cli = o.get("cli", {})
    api = o.get("api", {})

    rows = [("CLI Version", cli.get("cli_version", "not available"))]
    if api.get("available"):
        rows.append(("API Version", html.escape(str(api.get("version", "unknown")))))
    else:
        rows.append(
            ("API Status", html.escape(f"Not available: {api.get('error', '')}"))
        )
    return _kv_table(rows)


def _python_section(profile: dict) -> str:
    p = profile.get("python", {})
    rows = [
        ("Version", p.get("version", "not available")),
        ("Implementation", p.get("implementation", "not available")),
    ]
    return _kv_table(rows)


# ---------------------------------------------------------------------------
# Ollama models table (from model_stats.json)
# ---------------------------------------------------------------------------


def _ollama_models_table(stats: dict) -> str:
    """Build a sortable table of Ollama models from model_stats.json."""
    if not stats:
        return '<p class="note">No model statistics available. Run <code>discover_model_stats</code> first.</p>'

    rows = ""
    for name in sorted(stats.keys()):
        m = stats[name]
        size_str = _format_bytes(m.get("size", 0))
        family = m.get("family") or m.get("families") or "—"
        if isinstance(family, list):
            family = ", ".join(family)
        param_size = m.get("parameter_size") or "—"
        quant = m.get("quantization_level") or "—"
        fmt = m.get("format") or "—"
        ctx_len = m.get("context_length") or "—"
        if isinstance(ctx_len, (int, float)):
            ctx_len = f"{int(ctx_len):,}"

        rows += (
            f"<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(str(family))}</td>"
            f"<td>{html.escape(str(param_size))}</td>"
            f'<td data-sort-value="{m.get("size", 0)}">{size_str}</td>'
            f"<td>{html.escape(str(quant))}</td>"
            f"<td>{html.escape(str(fmt))}</td>"
            f'<td data-sort-value="{m.get("context_length", 0) or 0}">{ctx_len}</td>'
            f"</tr>\n"
        )

    return f"""<table>
<thead><tr>
  <th class="sortable">Name</th>
  <th class="sortable">Family</th>
  <th class="sortable">Param Size</th>
  <th class="sortable">Disk Size</th>
  <th class="sortable">Quant</th>
  <th class="sortable">Format</th>
  <th class="sortable">Context Length</th>
</tr></thead>
<tbody>{rows}</tbody></table>"""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _dataset_section() -> str:
    return _kv_table(
        [
            (
                "Dataset",
                '<a href="https://github.com/marcusm117/human-eval-enhanced">human-eval-enhanced-202307</a> by marcusm117',
            ),
            (
                "Source",
                '<a href="https://github.com/openai/human-eval">openai/human-eval</a> — the original 164-problem benchmark',
            ),
            (
                "Fixes applied",
                (
                    '<a href="https://github.com/openai/human-eval/pull/23">openai/human-eval#23</a> — '
                    "community-contributed corrections to buggy test cases and docstrings in the original dataset "
                    "(wrong expected outputs, ambiguous prompts, off-by-one errors)"
                ),
            ),
            (
                "File",
                f"<code>{DATA_FILENAME}</code> — loaded from <code>data/</code> or downloaded to the platform cache",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Vendored modules
# ---------------------------------------------------------------------------


def _vendored_section() -> str:
    return _kv_table(
        [
            (
                "pocketflow.py",
                'Vendored from <a href="https://github.com/The-Pocket/PocketFlow">PocketFlow</a> — lightweight async node/graph framework that I was playing with.',
            )
        ]
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_setup_html() -> None:
    """Write the setup page to output/html/setup.html."""
    profile = _load_system_profile()
    # model_stats = _load_model_stats()

    # Machine details section
    machine_html = ""
    if profile:
        machine_html += _section("CPU", _cpu_section(profile))
        machine_html += _section("GPU", _gpu_section(profile))
        machine_html += _section("Docker", _docker_section(profile))
        machine_html += _section("Ruff", _ruff_section(profile))
        machine_html += _section("Ollama", _ollama_section(profile))
        machine_html += _section("Python", _python_section(profile))
    else:
        machine_html = '<p class="note">No system profile found. Run <code>python -m ollama_codeeval.system_profile</code> to collect machine details.</p>'

    page = f"""<!doctype html>
<html>
<head>{HEAD_META}<title>Humaneval LLM Benchmark — Setup</title>
{CSS_BLOCK}{SORT_SCRIPT}</head>
<body>
{_nav_html("setup")}
<div class="page-content">
  <h1>Evaluation Setup</h1>
  <p>
    This page documents the default agent pipeline, configuration values,
    and machine environment used for evaluation.  Values shown are defaults
    from <code>config.py</code>; actual runs may override them via CLI flags.
  </p>

  <h2>Agent Flow Diagram</h2>
  <p>
    The PocketFlow graph defined in <code>agent.create_flow()</code>. Each node
    emits an action that routes to the next node.  The <b>Fix</b>/<b>FixHarder</b>
    loop retries up to <code>MAX_ITERATIONS</code> (default 5) per problem.
    <b>RuffFix</b> auto-formats and applies safe fixes with ruff before
    execution; <b>LintFix</b> asks the LLM to fix remaining lint errors.
  </p>
  <p>
    Recent models could have managed tools themselves. This hard scafolding enables 
    older models to be tested. It is intentionally forgiving to sloppy output (extra comments, code blocks, indentation), 
    and runs ruff to fix little errors before hitting the sandbox.
  </p>
  {_FLOW_SVG}

  
  <h2>Dataset</h2>
  {_dataset_section()}

  <h2>Machine Details</h2>
  {machine_html}

  <h2>Configuration Defaults</h2>
  <p>From <code>src/ollama_codeeval/config.py</code>:</p>
  {_config_rows()}

  <h2>AutoContext Defaults</h2>
  <p>
    The <code>AutoContextClient</code>
    wraps <code>ollama_think.Client</code> with automatic context/predict window
    growth.  When the model returns <code>done_reason: "length"</code>, the
    client grows <code>num_predict</code> or <code>num_ctx</code> and retries.
    Growth is multiplicative (1.5x) and never shrinks (1.0x shrinkage).
    <br/><br/>
    This is required, to keep num_ctx small where possible, but allow growth for thinking models.
  </p>
  {_autocontext_rows()}


  <h2>Vendored Modules</h2>
  {_vendored_section()}
</div>
</body>
</html>
"""

    with open(OUTPUT_DIR / "setup.html", "w", encoding="utf-8") as f:
        f.write(page)
