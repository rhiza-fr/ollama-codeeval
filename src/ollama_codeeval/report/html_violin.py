"""Generate cross-model comparison charts for the index page using Plotly."""

import json
import logging

from ollama_codeeval.report._shared import load_model_stats, resolve_model_name, tag_display_name

log = logging.getLogger(__name__)


def _is_cascade(d: dict) -> bool:
    return (d.get("tag") or "").startswith(("cascade", "champion"))


def _model_label(data: dict) -> str:
    """Short model label: 'model', 'model (Think)', 'model (Dataset: X)', etc."""
    if _is_cascade(data):
        return "★ " + tag_display_name(data)
    parts = []
    if data["think"]:
        parts.append("Think")
    dataset = data.get("dataset", "humaneval")
    if dataset != "humaneval":
        parts.append(f"Dataset: {dataset}")
    suffix = f" ({', '.join(parts)})" if parts else ""
    return f"{data['model']}{suffix}"


def _filter_complete(all_data: list[dict]) -> list[dict]:
    """Keep only models that ran the full test suite (max total_tests)."""
    if not all_data:
        return all_data
    full_count = max(d["total_tests"] for d in all_data)
    return [d for d in all_data if d["total_tests"] == full_count]


def _chart_quality_at_speed(all_data: list[dict]) -> str:
    """Scatter: pass rate (x) vs median iteration time (y).

    Base models shown as circles/diamonds with labels.
    Rewrite variants shown as smaller triangles, connected to their base by lines.
    """
    def _pct(d: dict) -> float:
        return round(d["passed_tests"] / d["total_tests"] * 100, 1)

    def _time(d: dict) -> float:
        return round(d["average_time_per_iteration"], 2)

    # Index base models only
    base_by_key: dict[tuple, dict] = {}
    for d in all_data:
        if d.get("dataset", "humaneval") != "humaneval":
            continue
        key = (d["model"], d["think"], d.get("tag") or "")
        base_by_key[key] = d

    # Base model traces (with labels)
    traces = []
    def _hover(d: dict) -> str:
        return f"{_model_label(d)}<br>{_pct(d)}%<br>{_time(d)}s"

    for think_val, symbol, name in [(False, "circle", "Standard"), (True, "diamond", "Think")]:
        subset = [d for d in base_by_key.values() if d["think"] == think_val and not _is_cascade(d)]
        if not subset:
            continue
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": [_pct(d) for d in subset],
            "y": [_time(d) for d in subset],
            "text": [_model_label(d) for d in subset],
            "hovertext": [_hover(d) for d in subset],
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 9},
            "marker": {"size": 12, "symbol": symbol},
            "name": name,
        })

    # Champion trace
    champions = [d for d in base_by_key.values() if _is_cascade(d)]
    if champions:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": [_pct(d) for d in champions],
            "y": [_time(d) for d in champions],
            "text": [_model_label(d) for d in champions],
            "hovertext": [_hover(d) for d in champions],
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 10, "color": "#e67e22"},
            "marker": {"size": 16, "symbol": "star", "color": "#e67e22"},
            "name": _model_label(champions[0]),
        })

    layout = {
        "xaxis": {"title": "Pass rate (%)"},
        "yaxis": {"title": "Median iteration time (s)", "type": "log"},
        "legend": {"orientation": "h", "y": -0.10},
        "margin": {"t": 20, "b": 60},
        "hovermode": "closest",
    }
    return (
        '<div id="chartSpeed" style="width:100%;height:750px;"></div>\n'
        f"<script>Plotly.newPlot('chartSpeed', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


_ITER_COLORS = ["#3498db", "#5dace0", "#7abfcc", "#96d0a8", "#78c27a", "#2ecc71"]


def _cum_avg_wall_time(d: dict) -> list[float | None]:
    """Mean cumulative wall-clock time (seconds) at each iteration boundary.

    Returns None for iterations where no tasks have that many iterations.
    """
    n_iters = len(d["cum_pass_per_iteration"])
    result: list[float | None] = []
    for i in range(n_iters):
        times = []
        for task in d["tasks"]:
            task_iters = task.get("iterations", [])
            if len(task_iters) >= i + 1:
                cum = sum(it.get("total_duration", 0) or 0 for it in task_iters[:i + 1]) / 1e9
                times.append(cum)
        result.append(round(sum(times) / len(times), 2) if times else None)
    return result


def _chart_quality_at_speed_iterations(all_data: list[dict]) -> str:
    """Scatter: cumulative pass rate per iteration (x) vs avg wall time at that iteration (y).

    One dot per (model, iteration) for non-rewrite runs.
    Per-model connecting lines show how quality and time cost grow together.
    """
    base = [d for d in all_data if d.get("dataset", "humaneval") == "humaneval"]
    if not base:
        return ""

    wall_times = {id(d): _cum_avg_wall_time(d) for d in base}

    # One polyline per model connecting its iteration dots in order (skip None)
    line_x: list = []
    line_y: list = []
    for d in base:
        wt = wall_times[id(d)]
        iters = d["cum_pass_per_iteration"]
        for px, py in zip(iters, wt):
            if py is None:
                break
            line_x.append(round(px, 1))
            line_y.append(py)
        line_x.append(None)
        line_y.append(None)

    # Last valid iteration index per model (for the green diamond)
    def _last_valid(d: dict) -> int:
        wt = wall_times[id(d)]
        for i in range(len(wt) - 1, -1, -1):
            if wt[i] is not None:
                return i
        return 0

    n_iters = max(len(d["cum_pass_per_iteration"]) for d in base)
    traces = [{
        "type": "scatter",
        "mode": "lines",
        "x": line_x,
        "y": line_y,
        "line": {"color": "#bdc3c7", "width": 1.5},
        "showlegend": False,
        "hoverinfo": "skip",
    }]
    for i in range(n_iters):
        color = _ITER_COLORS[min(i, len(_ITER_COLORS) - 1)]
        subset = [d for d in base if wall_times[id(d)][i] is not None]
        if not subset:
            continue
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [round(d["cum_pass_per_iteration"][i], 1) for d in subset],
            "y": [wall_times[id(d)][i] for d in subset],
            "hovertext": [
                f"{_model_label(d)}<br>{round(d['cum_pass_per_iteration'][i], 1)}%<br>{wall_times[id(d)][i]}s"
                for d in subset
            ],
            "hoverinfo": "text",
            "name": f"After {i + 1} attempt{'s' if i > 0 else ''}",
            "marker": {"size": 8, "color": color, "symbol": "circle"},
        })

    # Green diamond at each model's final valid iteration (non-champions)
    non_champs = [d for d in base if not _is_cascade(d)]
    traces.append({
        "type": "scatter",
        "mode": "markers+text",
        "x": [round(d["cum_pass_per_iteration"][_last_valid(d)], 1) for d in non_champs],
        "y": [wall_times[id(d)][_last_valid(d)] for d in non_champs],
        "text": [_model_label(d) for d in non_champs],
        "hovertext": [
            f"{_model_label(d)}<br>{round(d['cum_pass_per_iteration'][_last_valid(d)], 1)}%<br>{wall_times[id(d)][_last_valid(d)]}s"
            for d in non_champs
        ],
        "hoverinfo": "text",
        "textposition": "top center",
        "textfont": {"size": 9},
        "name": "Final attempt",
        "marker": {"size": 12, "color": "#2ecc71", "symbol": "diamond"},
    })

    # Champion trace — orange star
    champs = [d for d in base if _is_cascade(d)]
    if champs:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": [round(d["cum_pass_per_iteration"][_last_valid(d)], 1) for d in champs],
            "y": [wall_times[id(d)][_last_valid(d)] for d in champs],
            "text": [_model_label(d) for d in champs],
            "hovertext": [
                f"{_model_label(d)}<br>{round(d['cum_pass_per_iteration'][_last_valid(d)], 1)}%<br>{wall_times[id(d)][_last_valid(d)]}s"
                for d in champs
            ],
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 10, "color": "#e67e22"},
            "name": _model_label(champs[0]),
            "marker": {"size": 16, "symbol": "star", "color": "#e67e22"},
        })

    layout = {
        "xaxis": {"title": "Pass rate (%)"},
        "yaxis": {"title": "Avg wall time at iteration (s)", "type": "log"},
        "legend": {"orientation": "h", "y": -0.10},
        "margin": {"t": 20, "b": 60},
        "hovermode": "closest",
    }
    return (
        '<div id="chartSpeedIter" style="width:100%;height:750px;"></div>\n'
        f"<script>Plotly.newPlot('chartSpeedIter', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


def _chart_iteration_lift(all_data: list[dict]) -> str:
    """Dumbbell chart: one marker per iteration per model (no rewrites).

    Message: iteration improves results.
    Final iteration shown as a green diamond; earlier ones as circles.
    """
    base = [d for d in all_data if d.get("dataset", "humaneval") == "humaneval"]
    base.sort(key=lambda d: d["cum_pass_per_iteration"][0])
    labels = [_model_label(d) for d in base]

    n_iters = max(len(d["cum_pass_per_iteration"]) for d in base)
    # Per-iteration x values (None if a model has fewer iterations)
    iter_x: list[list[float | None]] = []
    for i in range(n_iters):
        iter_x.append([round(d["cum_pass_per_iteration"][i], 1) if i < len(d["cum_pass_per_iteration"]) else None for d in base])

    # Connecting lines from iter-1 to final for each model
    shapes = []
    for j, label in enumerate(labels):
        x0 = iter_x[0][j]
        x1 = iter_x[-1][j]
        if x0 is not None and x1 is not None:
            shapes.append({
                "type": "line",
                "x0": x0, "x1": x1,
                "y0": label, "y1": label,
                "yref": "y",
                "line": {"color": "#bdc3c7", "width": 2},
            })

    champion_mask = [_is_cascade(d) for d in base]
    _cascade_trace_name = next((_model_label(d) for d in base if _is_cascade(d)), "★ Cascade")

    traces = []
    for i in range(n_iters):
        is_final = i == n_iters - 1
        color = _ITER_COLORS[min(i, len(_ITER_COLORS) - 1)]
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [x if not champion_mask[j] else None for j, x in enumerate(iter_x[i])],
            "y": labels,
            "name": f"After {i + 1} attempt{'s' if i > 0 else ''}",
            "marker": {
                "size": 12 if is_final else 8,
                "color": "#2ecc71" if is_final else color,
                "symbol": "diamond" if is_final else "circle",
            },
        })

    # Champion overlay: gold star at each iteration
    for i in range(n_iters):
        is_final = i == n_iters - 1
        cx = [iter_x[i][j] for j, m in enumerate(champion_mask) if m]
        cy = [labels[j] for j, m in enumerate(champion_mask) if m]
        if cx:
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": cx,
                "y": cy,
                "showlegend": is_final,
                "name": _cascade_trace_name,
                "marker": {"size": 16 if is_final else 12, "color": "#e67e22", "symbol": "star"},
            })

    layout = {
        "xaxis": {"title": "Pass rate (%)", "range": [0, 105]},
        "yaxis": {"ticksuffix": "  "},
        "legend": {"orientation": "h", "y": -0.10},
        "hoverlabel": {"namelength": -1},
        "margin": {"l": 200, "t": 20, "b": 60},
        "height": max(300, len(labels) * 28 + 80),
        "shapes": shapes,
    }
    return (
        '<div id="chartIteration" style="width:100%;"></div>\n'
        f"<script>Plotly.newPlot('chartIteration', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


_REWRITE_STYLES = [
    {"color": "#2980b9", "symbol": "diamond"},
    {"color": "#8e44ad", "symbol": "diamond"},
    {"color": "#e67e22", "symbol": "diamond"},
    {"color": "#e74c3c", "symbol": "diamond"},
]


def _chart_rewrite_lift(all_data: list[dict]) -> str:
    """One row per base model, dots for no-rewrite and each rewrite model.

    Message: rewriting improves results.
    """
    # Group by base model key
    base_by_key: dict[tuple, dict] = {}
    rewrites_by_key: dict[tuple, list[dict]] = {}
    rewrite_models_seen: list[str] = []
    for d in all_data:
        key = (d["model"], d["think"])
        dataset = d.get("dataset", "humaneval")
        if dataset != "humaneval":
            rewrites_by_key.setdefault(key, []).append(d)
            if dataset not in rewrite_models_seen:
                rewrite_models_seen.append(dataset)
        else:
            base_by_key[key] = d

    # Only keep base models that have at least one rewrite
    keys = [k for k in base_by_key if k in rewrites_by_key]
    if not keys:
        return ""

    # Sort rows by base model's iteration 0 pass rate
    keys.sort(key=lambda k: base_by_key[k]["cum_pass_per_iteration"][0])
    labels = [_model_label(base_by_key[k]) for k in keys]

    # Connecting line per row: from base to best rewrite
    shapes = []
    for key, label in zip(keys, labels):
        base_pct = base_by_key[key]["cum_pass_per_iteration"][0]
        best_pct = max(rw["passed_tests"] / rw["total_tests"] for rw in rewrites_by_key[key]) * 100
        shapes.append({
            "type": "line",
            "x0": round(base_pct, 1), "x1": round(best_pct, 1),
            "y0": label, "y1": label,
            "yref": "y",
            "line": {"color": "#bdc3c7", "width": 2},
        })

    # Base trace (no rewrite)
    traces = [{
        "type": "scatter",
        "mode": "markers",
        "x": [round(base_by_key[k]["cum_pass_per_iteration"][0], 1) for k in keys],
        "y": labels,
        "name": "No rewrite",
        "marker": {"size": 10, "color": "#95a5a6"},
    }]

    # One trace per rewrite model
    for idx, rm in enumerate(rewrite_models_seen):
        style = _REWRITE_STYLES[idx % len(_REWRITE_STYLES)]
        x_vals = []
        y_vals = []
        for key, label in zip(keys, labels):
            for rw in rewrites_by_key.get(key, []):
                if rw.get("dataset", "humaneval") == rm:
                    x_vals.append(round(rw["passed_tests"] / rw["total_tests"] * 100, 1))
                    y_vals.append(label)
        if x_vals:
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": x_vals,
                "y": y_vals,
                "name": f"Rewrite: {rm}",
                "marker": {"size": 10, **style},
            })

    layout = {
        "xaxis": {"title": "Pass rate (%)", "range": [0, 105]},
        "yaxis": {"ticksuffix": "  "},
        "legend": {"orientation": "h", "y": -0.10},
        "hoverlabel": {"namelength": -1},
        "margin": {"l": 200, "t": 20, "b": 60},
        "height": max(350, len(labels) * 45 + 100),
        "shapes": shapes,
    }
    return (
        '<div id="chartRewrite" style="width:100%;"></div>\n'
        f"<script>Plotly.newPlot('chartRewrite', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


def chart_combined_progression(all_data: list[dict]) -> str:
    """One row per base model showing iter 1, final, and each rewrite result."""
    # Index everything by base key
    base_by_key: dict[tuple, dict] = {}
    rewrites_by_key: dict[tuple, dict[str, dict]] = {}  # key -> {rewrite_model: data}
    rewrite_models_seen: list[str] = []
    for d in all_data:
        key = (d["model"], d["think"], d.get("tag") or "")
        dataset = d.get("dataset", "humaneval")
        if dataset != "humaneval":
            rewrites_by_key.setdefault(key, {})[dataset] = d
            if dataset not in rewrite_models_seen:
                rewrite_models_seen.append(dataset)
        else:
            base_by_key[key] = d

    if not base_by_key:
        return ""

    # Sort by first-attempt pass rate
    keys = sorted(
        base_by_key,
        key=lambda k: base_by_key[k]["cum_pass_per_iteration"][0],
    )
    labels = [_model_label(base_by_key[k]) for k in keys]

    def _pct(d: dict) -> float:
        return round(d["passed_tests"] / d["total_tests"] * 100, 1)

    iter1 = [round(base_by_key[k]["cum_pass_per_iteration"][0], 1) for k in keys]
    final = [round(base_by_key[k]["cum_pass_per_iteration"][-1], 1) for k in keys]

    # Connecting line from min to max value per row
    shapes = []
    for i, key in enumerate(keys):
        all_x = [iter1[i], final[i]]
        for rm_data in rewrites_by_key.get(key, {}).values():
            all_x.append(_pct(rm_data))
        shapes.append({
            "type": "line",
            "x0": min(all_x), "x1": max(all_x),
            "y0": labels[i], "y1": labels[i],
            "yref": "y",
            "line": {"color": "#bdc3c7", "width": 2},
        })

    traces = [
        {
            "type": "scatter",
            "mode": "markers",
            "x": iter1,
            "y": labels,
            "name": "1st attempt",
            "marker": {"size": 8, "color": "#3498db"},
        },
        {
            "type": "scatter",
            "mode": "markers",
            "x": final,
            "y": labels,
            "name": "After all attempts",
            "marker": {"size": 10, "color": "#2ecc71", "symbol": "diamond"},
        },
    ]

    for idx, rm in enumerate(rewrite_models_seen):
        style = _REWRITE_STYLES[idx % len(_REWRITE_STYLES)]
        x_vals = []
        y_vals = []
        for key, label in zip(keys, labels):
            rw_data = rewrites_by_key.get(key, {}).get(rm)
            if rw_data:
                x_vals.append(_pct(rw_data))
                y_vals.append(label)
        if x_vals:
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "x": x_vals,
                "y": y_vals,
                "name": f"Rewrite: {rm}",
                "marker": {"size": 10, **style},
            })

    layout = {
        "xaxis": {"title": "Pass rate (%)", "range": [0, 105]},
        "yaxis": {"ticksuffix": "  "},
        "legend": {"orientation": "h", "y": -0.10},
        "hoverlabel": {"namelength": -1},
        "margin": {"l": 200, "t": 20, "b": 60},
        "height": max(400, len(labels) * 35 + 120),
        "shapes": shapes,
    }
    return (
        '<div id="chartCombined" style="width:100%;"></div>\n'
        f"<script>Plotly.newPlot('chartCombined', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


def _resolve_model_stats(model_name: str, stats: dict) -> dict | None:
    """Find stats entry for a model name.

    model_name may be a raw Ollama name ('qwen3:latest') or a resolved display
    name ('qwen3:8.2b').  Try exact match, :latest suffix, then reverse-resolve
    by scanning stats for any entry that canonicalises to this display name.
    """
    if model_name in stats:
        return stats[model_name]
    candidate = f"{model_name}:latest"
    if candidate in stats:
        return stats[candidate]
    # model_name is likely a resolved display name (e.g. "qwen3:8.2b" from "qwen3:latest").
    # Find the original stats key that resolves to it.
    for key in stats:
        if resolve_model_name(key, stats) == model_name:
            return stats[key]
    return None


def _chart_size_vs_pass_rate(all_data: list[dict]) -> str:
    """Scatter: pass rate (x) vs model parameter count (y, log).

    Results from the same base model sit on the same horizontal row,
    with their pass rates spread along the x-axis.
    """
    stats = load_model_stats()
    if not stats:
        log.warning("model_stats.json not found — skipping size vs pass rate chart")
        return ""

    from collections import defaultdict

    by_size: dict[int, list[dict]] = defaultdict(list)
    skipped = []
    for d in all_data:
        if d.get("dataset", "humaneval") != "humaneval":
            continue
        ms = _resolve_model_stats(d["model"], stats)
        if not ms or not ms.get("parameter_count"):
            skipped.append(d["model"])
            continue
        param_count = ms["parameter_count"]
        pct = round(d["passed_tests"] / d["total_tests"] * 100, 1)
        by_size[param_count].append({
            "label": _model_label(d),
            "pct": pct,
            "think": d["think"],
            "cascade": _is_cascade(d),
            "param_count": param_count,
            "param_size": ms.get("parameter_size", ""),
            "quantization": ms.get("quantization_level", ""),
        })

    if skipped:
        log.info("Size chart: skipped models without stats: %s", set(skipped))

    if not by_size:
        return ""

    standard_x, standard_y, standard_text, standard_label = [], [], [], []
    think_x, think_y, think_text, think_label = [], [], [], []
    champion_x, champion_y, champion_text, champion_label = [], [], [], []

    for param_count, entries in by_size.items():
        for entry in entries:
            hover = f"{entry['label']}<br>{entry['param_size']} ({entry['quantization']})<br>{entry['pct']}%"
            if entry["cascade"]:
                champion_x.append(entry["pct"])
                champion_y.append(param_count)
                champion_text.append(hover)
                champion_label.append(entry["label"])
            elif entry["think"]:
                think_x.append(entry["pct"])
                think_y.append(param_count)
                think_text.append(hover)
                think_label.append(entry["label"])
            else:
                standard_x.append(entry["pct"])
                standard_y.append(param_count)
                standard_text.append(hover)
                standard_label.append(entry["label"])

    traces = []
    if standard_x:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": standard_x,
            "y": standard_y,
            "text": standard_label,
            "hovertext": standard_text,
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 9},
            "name": "Standard",
            "marker": {"size": 10, "color": "#3498db"},
        })
    if think_x:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": think_x,
            "y": think_y,
            "text": think_label,
            "hovertext": think_text,
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 9},
            "name": "Think",
            "marker": {"size": 10, "color": "#e74c3c", "symbol": "diamond"},
        })
    if champion_x:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": champion_x,
            "y": champion_y,
            "text": champion_label,
            "hovertext": champion_text,
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 10, "color": "#e67e22"},
            "name": champion_label[0],
            "marker": {"size": 16, "color": "#e67e22", "symbol": "star"},
        })

    layout = {
        "xaxis": {"title": "Pass rate (%)", "range": [0, 105]},
        "yaxis": {"title": "Parameter count"},
        "legend": {"orientation": "h", "y": -0.10},
        "margin": {"t": 20, "b": 60},
        "hovermode": "closest",
    }
    return (
        '<div id="chartSize" style="width:100%;height:750px;"></div>\n'
        f"<script>Plotly.newPlot('chartSize', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


def _load_vram_bench() -> dict[str, int] | None:
    """Load the latest vram_bench_*.json and map model -> max VRAM (MB).

    Each model may have multiple entries (one per context size). We take the
    maximum ollama_size_vram_mb across context sizes as the model's VRAM.
    Returns None if no vram_bench files found.
    """
    from ollama_codeeval.config import OUTPUT_BASE

    latest_ts = ""
    latest_path = None
    for path in OUTPUT_BASE.glob("vram_bench_*.json"):
        ts = path.stem  # e.g. vram_bench_20260302_170213
        if ts > latest_ts:
            latest_ts = ts
            latest_path = path

    if latest_path is None:
        log.warning("No vram_bench_*.json files found — skipping VRAM vs pass rate chart")
        return None

    from collections import defaultdict
    data = json.loads(latest_path.read_text())
    log.info("Loaded VRAM bench data from %s", latest_path.name)

    # Group by model name, taking max VRAM
    model_vrams: dict[str, list[int]] = defaultdict(list)
    for entry in data["results"]:
        vram = entry.get("ollama_size_vram_mb")
        if vram is not None:
            model_vrams[entry["model"]].append(vram)

    return {model: max(vrams) for model, vrams in model_vrams.items()}


def _chart_vram_vs_pass_rate(all_data: list[dict]) -> str:
    """Scatter: pass rate (x) vs Ollama VRAM size in GB (y).

    Same structure as _chart_size_vs_pass_rate but uses measured VRAM usage
    instead of theoretical parameter count.
    """
    vram_data = _load_vram_bench()
    if not vram_data:
        return ""

    stats = load_model_stats()
    from collections import defaultdict

    by_vram: dict[float, list[dict]] = defaultdict(list)
    skipped = []
    for d in all_data:
        if d.get("dataset", "humaneval") != "humaneval":
            continue
        if _is_cascade(d):
            vram_mb = _cascade_vram_total(d, vram_data, stats)
        else:
            vram_mb = _resolve_vram(d["model"], vram_data, stats)
        if vram_mb is None:
            skipped.append(d["model"])
            continue
        vram_gb = round(vram_mb / 1024, 1)
        pct = round(d["passed_tests"] / d["total_tests"] * 100, 1)
        by_vram[vram_gb].append({
            "label": _model_label(d),
            "pct": pct,
            "think": d["think"],
            "cascade": _is_cascade(d),
            "vram_gb": vram_gb,
        })

    if skipped:
        log.info("VRAM chart: skipped models without VRAM data: %s", set(skipped))

    if not by_vram:
        return ""

    standard_x, standard_y, standard_text, standard_label = [], [], [], []
    think_x, think_y, think_text, think_label = [], [], [], []
    champion_x, champion_y, champion_text, champion_label = [], [], [], []

    for vram_gb, entries in by_vram.items():
        for entry in entries:
            vram_label = "Total VRAM" if entry["cascade"] else "VRAM"
            hover = f"{entry['label']}<br>{entry['vram_gb']:.1f} GB {vram_label}<br>{entry['pct']}%"
            if entry["cascade"]:
                champion_x.append(entry["pct"])
                champion_y.append(vram_gb)
                champion_text.append(hover)
                champion_label.append(entry["label"])
            elif entry["think"]:
                think_x.append(entry["pct"])
                think_y.append(vram_gb)
                think_text.append(hover)
                think_label.append(entry["label"])
            else:
                standard_x.append(entry["pct"])
                standard_y.append(vram_gb)
                standard_text.append(hover)
                standard_label.append(entry["label"])

    traces = []
    if standard_x:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": standard_x,
            "y": standard_y,
            "text": standard_label,
            "hovertext": standard_text,
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 9},
            "name": "Standard",
            "marker": {"size": 10, "color": "#3498db"},
        })
    if think_x:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": think_x,
            "y": think_y,
            "text": think_label,
            "hovertext": think_text,
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 9},
            "name": "Think",
            "marker": {"size": 10, "color": "#e74c3c", "symbol": "diamond"},
        })
    if champion_x:
        traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": champion_x,
            "y": champion_y,
            "text": champion_label,
            "hovertext": champion_text,
            "hoverinfo": "text",
            "textposition": "top center",
            "textfont": {"size": 10, "color": "#e67e22"},
            "name": champion_label[0],
            "marker": {"size": 16, "color": "#e67e22", "symbol": "star"},
        })

    layout = {
        "xaxis": {"title": "Pass rate (%)", "range": [0, 105]},
        "yaxis": {"title": "Ollama VRAM (GB)"},
        "legend": {"orientation": "h", "y": -0.10},
        "margin": {"t": 20, "b": 60},
        "hovermode": "closest",
    }
    return (
        '<div id="chartVram" style="width:100%;height:750px;"></div>\n'
        f"<script>Plotly.newPlot('chartVram', {json.dumps(traces)}, {json.dumps(layout)});</script>"
    )


def _cascade_vram_total(d: dict, vram_data: dict[str, int], stats: dict | None) -> int | None:
    """Sum VRAM across all unique models used in a cascade run."""
    seen: dict[str, None] = {}
    for task in d.get("tasks", []):
        for it in task.get("iterations", []):
            m = it.get("model")
            if m and m not in seen:
                seen[m] = None
    if not seen:
        return None
    total = 0
    for m in seen:
        vram = _resolve_vram(m, vram_data, stats)
        if vram is None:
            return None
        total += vram
    return total


def _resolve_vram(model_name: str, vram_data: dict[str, int], stats: dict | None) -> int | None:
    """Find VRAM size for a model, using the same resolution logic as _resolve_model_stats.

    Handles raw Ollama names, :latest suffixes, and display-name resolution
    (e.g. 'deepseek-r1:14b' → find 'deepseek-r1:latest' in vram_data).
    """
    if model_name in vram_data:
        return vram_data[model_name]

    # Try appending :latest
    candidate = f"{model_name}:latest"
    if candidate in vram_data:
        return vram_data[candidate]

    # If model_name is a resolved display name (e.g. 'deepseek-r1:8.2b'),
    # try to find the original stats key that resolves to it, then match VRAM.
    if stats:
        for key in stats:
            if resolve_model_name(key, stats) == model_name and key in vram_data:
                return vram_data[key]
            if key == model_name and key in vram_data:
                return vram_data[key]

    return None


def generate_charts_html(all_data: list[dict]) -> str:
    """Return all index-page chart HTML."""
    filtered = _filter_complete(all_data)
    sections = []

    sections.append("<h2>Pass Rate @5 vs Time</h2>")
    sections.append("<p>This shows log-time. Scores low and to the right are better. The <a href=\"cascade.html\">cascade</a> is a special run switching between three models.</p>")
    sections.append(_chart_quality_at_speed(filtered))

    iter_speed_html = _chart_quality_at_speed_iterations(filtered)
    if iter_speed_html:
        sections.append("<h2>Pass Rate vs Time — by Iteration</h2>")
        sections.append("<p>Iteration improves score at the cost of time. This shows log-time. Scores low and to the right are better. If a model moves up but not to the right it indicates that iterations are not improving the score</p>")
        sections.append(iter_speed_html)

    sections.append("<h2>Iteration Improves Results</h2>")
    sections.append("<p>Allowing multiple attempts lifts pass rates.</p>")
    sections.append(_chart_iteration_lift(filtered))

    vram_html = _chart_vram_vs_pass_rate(filtered)
    if vram_html:
        sections.append("<h2>VRAM Usage vs Pass Rate</h2>")
        sections.append("<p>Measured Ollama VRAM (GB) vs final pass rate. Thinking increases PassRate@5, at the expense of time.</p>")
        sections.append(vram_html)

    return "\n".join(sections)
