"""This module provides functions for analyzing time-series data and model performance. The first function interpolates a success rate at a specific time point. The second analyzes performance data after a time threshold."""

import json
import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from rich.logging import RichHandler

from ollama_codeeval.config import OUTPUT_HTML

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger(__name__)

MIN_INTERESTING_TIME = (
    0.1  # The time (in seconds) from which you start caring about the results.
)
RESOLUTION = 10000


def interpolate_success_rate(timeseries, time_point):
    """Finds the expected success rate at a specific time_point."""
    times = timeseries["times"]
    rates = timeseries["success_rates"]
    if not times or time_point <= times[0]:
        return rates[0] if rates else 0.0
    if time_point >= times[-1]:
        return rates[-1]
    idx = np.searchsorted(times, time_point, side="right")
    x1, y1, x2, y2 = (times[idx - 1], rates[idx - 1], times[idx], rates[idx])
    return y1 + (y2 - y1) / (x2 - x1) * (time_point - x1) if x1 != x2 else y1


def _display_name(series_title):
    model = series_title.split("_think=")[0]
    think = (
        series_title.split("_think=")[1].split("_rewrite=")[0]
        if "_think=" in series_title
        else ""
    )
    tag = series_title.split("_tag=")[1] if "_tag=" in series_title else ""
    label = model
    if think == "True":
        label += " (think)"
    if tag:
        label += f" [{tag}]"
    return label


def _load_and_filter_data(data_file, output_path, no_cache):
    """Load JSON data and filter out incomplete/rewritten model runs."""
    if not no_cache and output_path.exists():
        try:
            if output_path.stat().st_mtime > Path(data_file).stat().st_mtime:
                log.info("performance_envelope.html is up to date, skipping.")
                return None, None, None
        except FileNotFoundError:
            pass

    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        log.error("'%s' not found. Please run 'results_parser.py' first.", data_file)
        return None, None, None

    max_tasks = max(
        (d.get("metadata", {}).get("unique_tasks", 0) for d in data.values()),
        default=0,
    )
    incomplete = {
        name
        for name, d in data.items()
        if d.get("metadata", {}).get("unique_tasks", 0) < max_tasks
    }
    if incomplete:
        log.info(
            "Excluding %d model(s) that didn't attempt all %d tests: %s",
            len(incomplete),
            max_tasks,
            ", ".join(sorted(incomplete)),
        )
    rewritten = {
        name
        for name in data
        if (
            ("_rewrite=" in name and name.split("_rewrite=")[1].split("_tag=")[0])
            or ("_dataset=" in name and "rewritten" in name.split("_dataset=")[1])
        )
    }
    if rewritten:
        log.info(
            "Excluding %d rewritten model(s): %s",
            len(rewritten),
            ", ".join(sorted(rewritten)),
        )
    complete_data = {
        k: v for k, v in data.items() if k not in incomplete and k not in rewritten
    }
    return data, incomplete, rewritten, complete_data


def _compute_cascade_timeline(complete_data):
    """Compute which model is the champion at each time point."""
    max_time = max(
        (
            d["timeseries"]["times"][-1]
            for d in complete_data.values()
            if d["timeseries"]["times"]
        ),
        default=0,
    )
    if max_time == 0:
        log.warning("No time-series data found.")
        return None, None, None
    time_points = np.linspace(0, max_time, RESOLUTION)
    champions_over_time = []
    for t in time_points:
        best_model_at_t, _ = max(
            (
                (name, interpolate_success_rate(model["timeseries"], t))
                for name, model in complete_data.items()
            ),
            key=lambda item: item[1],
        )
        champions_over_time.append(best_model_at_t)
    intervals = []
    current_champion, start_time = (champions_over_time[0], time_points[0])
    for i in range(1, len(champions_over_time)):
        if champions_over_time[i] != current_champion:
            end_time = time_points[i]
            end_rate = interpolate_success_rate(
                complete_data[current_champion]["timeseries"], end_time
            )
            intervals.append(
                {
                    "model": current_champion,
                    "start": start_time,
                    "end": end_time,
                    "end_rate": end_rate,
                }
            )
            current_champion, start_time = (champions_over_time[i], end_time)
    end_rate = interpolate_success_rate(
        complete_data[current_champion]["timeseries"], time_points[-1]
    )
    intervals.append(
        {
            "model": current_champion,
            "start": start_time,
            "end": time_points[-1],
            "end_rate": end_rate,
        }
    )
    return time_points, champions_over_time, intervals


MODERN_COLORS = [
    "#0ea5e9",
    "#8b5cf6",
    "#ec4899",
    "#f59e0b",
    "#10b981",
    "#6366f1",
    "#ef4444",
    "#14b8a6",
    "#f97316",
    "#a855f7",
    "#06b6d4",
    "#84cc16",
    "#f43f5e",
    "#22c55e",
    "#3b82f6",
    "#d946ef",
    "#f472b6",
    "#fbbf24",
    "#34d399",
    "#818cf8",
]


def _add_model_traces(fig, data, incomplete, rewritten):
    """Add independent and sequential time-series traces to the figure."""
    color_idx = 0
    shown_models = []
    for model_name, model_data in data.items():
        ts = model_data["timeseries"]
        if not ts["times"] or not ts["success_rates"]:
            continue
        if model_name in incomplete or model_name in rewritten:
            continue

        color = MODERN_COLORS[color_idx % len(MODERN_COLORS)]
        shown_models.append(model_name)
        _tag_val = model_name.split("_tag=")[1] if "_tag=" in model_name else ""
        is_cascade = _tag_val.startswith(("cascade", "champion"))
        label = ("★ " if is_cascade else "") + _display_name(model_name)
        line_width = 5 if is_cascade else 2

        fig.add_trace(
            go.Scatter(
                x=ts["times"],
                y=ts["success_rates"],
                mode="lines",
                name=label,
                legendgroup=model_name,
                line=dict(width=line_width, color=color, shape="hv"),
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Budget: %{x:.1f}s<br>"
                + "Success: %{y:.1f}%<br>"
                + "<extra></extra>",
                visible=True,
            )
        )

        seq = model_data.get("seq_timeseries")
        if seq and seq["times"]:
            fig.add_trace(
                go.Scatter(
                    x=seq["times"],
                    y=seq["success_rates"],
                    mode="lines",
                    name=label,
                    legendgroup=model_name,
                    showlegend=True,
                    visible=False,
                    line=dict(width=line_width, color=color, shape="hv"),
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Cumulative: %{x:.1f}s<br>"
                    + "Success: %{y:.1f}%<br>"
                    + "<extra></extra>",
                )
            )

        color_idx += 1
    return shown_models


def _apply_layout(fig, n_models):
    """Apply layout, updatemenus, and styling to the figure."""
    n = n_models
    vis_independent = [i % 2 == 0 for i in range(2 * n)]
    vis_sequential = [i % 2 == 1 for i in range(2 * n)]
    leg_independent = [i % 2 == 0 for i in range(2 * n)]
    leg_sequential = [i % 2 == 1 for i in range(2 * n)]

    seq_annotation = dict(
        text="Each vertical step = one problem solved.<br>Flat stretches = failures consuming time without progress.",
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.13,
        showarrow=False,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#6b7280"),
        align="left",
    )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Independent budget per problem",
                        method="update",
                        args=[
                            {"visible": vis_independent, "showlegend": leg_independent},
                            {
                                "xaxis.title.text": "Time budget per problem (seconds)",
                                "annotations": [],
                            },
                        ],
                    ),
                    dict(
                        label="Sequential (fastest first)",
                        method="update",
                        args=[
                            {"visible": vis_sequential, "showlegend": leg_sequential},
                            {
                                "xaxis.title.text": "Cumulative wall-clock time (seconds)",
                                "annotations": [seq_annotation],
                            },
                        ],
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Model Performance Over Time",
            "font": {
                "family": "Inter, system-ui, -apple-system, sans-serif",
                "size": 24,
                "color": "#1f2937",
            },
            "x": 0.05,
            "xanchor": "left",
        },
        xaxis={
            "type": "log",
            "title": {
                "text": "Time budget per problem (seconds)",
                "font": {
                    "family": "Inter, system-ui, sans-serif",
                    "size": 14,
                    "color": "#4b5563",
                },
                "standoff": 15,
            },
            "gridcolor": "#e5e7eb",
            "gridwidth": 1,
            "zeroline": False,
            "showline": True,
            "linewidth": 2,
            "linecolor": "#d1d5db",
            "tickfont": {
                "family": "Inter, system-ui, sans-serif",
                "size": 12,
                "color": "#6b7280",
            },
        },
        yaxis={
            "title": {
                "text": "Success Percentage (%)",
                "font": {
                    "family": "Inter, system-ui, sans-serif",
                    "size": 14,
                    "color": "#4b5563",
                },
                "standoff": 15,
            },
            "gridcolor": "#e5e7eb",
            "gridwidth": 1,
            "zeroline": False,
            "showline": True,
            "linewidth": 2,
            "linecolor": "#d1d5db",
            "tickfont": {
                "family": "Inter, system-ui, sans-serif",
                "size": 12,
                "color": "#6b7280",
            },
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        hoverlabel={
            "bgcolor": "#1f2937",
            "bordercolor": "#1f2937",
            "font": {
                "family": "Inter, system-ui, sans-serif",
                "size": 13,
                "color": "#ffffff",
            },
        },
        legend={
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.02,
            "font": {
                "family": "Inter, system-ui, sans-serif",
                "size": 11,
                "color": "#374151",
            },
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": "#e5e7eb",
            "borderwidth": 1,
            "tracegroupgap": 5,
        },
        margin={"l": 80, "r": 250, "t": 80, "b": 70},
    )


def _build_and_save_figure(data, incomplete, rewritten, output_path):
    """Build the Plotly figure, apply layout, and save to HTML."""
    fig = go.Figure()
    shown_models = _add_model_traces(fig, data, incomplete, rewritten)
    _apply_layout(fig, len(shown_models))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_path,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "model_performance",
                "height": 800,
                "width": 1400,
                "scale": 2,
            },
        },
    )
    log.info("Interactive graph saved to: %s", output_path)


def analyze_performance_envelope(data_file, output_path=None, no_cache=False):
    """Analyzes model performance and produces an interactive HTML chart."""
    if output_path is None:
        output_path = OUTPUT_HTML / "performance_envelope.html"
    else:
        output_path = Path(output_path)

    result = _load_and_filter_data(data_file, output_path, no_cache)
    if result[0] is None:
        return
    data, incomplete, rewritten, complete_data = result

    timeline = _compute_cascade_timeline(complete_data)
    if timeline[0] is None:
        return

    _build_and_save_figure(data, incomplete, rewritten, output_path)


if __name__ == "__main__":
    analyze_performance_envelope("output/processed_results.json")
