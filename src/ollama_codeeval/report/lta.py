"""Log-Time AUC (LTA) — area under success_rate(t) in ln(t) space, normalized to [0, 1]."""

import bisect
import math


def compute_lta(times: list[float], rates: list[float], t_min: float, t_max: float) -> float:
    """
    Compute Log-Time AUC for a step-function timeseries.

    times: sorted event times (seconds), each marks when a task completed
    rates: success rate (0–100) at each event time
    t_min: log-scale lower bound (> 0), typically the earliest first-success across all models
    t_max: log-scale upper bound (> t_min), typically the latest completion across all models

    Returns a float in [0, 1]. Higher = better (fast AND accurate).
    Equal weight is given to each order of magnitude of time.
    """
    if not times or t_max <= t_min or t_min <= 0:
        return 0.0

    log_range = math.log(t_max / t_min)

    # All breakpoints in [t_min, t_max] where the step function could change value
    # Strict inequality: t_min and t_max are already in breakpoints, avoid duplicates
    inner = [t for t in times if t_min < t < t_max]
    breakpoints = sorted(set([t_min] + inner + [t_max]))

    auc = 0.0
    for i in range(len(breakpoints) - 1):
        left, right = breakpoints[i], breakpoints[i + 1]
        # Rate during [left, right): last event at or before `left`
        idx = bisect.bisect_right(times, left) - 1
        rate = rates[idx] if idx >= 0 else 0.0
        auc += rate * math.log(right / left)

    return auc / (log_range * 100)
