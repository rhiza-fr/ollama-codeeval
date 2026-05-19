import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

log = logging.getLogger(__name__)


class KDEEstimator:
    def __init__(self, measurements):
        self.measurements = np.array([m for m in measurements if m is not None])
        self.min = np.min(self.measurements)
        self.max = np.max(self.measurements)
        self.filtered_data = None
        self.raw_median = None
        self.robust_std = None
        self.lower_bound = None
        self.upper_bound = None
        self.kde = None
        self.x_eval = None
        self.y_density = None

    def filter_outliers(self, std_factor=6):
        self.raw_median = np.median(self.measurements)
        mad = np.median(np.abs(self.measurements - self.raw_median))
        self.robust_std = mad * 1.4826
        self.lower_bound = self.raw_median - std_factor * self.robust_std
        self.upper_bound = self.raw_median + std_factor * self.robust_std
        self.filtered_data = self.measurements[
            (self.measurements >= self.lower_bound)
            & (self.measurements <= self.upper_bound)
        ]

    def fit_kde(self):
        if self.filtered_data is None:
            raise ValueError("Filter outliers before fitting KDE.")
        if len(self.filtered_data) < 2:
            return
        self.kde = gaussian_kde(self.filtered_data)
        self.x_eval = np.linspace(
            min(self.filtered_data), max(self.filtered_data), 1000
        )
        self.y_density = self.kde(self.x_eval)

    def plot(self, output_file=None, label="KDE"):
        if (
            self.x_eval is None
            or self.y_density is None
            or self.filtered_data is None
            or self.raw_median is None
        ):
            raise ValueError("Fit KDE before plotting.")
        plt.figure(figsize=(10, 6))
        plt.hist(
            self.filtered_data,
            bins=50,
            density=True,
            alpha=0.6,
            color="skyblue",
            label="Histogram of Filtered Data",
        )
        plt.plot(
            self.x_eval,
            self.y_density,
            "r-",
            linewidth=2,
            label="KDE (Gaussian Kernel)",
        )
        plt.axvline(
            self.raw_median,
            color="g",
            linestyle="--",
            label=f"Raw Median ({self.raw_median:.2f})",
        )
        plt.title(f"{label} min: {self.min}, max: {self.max}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_file:
            plt.savefig(output_file)
            log.info("Plot saved to %s", output_file)
            plt.close()
        else:
            plt.show()

    def print_sample_kde(self, n=5):
        if self.x_eval is None or self.y_density is None:
            raise ValueError("Fit KDE before printing sample values.")
        log.info("Sample KDE density values:")
        for i in range(n):
            log.info("x = %.3f, density = %.3f", self.x_eval[i], self.y_density[i])


class MultiKDEPlotter:
    def __init__(self):
        self.kdes = []  # List of tuples: (filtered_data, x_eval, y_density, label, color)

    def add_kde(self, measurements, label=None, color=None, std_factor=4):
        measurements = np.array(measurements)
        raw_median = np.median(measurements)
        mad = np.median(np.abs(measurements - raw_median))
        robust_std = mad * 1.4826
        lower_bound = raw_median - std_factor * robust_std
        upper_bound = raw_median + std_factor * robust_std
        filtered_data = measurements[
            (measurements >= lower_bound) & (measurements <= upper_bound)
        ]
        kde = gaussian_kde(filtered_data)
        x_eval = np.linspace(min(filtered_data), max(filtered_data), 1000)
        y_density = kde(x_eval)
        self.kdes.append(
            {
                "filtered_data": filtered_data,
                "x_eval": x_eval,
                "y_density": y_density,
                "label": label,
                "color": color,
                "raw_median": raw_median,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    def plot(self, output_file=None, show_histogram=True):
        plt.figure(figsize=(10, 6))
        for kde_info in self.kdes:
            color = kde_info["color"] if kde_info["color"] else None
            label = kde_info["label"] if kde_info["label"] else "KDE"
            if show_histogram:
                plt.hist(
                    kde_info["filtered_data"],
                    bins=30,
                    density=True,
                    alpha=0.3,
                    color=color,
                    label=f"Histogram {label}",
                )
            plt.plot(
                kde_info["x_eval"],
                kde_info["y_density"],
                "-",
                linewidth=2,
                color=color,
                label=f"KDE {label}",
            )
            plt.axvline(
                kde_info["raw_median"],
                color=color if color else "g",
                linestyle="--",
                alpha=0.7,
                label=f"Median {label} ({kde_info['raw_median']:.2f})",
            )
        plt.xlabel("Measurement Values")
        plt.ylabel("Probability Density")
        plt.title("Multiple Kernel Density Estimates (KDEs)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_file:
            plt.savefig(output_file)
            log.info("Plot saved to %s", output_file)
            plt.close()
        else:
            plt.show()
