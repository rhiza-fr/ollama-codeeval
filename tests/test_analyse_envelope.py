"""Tests for analyse_envelope helper functions."""

import json
import logging

import pytest

from ollama_codeeval.report.analyse_envelope import interpolate_success_rate


class TestInterpolateSuccessRate:
    """Tests for interpolate_success_rate."""

    def test_empty_times_and_rates(self):
        ts = {"times": [], "success_rates": []}
        assert interpolate_success_rate(ts, 5.0) == 0.0

    def test_empty_times_nonempty_rates(self):
        ts = {"times": [], "success_rates": [42.0]}
        assert interpolate_success_rate(ts, 5.0) == 42.0

    def test_single_point_before(self):
        ts = {"times": [10.0], "success_rates": [50.0]}
        assert interpolate_success_rate(ts, 5.0) == 50.0

    def test_single_point_exact(self):
        ts = {"times": [10.0], "success_rates": [50.0]}
        assert interpolate_success_rate(ts, 10.0) == 50.0

    def test_single_point_after(self):
        ts = {"times": [10.0], "success_rates": [50.0]}
        assert interpolate_success_rate(ts, 15.0) == 50.0

    def test_exact_match_first(self):
        ts = {"times": [1.0, 2.0, 3.0], "success_rates": [10.0, 20.0, 30.0]}
        assert interpolate_success_rate(ts, 1.0) == 10.0

    def test_exact_match_last(self):
        ts = {"times": [1.0, 2.0, 3.0], "success_rates": [10.0, 20.0, 30.0]}
        assert interpolate_success_rate(ts, 3.0) == 30.0

    def test_exact_match_middle(self):
        ts = {"times": [1.0, 2.0, 3.0], "success_rates": [10.0, 20.0, 30.0]}
        assert interpolate_success_rate(ts, 2.0) == 20.0

    def test_interpolation_midpoint(self):
        ts = {"times": [0.0, 10.0], "success_rates": [0.0, 100.0]}
        assert interpolate_success_rate(ts, 5.0) == pytest.approx(50.0)

    def test_interpolation_quarter(self):
        ts = {"times": [0.0, 10.0], "success_rates": [0.0, 100.0]}
        assert interpolate_success_rate(ts, 2.5) == pytest.approx(25.0)

    def test_before_first_point(self):
        ts = {"times": [5.0, 10.0], "success_rates": [20.0, 80.0]}
        assert interpolate_success_rate(ts, 1.0) == 20.0

    def test_after_last_point(self):
        ts = {"times": [5.0, 10.0], "success_rates": [20.0, 80.0]}
        assert interpolate_success_rate(ts, 100.0) == 80.0

    def test_interpolation_between_segments(self):
        ts = {"times": [0.0, 5.0, 10.0], "success_rates": [0.0, 50.0, 100.0]}
        assert interpolate_success_rate(ts, 7.5) == pytest.approx(75.0)

    def test_interpolation_nonuniform_spacing(self):
        ts = {"times": [0.0, 1.0, 10.0], "success_rates": [0.0, 10.0, 100.0]}
        # Between 1.0 and 10.0: linear from 10% to 100%, at 5.5 => 10 + (90/9)*4.5 = 55
        assert interpolate_success_rate(ts, 5.5) == pytest.approx(55.0)

    def test_duplicate_times_returns_y1(self):
        ts = {"times": [0.0, 5.0, 5.0, 10.0], "success_rates": [0.0, 40.0, 60.0, 100.0]}
        # searchsorted with side="right" at 5.0 gives idx=3, so x1=5.0, x2=10.0
        # That's not a duplicate pair, so normal interpolation applies.
        # For a true duplicate at the query point: query between the duplicates
        # Actually with sorted data [0,5,5,10], querying 5.0 hits >= times[-1]=10? No.
        # searchsorted([0,5,5,10], 5.0, side="right") = 3, so x1=times[2]=5.0, x2=times[3]=10.0
        # normal interpolation: 60 + (100-60)/(10-5) * (5.0-5.0) = 60.0
        assert interpolate_success_rate(ts, 5.0) == pytest.approx(60.0)


class TestAnalyzePerformanceEnvelope:
    """Integration-style tests for analyze_performance_envelope."""

    def test_missing_file_does_not_raise(self, caplog):
        from ollama_codeeval.report.analyse_envelope import analyze_performance_envelope

        with caplog.at_level(logging.ERROR):
            analyze_performance_envelope("nonexistent_file.json")
        assert "not found" in caplog.text

    def test_empty_timeseries_data(self, tmp_path, caplog):
        from ollama_codeeval.report.analyse_envelope import analyze_performance_envelope

        data = {"model_a": {"timeseries": {"times": [], "success_rates": []}}}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))
        with caplog.at_level(logging.WARNING):
            analyze_performance_envelope(str(data_file))
        assert "No time-series data found" in caplog.text

    def test_creates_output_in_custom_path(self, tmp_path):
        from ollama_codeeval.report.analyse_envelope import analyze_performance_envelope

        data = {
            "model_a": {
                "timeseries": {
                    "times": [0, 1.0, 5.0],
                    "success_rates": [0.0, 50.0, 100.0],
                },
                "metadata": {},
            }
        }
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))
        out_path = tmp_path / "sub" / "envelope.html"
        analyze_performance_envelope(str(data_file), output_path=str(out_path))
        assert out_path.exists()
