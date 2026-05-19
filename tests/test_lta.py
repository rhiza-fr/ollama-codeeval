# tests/test_lta.py
import pytest
from ollama_codeeval.report.lta import compute_lta


class TestComputeLta:
    def test_empty_times_returns_zero(self):
        assert compute_lta([], [], t_min=1.0, t_max=10.0) == 0.0

    def test_t_max_lte_t_min_returns_zero(self):
        assert compute_lta([5.0], [100.0], t_min=10.0, t_max=1.0) == 0.0

    def test_t_min_zero_returns_zero(self):
        assert compute_lta([5.0], [100.0], t_min=0.0, t_max=10.0) == 0.0

    def test_instant_full_success_scores_one(self):
        # All tasks solved at t_min → rate=100% held for entire log range → LTA=1.0
        result = compute_lta([1.0], [100.0], t_min=1.0, t_max=10.0)
        assert result == pytest.approx(1.0)

    def test_success_at_halfway_log(self):
        # t_min=1, t_max=100, event at t=10 (geometric midpoint) with rate=100%
        # Segment [1,10]: rate=0 → area=0
        # Segment [10,100]: rate=100 → area=100*ln(10)
        # log_range = ln(100) = 2*ln(10)
        # LTA = 100*ln(10) / (2*ln(10)*100) = 0.5
        result = compute_lta([10.0], [100.0], t_min=1.0, t_max=100.0)
        assert result == pytest.approx(0.5)

    def test_step_function_two_events(self):
        # t_min=1, t_max=100
        # Event at t=10 (50% solved), event at t=100 (100% solved, but t_max=100 so last segment is zero width)
        # Segment [1,10]: rate=0
        # Segment [10,100]: rate=50
        # LTA = 50*ln(10) / (ln(100)*100) = 50*ln(10) / (2*ln(10)*100) = 0.25
        result = compute_lta([10.0, 100.0], [50.0, 100.0], t_min=1.0, t_max=100.0)
        assert result == pytest.approx(0.25)

    def test_event_before_t_min_counts_from_t_min(self):
        # Event at t=0.5 (before t_min=1) with rate=100% → held from t_min onwards → LTA=1.0
        result = compute_lta([0.5], [100.0], t_min=1.0, t_max=10.0)
        assert result == pytest.approx(1.0)

    def test_partial_success(self):
        # Constant 50% success rate over entire range → LTA = 0.5
        result = compute_lta([1.0], [50.0], t_min=1.0, t_max=10.0)
        assert result == pytest.approx(0.5)

    def test_result_in_zero_one_range(self):
        times = [2.0, 5.0, 8.0, 15.0, 30.0]
        rates = [20.0, 40.0, 60.0, 80.0, 100.0]
        result = compute_lta(times, rates, t_min=1.0, t_max=100.0)
        assert 0.0 <= result <= 1.0
