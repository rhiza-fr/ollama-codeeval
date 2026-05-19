"""Tests for AutoContextClient - context/prediction size management."""
from unittest.mock import MagicMock, patch

import pytest

from ollama_codeeval.autocontext import AutoContextClient


def _make_response(
    done_reason="stop",
    prompt_eval_count=10,
    eval_count=50,
    load_duration=1_000_000_000,
    prompt_eval_duration=500_000_000,
    eval_duration=2_000_000_000,
    content="def foo(): return None",
):
    """Create a mock ThinkResponse-like object."""
    data = {
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "load_duration": load_duration,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_duration": eval_duration,
        "message": {"content": content, "role": "assistant"},
    }
    mock = MagicMock()
    mock.get.side_effect = lambda key, default=None: data.get(key, default)
    mock.to_dict.return_value = data
    mock.__getitem__ = lambda self, k: data[k]
    return mock


@pytest.fixture
def client():
    """AutoContextClient with real parent init (no network calls at init time)."""
    return AutoContextClient(
        min_num_predict=512,
        max_num_predict=4096,
        min_num_ctx=4096,
        max_num_ctx=16384,
    )


class TestAutoContextClientInit:
    def test_valid_init(self):
        c = AutoContextClient(min_num_predict=512, min_num_ctx=4096)
        assert c.min_num_predict == 512
        assert c.min_num_ctx == 4096
        assert c._default_num_predict == 512
        assert c._default_num_ctx == 4096

    def test_predict_greater_than_ctx_raises(self):
        with pytest.raises(ValueError, match="min_num_predict"):
            AutoContextClient(min_num_predict=8192, min_num_ctx=4096)

    def test_max_predict_greater_than_max_ctx_raises(self):
        with pytest.raises(ValueError, match="max_num_predict"):
            AutoContextClient(
                min_num_predict=512,
                max_num_predict=32768,
                min_num_ctx=4096,
                max_num_ctx=16384,
            )


class TestRoundUpToChunk:
    def test_already_aligned(self, client):
        assert client._roundupToChunk(256, 64) == 256

    def test_rounds_up(self, client):
        assert client._roundupToChunk(300, 64) == 320

    def test_zero(self, client):
        assert client._roundupToChunk(0, 64) == 0


class TestEstimateNumCtx:
    def test_estimates_from_messages(self, client):
        messages = [{"content": "a" * 400}]  # 400 chars + 20 overhead = 420 / 4 = 105
        result = client._estimate_num_ctx_for_messages(messages)
        assert result == (400 + 20) // 4

    def test_empty_messages(self, client):
        assert client._estimate_num_ctx_for_messages([]) == 0

    def test_estimate_for_call_uses_prompt_when_no_messages(self, client):
        result = client._estimate_num_ctx_for_call("a" * 80, [])
        # Uses prompt as single user message: (80 + 20) // 4 = 25
        assert result == (80 + 20) // 4

    def test_estimate_for_call_uses_messages_when_provided(self, client):
        messages = [{"content": "a" * 80}]
        result = client._estimate_num_ctx_for_call("ignored prompt", messages)
        assert result == (80 + 20) // 4


class TestUpdateNumCtxStats:
    def test_first_call_initializes(self, client):
        client._update_num_ctx_stats(4096, 10.0, 100)
        stats = client._num_ctx_stats[4096]
        assert stats["count"] == 1
        assert stats["avg_tokens_per_second"] == 10.0
        assert stats["avg_num_tokens"] == 100.0

    def test_second_call_averages(self, client):
        client._update_num_ctx_stats(4096, 10.0, 100)
        client._update_num_ctx_stats(4096, 20.0, 200)
        stats = client._num_ctx_stats[4096]
        assert stats["count"] == 2
        assert stats["avg_tokens_per_second"] == 15.0
        assert stats["avg_num_tokens"] == 150.0


class TestStats:
    def test_stats_returns_dict(self, client):
        s = client.stats()
        assert "load_durations" in s
        assert "ctx_stats" in s

    def test_stats_after_update(self, client):
        client._load_durations[4096] = 1.5
        s = client.stats()
        assert s["load_durations"][4096] == 1.5


class TestNumPredictProperty:
    def test_defaults_to_min(self, client):
        assert client._num_predict == 512

    def test_setter(self, client):
        client._num_predict = 1024
        assert client._num_predict == 1024


class TestNumCtxProperty:
    def test_defaults_to_min(self, client):
        assert client._num_ctx == 4096

    def test_setter(self, client):
        client._num_ctx = 8192
        assert client._num_ctx == 8192


class TestCall:
    def test_normal_call_injects_options(self, client):
        resp = _make_response()
        with patch("ollama_think.Client.call", return_value=resp) as mock_call:
            client.call("model", prompt="hello", think=False, options={})
        call_kwargs = mock_call.call_args[1]
        assert "num_predict" in call_kwargs["options"]
        assert "num_ctx" in call_kwargs["options"]

    def test_adds_options_when_missing(self, client):
        resp = _make_response()
        with patch("ollama_think.Client.call", return_value=resp) as mock_call:
            client.call("model", prompt="hello")
        assert "options" in mock_call.call_args[1]

    def test_normal_response_returns_without_retry(self, client):
        resp = _make_response(done_reason="stop")
        with patch("ollama_think.Client.call", return_value=resp) as mock_call:
            result = client.call("model", prompt="hello")
        assert mock_call.call_count == 1
        assert result is resp

    def test_non_integer_prompt_tokens_returns_early(self, client):
        resp = _make_response()
        resp.get.side_effect = lambda k, default=None: {
            "prompt_eval_count": "not-an-int",
            "eval_count": 50,
        }.get(k, default)
        with patch("ollama_think.Client.call", return_value=resp) as mock_call:
            result = client.call("model", prompt="hello")
        assert mock_call.call_count == 1
        assert result is resp

    def test_non_integer_durations_returns_early(self, client):
        data = {
            "prompt_eval_count": 10,
            "eval_count": 50,
            "load_duration": 1_000_000_000,
            "prompt_eval_duration": "bad",
            "eval_duration": 2_000_000_000,
            "done_reason": "stop",
        }
        resp = MagicMock()
        resp.get.side_effect = lambda k, default=None: data.get(k, default)
        with patch("ollama_think.Client.call", return_value=resp) as mock_call:
            result = client.call("model", prompt="hello")
        assert mock_call.call_count == 1
        assert result is resp

    def test_length_reason_grows_num_ctx_and_retries(self, client):
        # First call: done_reason="length", total_tokens close to num_ctx (grow ctx)
        resp1 = _make_response(
            done_reason="length",
            prompt_eval_count=3700,  # > 4096 * 0.9 = 3686
            eval_count=100,
        )
        resp2 = _make_response(done_reason="stop")
        with patch("ollama_think.Client.call", side_effect=[resp1, resp2]) as mock_call:
            result = client.call("model", prompt="hello")
        assert mock_call.call_count == 2
        assert result is resp2

    def test_length_reason_grows_num_predict_and_retries(self, client):
        # eval_count close to num_predict (512 * 0.9 = 460)
        resp1 = _make_response(
            done_reason="length",
            prompt_eval_count=10,
            eval_count=470,  # > 512 * 0.9 = 460
        )
        resp2 = _make_response(done_reason="stop")
        with patch("ollama_think.Client.call", side_effect=[resp1, resp2]) as mock_call:
            result = client.call("model", prompt="hello")
        assert mock_call.call_count == 2
        assert result is resp2

    def test_length_reason_fallback_growth(self, client):
        # Neither total_tokens nor eval_tokens are close to limits -> fallback growth
        resp1 = _make_response(
            done_reason="length",
            prompt_eval_count=10,
            eval_count=10,  # low, not near limits
        )
        resp2 = _make_response(done_reason="stop")
        with patch("ollama_think.Client.call", side_effect=[resp1, resp2]) as mock_call:
            result = client.call("model", prompt="hello")
        # Should still retry since fallback grows num_predict
        assert mock_call.call_count == 2
        assert result is resp2

    def test_length_reason_max_retries_stops_retrying(self):
        # Use huge limits so growth always happens until MAX_RETRIES is exhausted
        c = AutoContextClient(
            min_num_predict=512,
            max_num_predict=10_000_000,
            min_num_ctx=4096,
            max_num_ctx=20_000_000,
        )
        resp = _make_response(
            done_reason="length",
            prompt_eval_count=3700,
            eval_count=470,
        )
        with patch("ollama_think.Client.call", return_value=resp) as mock_call:
            result = c.call("model", prompt="hello")
        assert mock_call.call_count == AutoContextClient._MAX_RETRIES + 1
        assert result is resp

    def test_pre_grows_num_ctx_for_large_prompt(self, client):
        # Pass a very large prompt so that estimated context exceeds current num_ctx
        large_prompt = "a" * 20000  # ~5000 tokens estimate
        resp = _make_response()
        initial_num_ctx = client._num_ctx
        with patch("ollama_think.Client.call", return_value=resp):
            client.call("model", prompt=large_prompt)
        # num_ctx should have grown to accommodate the large prompt
        assert client._num_ctx > initial_num_ctx

    def test_zero_duration_avoids_division_by_zero(self, client):
        data = {
            "prompt_eval_count": 10,
            "eval_count": 50,
            "load_duration": 0,
            "prompt_eval_duration": 0,
            "eval_duration": 0,
            "done_reason": "stop",
        }
        resp = MagicMock()
        resp.get.side_effect = lambda k, default=None: data.get(k, default)
        with patch("ollama_think.Client.call", return_value=resp):
            # Should not raise ZeroDivisionError
            client.call("model", prompt="hello")
