"""
AutoContextClient - Automatic context and prediction size management

This module implements :class:`AutoContextClient`, a thin wrapper around a
generic :class:`Client` that dynamically adjusts the *num_predict* (maximum
number of tokens to generate) and *num_ctx* (maximum context window size)
parameters based on the model's responses.  If the model truncates output
because it reached a length limit (``done_reason == "length"``), the client
re-invokes the request with an expanded context or prediction size until a
complete answer is obtained.

The wrapper preserves the original ``Client`` API, exposing a ``call`` method
that behaves like the base implementation but transparently handles
retries and parameter growth/shrinkage to provide more robust inference
without requiring manual tuning of the size limits."""

import logging
import threading

from ollama_think import Client, ThinkResponse


class AutoContextClient(Client):
    """A Client that automatically adjusts prediction and context sizes.

    The class inherits from :class:`Client` and extends its functionality by monitoring the model's `done_reason`.
    If the model stops due to hitting the `num_predict` or `num_ctx` limits (i.e., `done_reason` is `"length"`), the client automatically increases the relevant limits using the configured growth factors and retries the request. It also shrinks the limits when a prediction finishes successfully.
    The constructor accepts minimum and maximum bounds for both prediction and context sizes as well as growth/shrinkage multipliers that control how aggressively the limits are adjusted.

    This client is useful for workloads where the optimal number of tokens to generate or the amount of context required is unknown in advance."""

    def __init__(
        self,
        min_num_predict: int = 512,
        max_num_predict: int = 16384,
        num_predict_growth: float = 1.5,
        num_predict_shrinkage: float = 1.0,
        num_predict_chunk: int = 64,
        min_num_ctx: int = 4096,
        max_num_ctx: int = 16384,
        num_ctx_growth: float = 1.5,
        num_ctx_shrinkage: float = 1.0,
        num_ctx_chunk: int = 256,
        *args,
        **kwargs,
    ):
        """Initialise an AutoContextClient with bounds and growth/shrink factors for
        prediction and context sizes.

        The constructor validates that the minimum values do not exceed the
        corresponding maximum values and that the minimum prediction size does
        not exceed the minimum context size. It then stores the configuration
        parameters, sets the current `num_predict` and `num_ctx` values to the
        minimum bounds, and forwards any remaining arguments to the base
        :class:`Client` constructor.

        Args:
            min_num_predict (int): Minimum number of tokens to predict. Must be
                less than or equal to ``min_num_ctx``.
            max_num_predict (int): Maximum number of tokens to predict. Must be
                less than or equal to ``max_num_ctx``.
            num_predict_growth (float): Factor used to increase the prediction
                limit when the model stops because it hit the limit.
            num_predict_shrinkage (float): Factor used to reduce the prediction
                limit after a successful completion.
            num_predict_chunk (int): Chunk size for rounding up prediction limits.
            min_num_ctx (int): Minimum context window size. Must be greater than
                or equal to ``min_num_predict``.
            max_num_ctx (int): Maximum context window size. Must be greater than
                or equal to ``max_num_predict``.
            num_ctx_growth (float): Factor used to increase the context limit
                when the model stops because it hit the limit.
            num_ctx_shrinkage (float): Factor used to reduce the context limit
                after a successful completion.
            num_ctx_chunk (int): Chunk size for rounding up context limits.
            *args: Positional arguments forwarded to :class:`Client`.
            **kwargs: Keyword arguments forwarded to :class:`Client`.

        Raises:
            ValueError: If ``min_num_predict`` > ``min_num_ctx`` or
                ``max_num_predict`` > ``max_num_ctx``."""
        super().__init__(*args, **kwargs)
        if min_num_predict > min_num_ctx:
            raise ValueError("min_num_predict cannot be greater than min_num_ctx")
        if max_num_predict > max_num_ctx:
            raise ValueError("max_num_predict cannot be greater than max_num_ctx")
        self.min_num_predict = min_num_predict
        self.max_num_predict = max_num_predict
        self.num_predict_growth = num_predict_growth
        self.num_predict_shrinkage = num_predict_shrinkage
        self.num_predict_chunk = num_predict_chunk
        self.min_num_ctx = min_num_ctx
        self.max_num_ctx = max_num_ctx
        self.num_ctx_growth = num_ctx_growth
        self.num_ctx_shrinkage = num_ctx_shrinkage
        self.num_ctx_chunk = num_ctx_chunk
        self._default_num_predict = min_num_predict
        self._default_num_ctx = min_num_ctx
        self._tls = threading.local()
        self._stats_lock = threading.Lock()
        self._load_durations = {}
        self._num_ctx_stats = {}
        logging.debug(
            f"AutoContextClient: initialized with min_num_predict={min_num_predict}, max_num_predict={max_num_predict}, num_predict_growth={num_predict_growth}, num_predict_shrinkage={num_predict_shrinkage}"
        )
        logging.debug(
            f"AutoContextClient: initialized with min_num_ctx={min_num_ctx}, max_num_ctx={max_num_ctx}, num_ctx_growth={num_ctx_growth}, num_ctx_shrinkage={num_ctx_shrinkage}"
        )

    @property
    def _num_predict(self):
        if not hasattr(self._tls, "num_predict"):
            self._tls.num_predict = self._default_num_predict
        return self._tls.num_predict

    @_num_predict.setter
    def _num_predict(self, value):
        self._tls.num_predict = value

    @property
    def _num_ctx(self):
        if not hasattr(self._tls, "num_ctx"):
            self._tls.num_ctx = self._default_num_ctx
        return self._tls.num_ctx

    @_num_ctx.setter
    def _num_ctx(self, value):
        self._tls.num_ctx = value

    def _roundupToChunk(self, n: int, chunk_size: int):
        """Rounds up `n` to the next multiple of `chunk_size`."""
        return (n + chunk_size - 1) // chunk_size * chunk_size

    def _estimate_num_ctx_for_messages(self, messages: list) -> int:
        """Estimate the number of context tokens required for a list of messages.

        This is a rough estimate based on the length of the message contents.
        It assumes an average token length of 4 characters and adds some overhead
        for message metadata.

        Args:
            messages (list): List of message dicts, each with a 'content' field.

        Returns:
            int: Estimated number of context tokens.
        """
        total_length = 0
        for message in messages:
            content = message.get("content", "")
            total_length += len(content) + 20  # Add some overhead for metadata
        # Assume average token length of 4 characters
        return total_length // 4

    def _estimate_num_ctx_for_call(self, prompt: str, messages: list) -> int:
        """Estimate the number of context tokens required for a call.

        At this stage, messages could be empty, in which case we use the prompt.

        Returns:
            int: Estimated number of context tokens.
        """
        if not messages:
            messages = [{"role": "user", "content": prompt}]
        return self._estimate_num_ctx_for_messages(messages)

    def _update_num_ctx_stats(self, num_ctx, tokens_per_second, num_tokens):
        """Update statistics for the given context size.

        Maintains a per-context-size record that tracks how many calls have used
        that size and the running averages of the number of tokens processed per
        second and the total number of tokens generated.  These statistics are
        later used by the client to make informed decisions about increasing or
        decreasing the context window.

        Args:
            num_ctx (int): The context size (in tokens) used for the call.
            tokens_per_second (float): Throughput measured for the call.
            num_tokens (int): Total number of tokens produced by the call.

        Returns:
            None"""
        with self._stats_lock:
            # Initialize stats for num_ctx if it doesn't exist
            if num_ctx not in self._num_ctx_stats:
                self._num_ctx_stats[num_ctx] = {
                    "count": 0,
                    "avg_tokens_per_second": 0.0,
                    "avg_num_tokens": 0.0,
                }
            # Get current stats
            stats = self._num_ctx_stats[num_ctx]
            current_count = stats["count"]
            current_avg_tps = stats["avg_tokens_per_second"]
            current_avg_tokens = stats["avg_num_tokens"]
            # Update count and incremental averages
            stats["count"] += 1
            stats["avg_tokens_per_second"] = (
                current_avg_tps * current_count + tokens_per_second
            ) / stats["count"]
            stats["avg_num_tokens"] = (
                current_avg_tokens * current_count + num_tokens
            ) / stats["count"]
            logging.debug(self._num_ctx_stats)

    _MAX_RETRIES = 10

    def call(self, *args, _retry_depth: int = 0, **kwargs) -> ThinkResponse:
        """
        Overrides Client.call to automatically adjust num_predict and num_ctx
        based on the response from the model.

        It will repeat the call if the done_reason is "length", indicating that the model
        hit the limit for either num_predict or num_ctx.
        """
        if "options" not in kwargs:
            kwargs["options"] = {}
        # immediately increase num_ctx if the prompt/messages are larger than current num_ctx
        # this avoids unnecessary retries when the initial prompt is already large
        estimated_num_ctx = self._estimate_num_ctx_for_call(
            kwargs.get("prompt", ""), kwargs.get("messages", [])
        )
        if estimated_num_ctx + self._num_predict > self._num_ctx:
            old_num_ctx = self._num_ctx
            self._num_ctx = self._roundupToChunk(
                estimated_num_ctx + self._num_predict, self.num_ctx_chunk
            )
            logging.debug(
                f"AutoContextClient: To accommodate prompt/messages + num_predict growing num_ctx from {old_num_ctx} to {self._num_ctx}"
            )
        kwargs["options"]["num_predict"] = int(self._num_predict)
        kwargs["options"]["num_ctx"] = int(self._num_ctx)
        response = super().call(*args, **kwargs)
        prompt_tokens = response.get("prompt_eval_count", 0)
        eval_tokens = response.get("eval_count", 0)
        if not isinstance(prompt_tokens, int) or not isinstance(eval_tokens, int):
            logging.warning(
                f"AutoContextClient: Non-integer token counts in response: prompt_eval_count={prompt_tokens}, eval_count={eval_tokens}"
            )
            # quit early to avoid further issues
            return response
        total_tokens = prompt_tokens + eval_tokens
        load_duration = (
            response.get("load_duration") or 0
        ) / 1000000000  # convert ns to s
        prompt_eval_duration = response.get("prompt_eval_duration") or 0
        eval_duration = response.get("eval_duration") or 0
        if not isinstance(prompt_eval_duration, int) or not isinstance(
            eval_duration, int
        ):
            logging.warning(
                f"AutoContextClient: Non-integer durations in response: prompt_eval_duration={prompt_eval_duration}, eval_duration={eval_duration}"
            )
            # quit early to avoid further issues
            return response
        total_duration = (
            prompt_eval_duration + eval_duration
        ) / 1000000000  # convert ns to s
        tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
        logging.debug(
            f"AutoContextClient: total tokens {total_tokens},  load duration {load_duration:0.2f}s, inference duration {total_duration:0.2f}s tokens/s {tokens_per_second:0.2f}"
        )
        with self._stats_lock:
            self._load_durations[self._num_ctx] = max(
                self._load_durations.get(self._num_ctx, 0), load_duration
            )
        self._update_num_ctx_stats(self._num_ctx, tokens_per_second, total_tokens)
        if response.get("done_reason") == "length":
            grew_something = False
            old_num_ctx = self._num_ctx
            if total_tokens >= self._num_ctx * 0.9:
                self._num_ctx = min(
                    self.max_num_ctx,
                    self._roundupToChunk(
                        int(self._num_ctx * self.num_ctx_growth), self.num_ctx_chunk
                    ),
                )
                if self._num_ctx > old_num_ctx:
                    grew_something = True
                    logging.info(
                        f"AutoContextClient: Response truncated and prompt_tokens is high, growing num_ctx from {old_num_ctx} to {self._num_ctx}"
                    )
            old_num_predict = self._num_predict
            if eval_tokens >= self._num_predict * 0.9:
                self._num_predict = int(
                    min(
                        self.max_num_predict,
                        self._num_ctx,
                        self._roundupToChunk(
                            int(self._num_predict * self.num_predict_growth),
                            self.num_predict_chunk,
                        ),
                    )
                )
                if self._num_predict > old_num_predict:
                    grew_something = True
                    logging.info(
                        f"AutoContextClient: Response truncated and eval_tokens is high, growing num_predict from {old_num_predict} to {self._num_predict}"
                    )
            if not grew_something:
                # Fallback growth attempt for num_predict, as it's the most common cause of 'length' issues.
                old_num_predict = self._num_predict
                self._num_predict = int(
                    min(
                        self.max_num_predict,
                        self._num_ctx,
                        self._roundupToChunk(
                            int(self._num_predict * self.num_predict_growth),
                            self.num_predict_chunk,
                        ),
                    )
                )
                if self._num_predict > old_num_predict:
                    grew_something = True
                    logging.error(
                        f"AutoContextClient: Response truncated (fallback), growing num_predict from {old_num_predict} to {self._num_predict}"
                    )
            if grew_something and _retry_depth < self._MAX_RETRIES:
                return self.call(*args, _retry_depth=_retry_depth + 1, **kwargs)
            else:
                if _retry_depth >= self._MAX_RETRIES:
                    logging.warning(
                        f"AutoContextClient: Response truncated, but reached max retries ({self._MAX_RETRIES}). Returning response."
                    )
                else:
                    logging.warning(
                        "AutoContextClient: Response truncated, but already at max context/predict size. Returning response."
                    )
                return response
        # no shrinkage for now
        # else:
        #     self._num_predict = int(max(
        #         self.min_num_predict, self._num_predict * self.num_predict_shrinkage
        #     ))
        # if total_tokens < self._num_ctx * 0.5:
        #     self._num_ctx = int(max(
        #         self.min_num_ctx, self._num_ctx * self.num_ctx_shrinkage)
        #     )
        # keep num_predict <= num_ctx
        self._num_predict = int(min(self._num_predict, self._num_ctx))
        return response

    def stats(self) -> dict:
        """Returns the client's load durations and context size statistics."""
        return {
            "load_durations": self._load_durations,
            "ctx_stats": self._num_ctx_stats,
        }


__all__ = ["AutoContextClient"]
