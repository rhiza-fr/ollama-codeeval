"""Tests for async pocketflow classes: AsyncNode, AsyncBatchNode, AsyncFlow, etc."""
import asyncio
import warnings

import pytest

from ollama_codeeval.pocketflow import (
    AsyncBatchFlow,
    AsyncBatchNode,
    AsyncFlow,
    AsyncNode,
    AsyncParallelBatchFlow,
    AsyncParallelBatchNode,
    Node,
)


class TestAsyncNode:
    def test_async_exec_success(self):
        class MyAsync(AsyncNode):
            async def exec_async(self, prep_res):
                return prep_res + " done"

        result = asyncio.run(MyAsync()._exec("hello"))
        assert result == "hello done"

    def test_async_exec_retry(self):
        class MyAsync(AsyncNode):
            def __init__(self):
                super().__init__(max_retries=3)
                self.attempts = 0

            async def exec_async(self, prep_res):
                self.attempts += 1
                if self.attempts < 3:
                    raise ValueError("fail")
                return "ok"

        node = MyAsync()
        assert asyncio.run(node._exec(None)) == "ok"
        assert node.attempts == 3

    def test_async_exec_fallback(self):
        class MyAsync(AsyncNode):
            async def exec_async(self, prep_res):
                raise ValueError("always fails")

            async def exec_fallback_async(self, prep_res, exc):
                return "async fallback"

        assert asyncio.run(MyAsync()._exec(None)) == "async fallback"

    def test_async_exec_fallback_raises_by_default(self):
        class MyAsync(AsyncNode):
            async def exec_async(self, prep_res):
                raise ValueError("fail")

        with pytest.raises(ValueError):
            asyncio.run(MyAsync()._exec(None))

    def test_async_run_async_full_lifecycle(self):
        class MyAsync(AsyncNode):
            async def prep_async(self, shared):
                return "prep"

            async def exec_async(self, prep_res):
                return prep_res + "-exec"

            async def post_async(self, shared, prep_res, exec_res):
                shared["result"] = exec_res

        shared = {}
        asyncio.run(MyAsync().run_async(shared))
        assert shared["result"] == "prep-exec"

    def test_run_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Use run_async"):
            AsyncNode()._run({})

    def test_async_run_warns_with_successors(self):
        class MyAsync(AsyncNode):
            async def exec_async(self, prep_res):
                return None

        n1, n2 = MyAsync(), MyAsync()
        n1 >> n2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            asyncio.run(n1.run_async({}))
        assert any("won't run successors" in str(x.message) for x in w)

    def test_async_exec_with_wait(self):
        from unittest.mock import AsyncMock, patch

        class MyAsync(AsyncNode):
            def __init__(self):
                super().__init__(max_retries=2, wait=1)
                self.attempts = 0

            async def exec_async(self, prep_res):
                self.attempts += 1
                if self.attempts == 1:
                    raise ValueError("fail")
                return "ok"

        node = MyAsync()

        async def run():
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await node._exec(None)
                mock_sleep.assert_called_once_with(delay=1)
                return result

        assert asyncio.run(run()) == "ok"

    def test_async_prep_and_post_defaults(self):
        async def run():
            node = AsyncNode()
            p = await node.prep_async({})
            e = await node.exec_async(None)
            r = await node.post_async({}, None, None)
            return p, e, r

        p, e, r = asyncio.run(run())
        assert p is None
        assert e is None
        assert r is None


class TestAsyncBatchNode:
    def test_processes_list(self):
        class MyAsyncBatch(AsyncBatchNode):
            async def exec_async(self, prep_res):
                return prep_res * 2

        result = asyncio.run(MyAsyncBatch()._exec([1, 2, 3]))
        assert result == [2, 4, 6]

    def test_none_treated_as_empty(self):
        class MyAsyncBatch(AsyncBatchNode):
            async def exec_async(self, prep_res):
                return prep_res

        assert asyncio.run(MyAsyncBatch()._exec(None)) == []


class TestAsyncParallelBatchNode:
    def test_processes_list_in_parallel(self):
        class MyParallelBatch(AsyncParallelBatchNode):
            async def exec_async(self, prep_res):
                return prep_res * 3

        result = asyncio.run(MyParallelBatch()._exec([1, 2, 3]))
        assert sorted(result) == [3, 6, 9]

    def test_none_treated_as_empty(self):
        class MyParallelBatch(AsyncParallelBatchNode):
            async def exec_async(self, prep_res):
                return prep_res

        assert asyncio.run(MyParallelBatch()._exec(None)) == []


class TestAsyncFlow:
    def test_simple_async_flow(self):
        class NodeA(AsyncNode):
            async def post_async(self, shared, prep_res, exec_res):
                shared["a"] = True
                return "default"

        class NodeB(AsyncNode):
            async def post_async(self, shared, prep_res, exec_res):
                shared["b"] = True

        a, b = NodeA(), NodeB()
        a >> b
        shared = {}
        asyncio.run(AsyncFlow(start=a).run_async(shared))
        assert shared["a"] is True
        assert shared["b"] is True

    def test_async_flow_with_sync_node(self):
        class SyncN(Node):
            def post(self, shared, prep_res, exec_res):
                shared["sync"] = True
                return "default"

        class AsyncN(AsyncNode):
            async def post_async(self, shared, prep_res, exec_res):
                shared["async"] = True

        sync, async_n = SyncN(), AsyncN()
        sync >> async_n
        shared = {}
        asyncio.run(AsyncFlow(start=sync).run_async(shared))
        assert shared["sync"] is True
        assert shared["async"] is True

    def test_async_flow_post_returns_last_action(self):
        class NodeA(AsyncNode):
            async def post_async(self, shared, prep_res, exec_res):
                return "final"

        flow = AsyncFlow(start=NodeA())
        result = asyncio.run(flow.run_async({}))
        assert result == "final"


class TestAsyncBatchFlow:
    def test_runs_for_each_item(self):
        class WriteNode(AsyncNode):
            async def post_async(self, shared, prep_res, exec_res):
                shared.setdefault("results", []).append(self.params.get("value"))

        class MyAsyncBatch(AsyncBatchFlow):
            async def prep_async(self, shared):
                return [{"value": 1}, {"value": 2}]

        shared = {}
        asyncio.run(MyAsyncBatch(start=WriteNode()).run_async(shared))
        assert sorted(shared["results"]) == [1, 2]

    def test_none_prep_treated_as_empty(self):
        class WriteNode(AsyncNode):
            async def post_async(self, shared, prep_res, exec_res):
                shared.setdefault("results", []).append(1)

        class MyAsyncBatch(AsyncBatchFlow):
            async def prep_async(self, shared):
                return None

        shared = {}
        asyncio.run(MyAsyncBatch(start=WriteNode()).run_async(shared))
        assert "results" not in shared


class TestAsyncParallelBatchFlow:
    def test_runs_in_parallel(self):
        class WriteNode(AsyncNode):
            async def post_async(self, shared, prep_res, exec_res):
                shared.setdefault("results", []).append(self.params.get("value"))

        class MyParallelBatch(AsyncParallelBatchFlow):
            async def prep_async(self, shared):
                return [{"value": "x"}, {"value": "y"}]

        shared = {}
        asyncio.run(MyParallelBatch(start=WriteNode()).run_async(shared))
        assert sorted(shared["results"]) == ["x", "y"]
