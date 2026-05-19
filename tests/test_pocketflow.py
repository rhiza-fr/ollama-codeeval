"""Tests for sync pocketflow classes: BaseNode, Node, BatchNode, Flow, BatchFlow."""

import warnings

import pytest

from ollama_codeeval.pocketflow import (
    BaseNode,
    BatchFlow,
    BatchNode,
    Flow,
    Node,
    _ConditionalTransition,
)


class TestBaseNode:
    def test_set_params(self):
        node = BaseNode()
        node.set_params({"key": "value"})
        assert node.params == {"key": "value"}

    def test_next_registers_default_successor(self):
        n1, n2 = BaseNode(), BaseNode()
        result = n1.next(n2)
        assert result is n2
        assert n1.successors["default"] is n2

    def test_next_custom_action(self):
        n1, n2 = BaseNode(), BaseNode()
        n1.next(n2, action="done")
        assert n1.successors["done"] is n2

    def test_next_overwrite_warns(self):
        n1, n2, n3 = BaseNode(), BaseNode(), BaseNode()
        n1.next(n2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n1.next(n3)
        assert any("Overwriting" in str(x.message) for x in w)
        assert n1.successors["default"] is n3

    def test_prep_returns_none(self):
        assert BaseNode().prep({}) is None

    def test_exec_returns_none(self):
        assert BaseNode().exec(None) is None

    def test_post_returns_none(self):
        assert BaseNode().post({}, None, None) is None

    def test_run_warns_with_successors(self):
        n1, n2 = BaseNode(), BaseNode()
        n1.next(n2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n1.run({})
        assert any("won't run successors" in str(x.message) for x in w)

    def test_run_without_successors_returns_none(self):
        assert BaseNode().run({}) is None

    def test_rshift_connects_nodes(self):
        n1, n2 = BaseNode(), BaseNode()
        result = n1 >> n2
        assert result is n2
        assert n1.successors["default"] is n2

    def test_sub_returns_conditional_transition(self):
        node = BaseNode()
        ct = node - "ok"
        assert isinstance(ct, _ConditionalTransition)
        assert ct.action == "ok"
        assert ct.src is node

    def test_sub_non_string_raises(self):
        with pytest.raises(TypeError):
            _ = BaseNode().__sub__(42)  # ty: ignore


class TestConditionalTransition:
    def test_rshift_registers_successor(self):
        n1, n2 = BaseNode(), BaseNode()
        ct = _ConditionalTransition(src=n1, action="done")
        result = ct >> n2
        assert result is n2
        assert n1.successors["done"] is n2


class TestNode:
    def test_exec_success_first_try(self):
        class MyNode(Node):
            def exec(self, prep_res):
                return "result"

        assert MyNode()._exec(None) == "result"

    def test_exec_retry_then_success(self):
        class MyNode(Node):
            def __init__(self):
                super().__init__(max_retries=3)
                self.attempts = 0

            def exec(self, prep_res):
                self.attempts += 1
                if self.attempts < 3:
                    raise ValueError("fail")
                return "success"

        node = MyNode()
        assert node._exec(None) == "success"
        assert node.attempts == 3

    def test_exec_fallback_called_on_exhaustion(self):
        class MyNode(Node):
            def __init__(self):
                super().__init__(max_retries=2)

            def exec(self, prep_res):
                raise ValueError("always fails")

            def exec_fallback(self, prep_res, exc):
                return "fallback"

        assert MyNode()._exec(None) == "fallback"

    def test_exec_fallback_raises_by_default(self):
        class MyNode(Node):
            def exec(self, prep_res):
                raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            MyNode()._exec(None)

    def test_exec_with_wait_sleeps(self):
        from unittest.mock import patch

        class MyNode(Node):
            def __init__(self):
                super().__init__(max_retries=2, wait=1)
                self.attempts = 0

            def exec(self, prep_res):
                self.attempts += 1
                if self.attempts == 1:
                    raise ValueError("first fail")
                return "ok"

        node = MyNode()
        with patch("ollama_codeeval.pocketflow.time.sleep") as mock_sleep:
            result = node._exec(None)
        mock_sleep.assert_called_once_with(1)
        assert result == "ok"

    def test_run_executes_prep_exec_post(self):
        class MyNode(Node):
            def prep(self, shared):
                shared["prep"] = True
                return "prep_result"

            def exec(self, prep_res):
                return prep_res + "_exec"

            def post(self, shared, prep_res, exec_res):
                shared["post"] = exec_res
                return "done"

        shared = {}
        result = MyNode().run(shared)
        assert shared["prep"] is True
        assert shared["post"] == "prep_result_exec"
        assert result == "done"


class TestBatchNode:
    def test_exec_processes_list(self):
        class MyBatch(BatchNode):
            def exec(self, prep_res):
                return prep_res * 2

        assert MyBatch()._exec([1, 2, 3]) == [2, 4, 6]

    def test_exec_empty_list(self):
        class MyBatch(BatchNode):
            def exec(self, prep_res):
                return prep_res

        assert MyBatch()._exec([]) == []

    def test_exec_none_treated_as_empty(self):
        class MyBatch(BatchNode):
            def exec(self, prep_res):
                return prep_res

        assert MyBatch()._exec(None) == []


class TestFlow:
    def test_simple_two_node_flow(self):
        class NodeA(Node):
            def post(self, shared, prep_res, exec_res):
                shared["a"] = True
                return "default"

        class NodeB(Node):
            def post(self, shared, prep_res, exec_res):
                shared["b"] = True

        a, b = NodeA(), NodeB()
        a >> b
        shared = {}
        Flow(start=a).run(shared)
        assert shared["a"] is True
        assert shared["b"] is True

    def test_start_method(self):
        n = BaseNode()
        flow = Flow()
        result = flow.start(n)
        assert result is n
        assert flow.start_node is n

    def test_get_next_node_found(self):
        n1, n2 = BaseNode(), BaseNode()
        n1.next(n2)
        assert Flow().get_next_node(n1, "default") is n2

    def test_get_next_node_not_found_warns(self):
        n1, n2 = BaseNode(), BaseNode()
        n1.next(n2, "ok")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = Flow().get_next_node(n1, "fail")
        assert result is None
        assert any("Flow ends" in str(x.message) for x in w)

    def test_get_next_node_no_successors_no_warning(self):
        n1 = BaseNode()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = Flow().get_next_node(n1, "default")
        assert result is None
        assert not w

    def test_conditional_transition_routing(self):
        class Router(Node):
            def __init__(self, action):
                super().__init__()
                self._action = action

            def post(self, shared, prep_res, exec_res):
                return self._action

        class Writer(Node):
            def __init__(self, label):
                super().__init__()
                self._label = label

            def post(self, shared, prep_res, exec_res):
                shared["result"] = self._label

        router = Router("b")
        a, b = Writer("a"), Writer("b")
        router - "a" >> a
        router - "b" >> b
        shared = {}
        Flow(start=router).run(shared)
        assert shared["result"] == "b"

    def test_params_passed_to_nodes(self):
        class ParamNode(Node):
            def post(self, shared, prep_res, exec_res):
                shared["params"] = self.params

        n = ParamNode()
        flow = Flow(start=n)
        flow.set_params({"x": 1})
        shared = {}
        flow.run(shared)
        assert shared["params"] == {"x": 1}

    def test_action_none_uses_default(self):
        class NodeA(Node):
            def post(self, shared, prep_res, exec_res):
                return None  # None should route to "default"

        class NodeB(Node):
            def post(self, shared, prep_res, exec_res):
                shared["reached"] = True

        a, b = NodeA(), NodeB()
        a >> b
        shared = {}
        Flow(start=a).run(shared)
        assert shared["reached"] is True


class TestBatchFlow:
    def test_runs_for_each_batch_item(self):
        class WriteNode(Node):
            def post(self, shared, prep_res, exec_res):
                shared.setdefault("results", []).append(self.params.get("value"))

        class MyBatchFlow(BatchFlow):
            def prep(self, shared):
                return [{"value": 1}, {"value": 2}, {"value": 3}]

        flow = MyBatchFlow(start=WriteNode())
        shared = {}
        flow.run(shared)
        assert sorted(shared["results"]) == [1, 2, 3]

    def test_empty_batch(self):
        class WriteNode(Node):
            def post(self, shared, prep_res, exec_res):
                shared.setdefault("results", []).append(1)

        class MyBatchFlow(BatchFlow):
            def prep(self, shared):
                return []

        shared = {}
        MyBatchFlow(start=WriteNode()).run(shared)
        assert "results" not in shared

    def test_none_batch_treated_as_empty(self):
        class WriteNode(Node):
            def post(self, shared, prep_res, exec_res):
                shared.setdefault("results", []).append(1)

        class MyBatchFlow(BatchFlow):
            def prep(self, shared):
                return None

        shared = {}
        MyBatchFlow(start=WriteNode()).run(shared)
        assert "results" not in shared
