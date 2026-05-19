"""Tests for the Sandbox class."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_session_cls():
    """Patch SandboxSession and return the mock instance."""
    with patch("ollama_codeeval.sandbox.SandboxSession") as MockCls:
        instance = MockCls.return_value
        instance.run.return_value = MagicMock(stdout="out", stderr="", exit_code=0)
        yield MockCls, instance


class TestSandboxInit:
    def test_opens_session_on_init(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        MockCls, instance = mock_session_cls
        sb = Sandbox(lang="python", image="test-image", execution_timeout=5.0)
        MockCls.assert_called_once_with(
            lang="python", image="test-image", execution_timeout=5.0
        )
        instance.open.assert_called_once()
        assert sb.lang == "python"
        assert sb.image == "test-image"
        assert sb.execution_timeout == 5.0
        assert sb._closed is False
        sb.close()

    def test_default_timeout(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        MockCls, instance = mock_session_cls
        sb = Sandbox(lang="python", image="test-image")
        assert sb.execution_timeout == 10.0
        sb.close()


class TestSandboxRun:
    def test_run_returns_stdout_stderr_exit_code(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        _, instance = mock_session_cls
        instance.run.return_value = MagicMock(stdout="hello", stderr="", exit_code=0)
        sb = Sandbox(lang="python", image="test-image")
        result = sb.run("print('hello')")
        assert result == {"stdout": "hello", "stderr": "", "exit_code": 0}
        sb.close()

    def test_run_handles_timeout_error(self, mock_session_cls):
        from llm_sandbox.exceptions import SandboxTimeoutError

        from ollama_codeeval.sandbox import Sandbox

        _, instance = mock_session_cls
        instance.run.side_effect = SandboxTimeoutError("timed out")
        sb = Sandbox(lang="python", image="test-image")
        result = sb.run("while True: pass")
        assert result["exit_code"] == 1
        assert "timed out" in str(result["stderr"])
        assert result["stdout"] == ""
        sb.close()

    def test_run_handles_generic_exception(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        _, instance = mock_session_cls
        instance.run.side_effect = RuntimeError("docker error")
        sb = Sandbox(lang="python", image="test-image")
        result = sb.run("code")
        assert result["exit_code"] == 1
        assert "docker error" in str(result["stderr"])
        sb.close()


class TestSandboxClose:
    def test_close_marks_closed(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        _, instance = mock_session_cls
        sb = Sandbox(lang="python", image="test-image")
        sb.close()
        instance.close.assert_called_once()
        assert sb._closed is True

    def test_close_idempotent(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        _, instance = mock_session_cls
        sb = Sandbox(lang="python", image="test-image")
        sb.close()
        sb.close()  # second call should not invoke session.close again
        instance.close.assert_called_once()


class TestSandboxContextManager:
    def test_enter_returns_self(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        _, instance = mock_session_cls
        sb = Sandbox(lang="python", image="test-image")
        with sb as s:
            assert s is sb
        instance.close.assert_called()

    def test_exit_closes_session(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        _, instance = mock_session_cls
        with Sandbox(lang="python", image="test-image"):
            pass
        instance.close.assert_called()

    def test_exit_returns_false_does_not_suppress_exceptions(self, mock_session_cls):
        from ollama_codeeval.sandbox import Sandbox

        _, instance = mock_session_cls
        with pytest.raises(ValueError):
            with Sandbox(lang="python", image="test-image"):
                raise ValueError("test error")
