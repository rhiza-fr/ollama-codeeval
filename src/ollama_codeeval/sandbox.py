"""A sandboxed environment for executing code in a controlled manner."""

import atexit
import threading

from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SandboxTimeoutError


class Sandbox:
    """Runs code in a sandboxed environment with specified language and image.
    Args:
        code (str): The code to execute.
    Returns:
        dict[str, str | int]: The output of the executed code."""

    def __init__(self, lang: str, image: str, execution_timeout: float = 10.0):
        """Initialize a Sandbox instance with the specified language, image, and execution timeout.
        Args:             lang (str): The programming language to use.
        image (str): The Docker image to use for the sandbox.
        execution_timeout (float, optional): The maximum time in seconds to allow code execution. Defaults to 10.0."""
        self.lang = lang
        self.image = image
        self.execution_timeout = execution_timeout
        self._closed = False
        self._lock = threading.Lock()
        self.session = SandboxSession(
            lang=lang, image=image, execution_timeout=execution_timeout
        )
        self.session.open()
        atexit.register(self.close)

    def run(self, code: str) -> dict[str, str | int]:
        """
        Runs the provided code in the sandboxed environment.

        Args:
            code (str): The code to execute.

        Returns:
            The output of the executed code.
        """
        with self._lock:
            try:
                output = self.session.run(code)
                return {
                    "stdout": output.stdout,
                    "stderr": output.stderr,
                    "exit_code": output.exit_code,
                }
            except SandboxTimeoutError as e:
                return {"stdout": "", "stderr": str(e), "exit_code": 1}
            except Exception as e:
                return {"stdout": "", "stderr": str(e), "exit_code": 1}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
        return False

    def close(self):
        """Close the sandbox session."""
        if not self._closed:
            self._closed = True
            self.session.close()


if __name__ == "__main__":
    sandbox = Sandbox(lang="python", image="python-sandbox")
    try:
        result = sandbox.run("print('Hello, World!')")
        print(result)
    finally:
        sandbox.close()
