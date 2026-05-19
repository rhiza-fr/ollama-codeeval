"""Utilities for running ruff format/fix locally as a subprocess."""

import os
import re
import subprocess
import tempfile


def ruff_fix(code: str) -> tuple[str, int, str]:
    """Run ruff format + ruff check --fix on code locally.

    Writes code to a temp file, runs ruff format then ruff check --fix,
    reads back the fixed code, and returns any remaining errors.

    Args:
        code: The Python source code to fix.

    Returns:
        (fixed_code, exit_code, remaining_errors) where exit_code == 0 means
        ruff found no remaining issues after auto-fix.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(code)
        tmp.close()

        # Step 1: ruff format
        subprocess.run(
            ["ruff", "format", tmp.name],
            capture_output=True,
            encoding="utf-8",
        )

        # Step 2: ruff check --fix, only real errors
        chk = subprocess.run(
            ["ruff", "check", "--fix", "--select", "E9,F63,F7,F82", tmp.name],
            capture_output=True,
            encoding="utf-8",
        )

        # Read back the fixed code
        with open(tmp.name, encoding="utf-8") as f:
            fixed = f.read()

        fixed_code = fixed if fixed else code
        remaining_errors = (
            (chk.stdout or "") + (chk.stderr or "")
            if chk.returncode != 0
            else ""
        )
        return (fixed_code, chk.returncode, remaining_errors)
    finally:
        os.unlink(tmp.name)


def compact_ruff_error(err: str) -> str:
    """Extract the first ruff error message up to the squiggly-line pointer.

    Keeps diagnostics concise for LLM prompting by truncating after the first
    error context block.

    Args:
        err: Full ruff error/warning output.

    Returns:
        The trimmed error containing only the first error and its context.
        If no squiggly-line pointer is found, returns the original string.
    """
    err_reg = re.compile(r"(.*?)(\n\s+\|\s+\^{2,}[^\n]*\n)", re.MULTILINE | re.DOTALL)
    match = re.search(err_reg, err)
    if match:
        return match.group(1) + match.group(2)
    return err
