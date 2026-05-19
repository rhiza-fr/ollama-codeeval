"""This module provides functions to generate test code with IPython pytest integration, extract traceback details from pytest logs, and clean test results. The functions help improve assertion visibility and error handling in test outputs."""


def generate_test(generated_code: str, test: str, entrypoint: str) -> str:
    """Add ipython pytest around code to get better assertion results"""
    return (
        "\nimport sys\nimport pytest\n\n"
        + f"{generated_code}\n\n{test}\n"
        + f"\ndef test():\n    check({entrypoint})\n\nif __name__ == '__main__':\n    sys.exit(pytest.main(['-q', '--tb=short', __file__]))\n"
    )


def extract_traceback_details(log_text: str) -> str:
    """
    Extracts the detailed traceback section from a pytest failure log.

    This function finds the block of text that starts after a line of underscores
    and ends before a line containing a file path.

    Args:
        log_text: A string containing the pytest failure output.

    Returns:
        A string containing the isolated traceback details, or an empty
        string if the section is not found.
    """
    lines = log_text.splitlines()
    capturing = False
    extracted_lines = []
    for line in lines:
        # A long line of underscores indicates the start of the detailed traceback
        if line.endswith("in check"):
            capturing = True
            continue
        # Remove the summary
        if capturing and line.startswith("==========="):
            break
        if capturing:
            if not line.find("Use -v to get more diff") >= 0:
                extracted_lines.append(line)
    # We remove the first empty line if it exists
    if extracted_lines and (not extracted_lines[0]):
        extracted_lines.pop(0)
    return "\n".join(extracted_lines).strip()


def clean_test_output(test_result: str) -> str:
    """Return cleaned test output details from a test result string. If no failures are found, return an empty string. Otherwise, extract and return traceback details."""
    if "FAILURES" not in test_result:
        return ""
    return extract_traceback_details(test_result)


def consume_test_result(result: dict[str, str | int]) -> dict[str, str | int]:
    """Cleanup output of pytest, ignoring irrelevant and moving errors to stderr"""
    if len(str(result["stderr"])) > 0:
        return result
    tb = clean_test_output(str(result["stdout"]))
    if not tb and result["exit_code"] != 0:
        # Process exited non-zero but stdout had no parseable traceback.
        # Happens when Docker drops pytest output (e.g. reentrancy suppresses capture).
        tb = "Test failed (no traceback captured)"
    return {"stdout": "", "stderr": tb, "exit_code": 1 if tb else 0}
