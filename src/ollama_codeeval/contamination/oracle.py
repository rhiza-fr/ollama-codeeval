# src/ollama_codeeval/contamination/oracle.py
import ast

from ollama_codeeval.sandbox import Sandbox


def run_oracle(
    solution_code: str,
    entry_point: str,
    inputs: list,
    sandbox: Sandbox,
) -> list[tuple]:
    """Run solution_code against each input tuple using the sandbox as oracle.

    Returns a list of (args_tuple, result) pairs for inputs that ran without error.
    Inputs that raise exceptions inside the sandbox are silently dropped.
    """
    oracle_script = _build_oracle_script(solution_code, entry_point, inputs)
    result = sandbox.run(oracle_script)
    stdout: str = str(result["stdout"])
    if result["exit_code"] != 0 or not stdout.strip():
        return []
    return _parse_oracle_output(stdout)


def _build_oracle_script(solution_code: str, entry_point: str, inputs: list) -> str:
    return f"""\
{solution_code}

_inputs = {repr(inputs)}
for _args in _inputs:
    try:
        _result = {entry_point}(*_args)
        print(repr({{"ok": True, "args": _args, "result": _result}}))
    except Exception as _e:
        print(repr({{"ok": False, "args": _args, "result": None, "error": str(_e)}}))
"""


def _parse_oracle_output(stdout: str) -> list[tuple]:
    pairs = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if record.get("ok"):
            args = record["args"]
            result = record["result"]
            pairs.append((tuple(args) if isinstance(args, list) else args, result))
    return pairs
