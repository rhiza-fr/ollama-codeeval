# src/ollama_codeeval/contamination/variant_utils.py
import ast
import re


def parse_input_list(llm_output: str) -> list:
    """Extract and parse a Python list literal from LLM output.

    Searches for the outermost [...] block and evaluates it safely.
    Raises ValueError if no valid list is found.
    """
    start = llm_output.find("[")
    if start == -1:
        raise ValueError(f"No list found in LLM output: {llm_output!r}")
    depth = 0
    for i, ch in enumerate(llm_output[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = llm_output[start : i + 1]
                try:
                    return ast.literal_eval(candidate)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(
                        f"Could not parse input list: {candidate!r}"
                    ) from exc
    raise ValueError(f"Unmatched bracket in LLM output: {llm_output!r}")


MAX_REPR_LEN = 200


def build_check_function(entry_point: str, pairs: list[tuple]) -> str:
    """Build a HumanEval-compatible check(candidate) test string.

    Pairs whose expected-value repr exceeds MAX_REPR_LEN characters are silently
    dropped to avoid embedding enormous string literals in the test.

    Args:
        entry_point: Function name.
        pairs: List of (args_tuple, expected_result) pairs.
    """
    if not pairs:
        raise ValueError("Cannot build check function with no pairs")
    lines = ["def check(candidate):"]
    for args, expected in pairs:
        expected_repr = repr(expected)
        if len(expected_repr) > MAX_REPR_LEN:
            continue
        args_str = ", ".join(repr(a) for a in args)
        lines.append(f"    assert candidate({args_str}) == {expected_repr}")
    if len(lines) == 1:
        raise ValueError("All pairs had oversized expected values; no assertions generated")
    return "\n".join(lines) + "\n"


def extract_function_name(code: str) -> str | None:
    """Return the name of the first function defined in code, or None."""
    m = re.search(r"\bdef\s+(\w+)\s*\(", code)
    return m.group(1) if m else None


_TYPING_NAMES = {"List", "Dict", "Tuple", "Optional", "Set", "FrozenSet", "Union", "Any", "Callable", "Sequence", "Iterator", "Generator", "Type"}


def ensure_typing_imports(code: str) -> str:
    """Prepend 'from typing import ...' if the code uses typing names without importing them."""
    already_imported = re.search(r"^from typing import", code, re.MULTILINE)
    if already_imported:
        return code
    needed = sorted(name for name in _TYPING_NAMES if re.search(rf"\b{name}\b", code))
    if not needed:
        return code
    return f"from typing import {', '.join(needed)}\n\n\n{code}"


def contamination_score(pairs: list[tuple[bool, bool]]) -> float:
    """Compute contamination score from (pass_original, pass_variant) pairs.

    Returns P(pass_orig AND fail_var) - P(fail_orig AND pass_var).
    Positive values indicate contamination; near-zero means clean.
    """
    if not pairs:
        return 0.0
    n = len(pairs)
    forward = sum(1 for orig, var in pairs if orig and not var)
    reverse = sum(1 for orig, var in pairs if not orig and var)
    return (forward - reverse) / n
