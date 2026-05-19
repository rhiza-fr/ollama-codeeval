# src/ollama_codeeval/contamination/prompts.py

MORPH_SOLUTION_PROMPT = """\
You are given a Python function. Rewrite it to solve an isomorphic problem in a completely different domain.

Rules:
- Change the real-world metaphor entirely (e.g. numbers → sensor readings, strings → product codes, lists of floats → sequences of measurements)
- Rename the function to a new snake_case name that fits the new domain
- Rename all parameters and local variables to match the new domain
- Update type hints and docstring to match the new domain
- Keep EXACTLY the same algorithmic structure — do not change the logic
- Output ONLY the complete Python function (no markdown, no explanation)

Original entry point: {entry_point}

Original prompt stub:
{prompt}

Canonical solution:
{canonical_solution}
"""

FUZZ_INPUTS_PROMPT = """\
Generate 8 diverse Python test inputs for the following function.
Cover: typical cases, edge cases (empty/minimal input), boundary values, and varied sizes.
Output ONLY a Python list of tuples, where each tuple contains the positional arguments for one call.
No explanation. No markdown. Just the list literal.

Example output format (8 tuples): [([1, 2], 0.5), ([], 0.1), ([3], 1.0), ([1, 2, 3], 0.2), ([5], 0.0), ([1, 1], 0.5), ([10, 20, 30, 40], 5.0), ([2, 2], 0.1)]

Function:
{morphed_function}
"""

WRITE_QUESTION_PROMPT = """\
Based only on the function signature and input/output examples below, write a complete Python function stub.
The stub must include a Google-style docstring that describes what the function does, its parameters, return type, and examples.
Do NOT use any knowledge of the original HumanEval dataset. Derive the description purely from the examples.
Output ONLY the Python stub (def line + docstring + pass). No markdown. No explanation.

Function name: {entry_point}
Function signature: def {entry_point}{signature_tail}

Input/output examples:
{examples}
"""
