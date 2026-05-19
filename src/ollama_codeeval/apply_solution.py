"""Applies a generated solution to a problem by cleaning and reconstructing the full code. It uses helper functions to strip thinking tags, extract code from markdown, and dedent code blocks. The function takes a prompt, solution, and entrypoint to produce a cleaned solution."""

# we are applying a generated solution to a problem
# the result from the llm may be:
# 1) a full solution including imports
# 2) a redefinition of just the method
# 3) just the method contents
# 4) one of the above but enclosed in markdown code blocks
import re

# Handle optional carriage return for cross-platform compatibility
MARKDOWN_PYTHON_REGEX = re.compile("```python[23]?[0-9]?\\r?\\n(.*?)```", re.DOTALL)


def _strip_thinking_tags(solution: str) -> str:
    """
    Removes <think> and <response> tags from the solution string.
    It prioritizes content within <response> tags if they exist.
    """
    # First, try to find a <response> block and extract its content.
    response_match = re.search(
        "<response>(?P<content>.*?)</response>", solution, re.DOTALL
    )
    if response_match:
        return response_match.group("content").strip()
    # If no <response> block, check for a <think> block and return what's after it.
    # This handles cases where a model might output thinking without a response wrapper.
    think_match = re.search("<think>.*?</think>(?P<content>.*)", solution, re.DOTALL)
    if think_match:
        return think_match.group("content").strip()
    # If no tags are found, return the original solution.
    return solution


def _extract_from_markdown(solution: str, entrypoint: str) -> str:
    """Extracts Python code from the last markdown block containing the entrypoint."""
    code_blocks = MARKDOWN_PYTHON_REGEX.findall(solution)
    if not code_blocks:
        if "```python" in solution:
            # nemotron often fails to end the code block
            # so we try to salvage it by taking everything after the first ```python
            # it sometimes uses ```python37 or ```python2, so we'll take everything after that line
            parts = re.split("```python[23]?[0-9]?\\r?\\n", solution, maxsplit=1)
            if len(parts) > 1:
                return parts[1].strip()
        return solution
    # Prefer the last code block that contains the entrypoint
    for block in reversed(code_blocks):
        if f"def {entrypoint}" in block:
            return block
    # Fallback to the last found code block if entrypoint is not in any
    return code_blocks[-1]


def _dedent_code(code: str) -> str:
    """
    Dedents a code block by finding the minimum indentation of all non-empty lines
    and removing that amount of indentation from every line.
    """
    lines = code.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return code  # Return original string if no content
    # Find the minimum indentation
    min_indent = min((len(line) - len(line.lstrip()) for line in non_empty_lines))
    if min_indent > 0:
        # Remove the minimum indentation from all lines
        return "\n".join((line[min_indent:] for line in lines))
    return code


def apply_solution(prompt: str, solution: str, entrypoint: str) -> str:
    """
    Applies a generated solution to a problem by cleaning and reconstructing the full code.

    Args:
        prompt: The original problem prompt.
        solution: The generated solution from the model.
        entrypoint: The name of the method to be applied.

    Returns:
        A string containing the full, cleaned, and reconstructed solution.
    """
    solution = _strip_thinking_tags(solution)
    solution = _extract_from_markdown(solution, entrypoint)
    solution = solution.replace("```", "").strip()
    # Only dedent if the solution appears to contain a full function definition.
    # This avoids stripping indentation from function bodies.
    if f"def {entrypoint}" in solution:
        solution = _dedent_code(solution)
    # Extract preamble (imports) and method signature from the prompt
    prompt_match = re.search(
        f"(.*?)(\\bdef\\s+{entrypoint}[^\\n]*\\n)", prompt, re.DOTALL
    )
    preamble = prompt_match.group(1) if prompt_match else ""
    method_definition = prompt_match.group(2) if prompt_match else ""
    # If the solution is just the body of the function, prepend the definition
    if not re.search(f"\\bdef\\s+{entrypoint}\\b", solution):
        if method_definition:
            solution = method_definition + solution
        else:
            # print(f"entrypoint: {entrypoint}")
            # print(f"solution: {solution}")
            # print(f"prompt: {prompt}")
            pass

            # # This case should be rare if the prompt is always well-formed
            # raise ValueError(
            #     f"Neither the solution nor the prompt contains the method definition '{entrypoint}'."
            # )
    # If the preamble (imports) is missing from the solution, add it
    if preamble and preamble.strip() and (preamble.strip() not in solution):
        solution = preamble + solution
    return solution
