"""Prompt templates and builders used by the agent nodes."""


def _truncate_code(code: str, max_lines: int = 200) -> str:
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    kept = lines[:max_lines]
    return "\n".join(kept) + f"\n... [{len(lines) - max_lines} lines truncated]"


def build_fix_prompt(generated_prompt: str, code: str, error: str, iterations: list) -> str:
    """Build a prompt asking the LLM to fix a failed solution."""
    prompt = (
        "You were originally asked the question:\n\n"
        + generated_prompt
        + "\n\nYou responded (formatted):\n\n"
        + "```python\n"
        + _truncate_code(code)
        + "\n```\n\n\n\nThis generated errors during testing:\n\n"
        + error
        + "\n\nBefore fixing: re-read the original question as if seeing it for the first time.\n"
        + "The error may mean you misunderstood the task — not that your code has a bug.\n\n"
        + "Trace the failing case: take the input that caused the error and the\n"
        + "expected output. Walk through the docstring step by step with that input.\n"
        + "If your algorithm would not produce that expected output, the algorithm\n"
        + "itself is wrong — not just the code.\n\n"
        + "Check for these common misreadings:\n"
        + "- Inclusive vs exclusive bounds (< vs <=, 'at least N' vs 'more than N')\n"
        + "- 0-indexed vs 1-indexed counting\n"
        + "- Whether the input is sorted or unsorted (do not assume)\n"
        + "- 'Unique': deduplication, OR appears exactly once — verify against examples\n"
        + "- Return type: int vs float, list vs tuple, None vs empty list\n"
        + "- Whether special characters (e.g. '-', '.') count as part of the data\n"
        + "- 'Closest' by absolute difference vs by position\n\n"
        + "Look for constraints that the examples imply but the text does not state.\n\n"
        + "Start your solution with a single comment line: # This function [your plain-English description].\n"
        + "Make sure that description is consistent with every example in the docstring.\n"
        + "Then write the implementation.\n\n"
        + "Please regenerate a solution that solves this test. Do not repeat the exact same solution that caused the error."
    )

   # It is debatable whether it is a good idea, to show the model so many of its own mistakes.
   #  previous_attempts = []
   #  for i, iteration in enumerate(iterations):
   #      msg = iteration.get("message", {}).get("content", "")
   #      if msg:
   #          lines = msg.strip().splitlines()[:10]
   #          previous_attempts.append(f"Attempt {i + 1}:\n" + "\n".join(lines))
   #  if previous_attempts:
   #      prompt += (
   #          "\n\nYour previous attempts that failed:\n"
   #          + "\n\n".join(previous_attempts)
   #          + "\n\nYou MUST try a fundamentally different approach. Do not reuse the same algorithm or logic structure."
   #      )
    return prompt


REWRITE_SYSTEM_PROMPT = """\
You are a preprocessing assistant that rewrites programming questions so a small language model can answer them correctly with minimal confusion.

Your task is to **rewrite, clarify, and structure** the question — NOT to solve it.

### Core Rules

1. **Do not change the task**

   * Preserve the exact goal.
   * Do not add or remove requirements.
   * Do not introduce new behavior.

2. **Eliminate ambiguity**

   * Replace vague words with precise ones.
   * Make implied constraints explicit.
   * Define any technical term that could be interpreted multiple ways.

3. **Make all requirements explicit**

   * Inputs
   * Outputs
   * Return types
   * Edge cases
   * Constraints
   * Allowed assumptions

4. **Be redundancy-friendly**

   * Restate critical constraints more than once if helpful.
   * Small LLMs benefit from repetition of key rules.

5. **Strict structure**
   Always use this order:

   ---

   GOAL
   (One-sentence description of what must be implemented.)

   TASK
   (Clear description of what the function/program must do.)

   INPUT
   (Type, format, assumptions.)

   OUTPUT
   (Type and exact expectations.)

   RULES
   (Bulleted list of requirements and constraints.)

   EDGE CASES
   (Explicitly list if present or implied.)

   EXAMPLES
   (Copy exactly from the original if provided.)

   OUTPUT FORMAT REQUIREMENTS
   (Exactly what the answering model must output and not output.)

   ---

6. **Formatting clarity**

   * Use bullet points and short sentences.
   * Avoid long paragraphs.
   * Avoid figurative language.

7. **No solution content**

   * Do not include hints beyond the original.
   * Do not include algorithms unless already given.
   * Do not include code unless it was provided.

8. **Faithful examples**

   * Preserve examples exactly.
   * Do not modify example outputs.

9. **Concise but complete**

   * Remove fluff.
   * Keep all technical detail.

10. **Only output the rewritten question**

    * No commentary.
    * No explanations.
    * No meta-text.

11. **Be very careful not to invent constraints that are inconsistent with the examples**

    * The text may lead you astray. Pay very careful attention to the examples and follow them exactly

---

Rewrite the following question accordingly:

<USER_QUESTION>
{question}
</USER_QUESTION>"""
