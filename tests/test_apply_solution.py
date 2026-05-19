"""Tests for apply_solution module, converted from inline test_apply_solution()."""

from ollama_codeeval.apply_solution import (
    _dedent_code,
    _extract_from_markdown,
    _strip_thinking_tags,
    apply_solution,
)

PROMPT = (
    'from typing import List\n\n'
    'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
    '    """ Docstring """\n'
)
ENTRYPOINT = "has_close_elements"


class TestStripThinkingTags:
    def test_response_tag(self):
        s = "<think>thinking...</think><response>answer</response>"
        assert _strip_thinking_tags(s) == "answer"

    def test_think_tag_only(self):
        s = "<think>thinking...</think>answer after"
        assert _strip_thinking_tags(s) == "answer after"

    def test_no_tags(self):
        assert _strip_thinking_tags("plain text") == "plain text"


class TestExtractFromMarkdown:
    def test_single_block(self):
        s = "```python\ndef foo(): pass\n```"
        assert _extract_from_markdown(s, "foo") == "def foo(): pass\n"

    def test_prefers_block_with_entrypoint(self):
        s = "```python\nx = 1\n```\n```python\ndef bar(): pass\n```"
        assert "def bar" in _extract_from_markdown(s, "bar")

    def test_unclosed_block(self):
        s = "```python\ndef foo(): pass\n"
        result = _extract_from_markdown(s, "foo")
        assert "def foo" in result

    def test_no_markdown(self):
        s = "def foo(): pass"
        assert _extract_from_markdown(s, "foo") == s


class TestDedentCode:
    def test_indented(self):
        code = "    def foo():\n        pass"
        assert _dedent_code(code) == "def foo():\n    pass"

    def test_already_dedented(self):
        code = "def foo():\n    pass"
        assert _dedent_code(code) == code

    def test_empty(self):
        assert _dedent_code("") == ""


class TestApplySolution:
    def test_full_solution_with_preamble(self):
        solution = (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "  return False\n"
        )
        result = apply_solution(PROMPT, solution, ENTRYPOINT)
        assert "from typing import List" in result
        assert "def has_close_elements" in result
        assert "return False" in result

    def test_markdown_block(self):
        solution = (
            "```python\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "  return False\n"
            "```"
        )
        result = apply_solution(PROMPT, solution, ENTRYPOINT)
        assert "from typing import List" in result
        assert "return False" in result

    def test_body_only(self):
        solution = "  # useful code\n  return True"
        result = apply_solution(PROMPT, solution, ENTRYPOINT)
        assert "def has_close_elements" in result
        assert "return True" in result

    def test_extra_indentation(self):
        solution = (
            "    def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "        return False"
        )
        result = apply_solution(PROMPT, solution, ENTRYPOINT)
        assert "def has_close_elements" in result
        assert result.strip().endswith("return False")

    def test_thinking_tags(self):
        solution = (
            "<think>The user wants to solve this problem.</think>"
            "<response>"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "  return False\n"
            "</response>"
        )
        result = apply_solution(PROMPT, solution, ENTRYPOINT)
        assert "<think>" not in result
        assert "return False" in result

    def test_multiple_markdown_blocks_prefers_entrypoint(self):
        solution = (
            '\nSome text.\n'
            '```python\nprint("hello")\n```\n'
            '```python\n'
            'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
            '    return True\n'
            '```\n'
            '```python\nx = 1\n```\n'
        )
        result = apply_solution(PROMPT, solution, ENTRYPOINT)
        assert "return True" in result
        assert "print" not in result
