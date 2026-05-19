#!/usr/bin/env python
# src/ollama_codeeval/contamination/generate_variants.py
"""Generate isomorphic HumanEval variants for contamination testing.

Pipeline per task:
  1. morph_solution   (qwen3-coder) — new domain, new function name
  2. fuzz_inputs      (qwen3-coder) — 8 diverse input tuples
  3. run_oracle       (sandbox)     — produce (input, output) pairs
  4. write_question   (gpt-oss:20b) — new docstring from examples only
  5. verify           (sandbox)     — round-trip check
  6. emit             — append to variants.jsonl.gz
"""

import argparse
import gzip
import json
import re
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from ollama_codeeval.apply_solution import _extract_from_markdown, _strip_thinking_tags
from ollama_codeeval.autocontext import AutoContextClient
from ollama_codeeval.config import OLLAMA_HOST, SANDBOX_IMAGE, SANDBOX_LANG, SANDBOX_TIMEOUT
from ollama_codeeval.data import load_data
from ollama_codeeval.sandbox import Sandbox

from ollama_codeeval.contamination.oracle import run_oracle
from ollama_codeeval.contamination.prompts import (
    FUZZ_INPUTS_PROMPT,
    MORPH_SOLUTION_PROMPT,
    WRITE_QUESTION_PROMPT,
)
from ollama_codeeval.contamination.variant_utils import (
    build_check_function,
    ensure_typing_imports,
    extract_function_name,
    parse_input_list,
)

MORPH_MODEL = "qwen3-coder"
QUESTION_MODEL = "gpt-oss:20b"
MIN_PAIRS = 4
MAX_RETRIES = 3

console = Console()
client = AutoContextClient(host=OLLAMA_HOST)


def _llm(model: str, prompt: str) -> str:
    response = client.call(model, prompt, think=False, options={"temperature": 0.5})
    return response.to_dict()["message"]["content"]


def _clean_code(raw: str, entry_point: str) -> str:
    raw = _strip_thinking_tags(raw)
    return _extract_from_markdown(raw, entry_point).replace("```", "").strip()


def _extract_signature_tail(code: str, entry_point: str) -> str:
    """Extract '(params) -> return_type:' tail from first def line."""
    m = re.search(rf"\bdef\s+{re.escape(entry_point)}\s*(\(.*)", code, re.DOTALL)
    if not m:
        return "(*args)"
    tail = m.group(1)
    # Find the matching closing paren for the parameter list
    depth = 0
    close_paren_idx = -1
    for i, ch in enumerate(tail):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_paren_idx = i
                break
    if close_paren_idx == -1:
        return tail.split("\n")[0]
    rest = tail[close_paren_idx + 1 :]
    # Find the top-level colon that ends the def line,
    # skipping colons nested inside [], {}, ()
    bracket_depth = 0
    for j, ch in enumerate(rest):
        if ch in "([{":
            bracket_depth += 1
        elif ch in ")]}":
            bracket_depth -= 1
        elif ch == ":" and bracket_depth == 0:
            return tail[: close_paren_idx + 1] + rest[: j + 1]
    return tail[: close_paren_idx + 1] + ":"


def _format_examples(pairs: list[tuple]) -> str:
    lines = []
    for args, result in pairs:
        args_str = ", ".join(repr(a) for a in args)
        lines.append(f"  f({args_str}) -> {repr(result)}")
    return "\n".join(lines)


def generate_variant(task: dict, sandbox: Sandbox) -> dict | None:
    """Attempt to generate a single variant. Returns None if all retries fail."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            variant = _try_generate(task, sandbox)
            if variant is not None:
                return variant
        except Exception as exc:
            console.print(f"    [yellow]attempt {attempt} error: {exc}[/yellow]")
    return None


def _truncate_to_stub(code: str) -> str:
    """Strip the function body after the docstring, leaving only def + docstring + pass."""
    lines = code.splitlines()
    in_docstring = False
    docstring_char = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                # Check if it's a one-liner docstring: opens and closes on same line
                rest = stripped[3:]
                if docstring_char in rest:
                    return "\n".join(lines[: i + 1]) + "\n    pass\n"
                in_docstring = True
        else:
            if docstring_char and docstring_char in stripped:
                return "\n".join(lines[: i + 1]) + "\n    pass\n"
    # No docstring found — keep only the def line, append pass
    def_line_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("def ")), 0
    )
    return "\n".join(lines[: def_line_idx + 1]) + "\n    pass\n"


def _try_generate(task: dict, sandbox: Sandbox) -> dict | None:
    original_entry = task["entry_point"]

    # Step 1: morph solution
    morph_prompt = MORPH_SOLUTION_PROMPT.format(
        entry_point=original_entry,
        prompt=task["prompt"],
        canonical_solution=task["canonical_solution"],
    )
    raw_morph = _llm(MORPH_MODEL, morph_prompt)
    morphed_code = _clean_code(raw_morph, original_entry)
    new_entry = extract_function_name(morphed_code)
    if not new_entry:
        return None
    morphed_code = _clean_code(raw_morph, new_entry)

    # Step 2: fuzz inputs
    fuzz_prompt = FUZZ_INPUTS_PROMPT.format(morphed_function=morphed_code)
    raw_inputs = _llm(MORPH_MODEL, fuzz_prompt)
    try:
        inputs = parse_input_list(raw_inputs)
    except ValueError:
        return None
    inputs = [i if isinstance(i, tuple) else (i,) for i in inputs]

    # Step 3: oracle
    pairs = run_oracle(morphed_code, new_entry, inputs, sandbox)
    if len(pairs) < MIN_PAIRS:
        return None

    # Step 4: write question (gpt-oss:20b sees only signature + examples)
    sig_tail = _extract_signature_tail(morphed_code, new_entry)
    examples_str = _format_examples(pairs[:6])
    question_prompt = WRITE_QUESTION_PROMPT.format(
        entry_point=new_entry,
        signature_tail=sig_tail,
        examples=examples_str,
    )
    raw_question = _llm(QUESTION_MODEL, question_prompt)
    new_stub = _clean_code(raw_question, new_entry)
    new_stub = _truncate_to_stub(new_stub)
    new_stub = ensure_typing_imports(new_stub)
    if f"def {new_entry}" not in new_stub:
        return None

    # Step 5: verify
    check_fn = build_check_function(new_entry, pairs)
    verify_script = f"{morphed_code}\n\n{check_fn}\ncheck({new_entry})\n"
    result = sandbox.run(verify_script)
    if result["exit_code"] != 0:
        return None

    # Step 6: emit
    task_num = task["task_id"].split("/")[-1]
    return {
        "task_id": f"HumanEvalVariant/{task_num}",
        "original_task_id": task["task_id"],
        "prompt": new_stub + "\n",
        "entry_point": new_entry,
        "canonical_solution": morphed_code,
        "test": check_fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HumanEval variant dataset")
    parser.add_argument(
        "--output",
        default="data/variants.jsonl.gz",
        help="Output path for variant dataset (default: data/variants.jsonl.gz)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N tasks (default: all)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_data()
    if args.limit:
        tasks = tasks[: args.limit]

    console.print(f"Generating variants for {len(tasks)} tasks → {output_path}")
    console.print(f"  Morph model:    {MORPH_MODEL}")
    console.print(f"  Question model: {QUESTION_MODEL}")

    succeeded = 0
    failed_ids = []

    with Sandbox(lang=SANDBOX_LANG, image=SANDBOX_IMAGE, execution_timeout=SANDBOX_TIMEOUT) as sandbox:
        with gzip.open(output_path, "wt", encoding="utf-8") as out_f:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
            ) as progress:
                prog = progress.add_task("Variants", total=len(tasks))
                for task in tasks:
                    tid = task["task_id"]
                    progress.update(prog, description=f"  {tid}")
                    variant = generate_variant(task, sandbox)
                    if variant is not None:
                        out_f.write(json.dumps(variant) + "\n")
                        succeeded += 1
                    else:
                        failed_ids.append(tid)
                    progress.advance(prog)

    console.print(f"\n[green]Done.[/green] {succeeded}/{len(tasks)} variants generated.")
    if failed_ids:
        console.print(f"[yellow]Failed ({len(failed_ids)}):[/yellow] {', '.join(failed_ids)}")


if __name__ == "__main__":
    main()
