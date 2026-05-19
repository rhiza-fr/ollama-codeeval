"""Generate a markdown report comparing original HumanEval problems with their variants.

Usage:
    uv run python -m ollama_codeeval.contamination.dataset_diff_report [original.jsonl.gz] [variants.jsonl.gz]

Both arguments are optional; defaults to the standard data files.
"""

import argparse
import difflib
import sys
from pathlib import Path

from ollama_codeeval.data import _get_data_path, load_jsonl_gz

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
DEFAULT_VARIANTS = DATA_DIR / "variants.jsonl.gz"


def _prompt_diff(original: str, variant: str) -> str:
    """Return a unified diff of two prompts as a fenced markdown block."""
    orig_lines = original.splitlines(keepends=True)
    var_lines = variant.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, var_lines, fromfile="original", tofile="variant", lineterm="")
    diff_text = "".join(diff)
    if not diff_text:
        return "_No difference in prompt._\n"
    return f"```diff\n{diff_text}\n```\n"


def generate_report(original_path: Path, variants_path: Path) -> str:
    original: dict[str, dict] = {}
    for rec in load_jsonl_gz(original_path):
        original[rec["task_id"]] = rec

    variants: dict[str, dict] = {}
    for rec in load_jsonl_gz(variants_path):
        variants[rec["original_task_id"]] = rec

    paired = sorted(set(original) & set(variants))
    only_original = sorted(set(original) - set(variants))

    lines: list[str] = []
    lines.append("# Dataset Diff Report\n")
    lines.append(f"- **Original problems:** {len(original)}")
    lines.append(f"- **Variant problems:** {len(variants)}")
    lines.append(f"- **Paired (have variant):** {len(paired)}")
    lines.append(f"- **No variant yet:** {len(only_original)}\n")

    if paired:
        lines.append("---\n")
        lines.append("## Paired Problems\n")
        for task_id in paired:
            orig = original[task_id]
            var = variants[task_id]
            lines.append(f"### {task_id} -> {var['task_id']}\n")
            lines.append(
                f"| | Original | Variant |\n"
                f"|---|---|---|\n"
                f"| **entry_point** | `{orig['entry_point']}` | `{var['entry_point']}` |\n"
            )
            lines.append("\n**Prompt diff:**\n")
            lines.append(_prompt_diff(orig["prompt"], var["prompt"]))

    if only_original:
        lines.append("---\n")
        lines.append("## Problems Without Variants\n")
        lines.append(", ".join(f"`{t}`" for t in only_original))
        lines.append("\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a markdown dataset diff report.")
    parser.add_argument("original", nargs="?", type=Path, default=None, help="Path to original .jsonl.gz")
    parser.add_argument("variants", nargs="?", type=Path, default=DEFAULT_VARIANTS, help="Path to variants .jsonl.gz")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write report to file instead of stdout")
    args = parser.parse_args()

    original_path = args.original or _get_data_path()
    variants_path = args.variants

    if not original_path.exists():
        print(f"Error: original dataset not found at {original_path}", file=sys.stderr)
        sys.exit(1)
    if not variants_path.exists():
        print(f"Error: variants dataset not found at {variants_path}", file=sys.stderr)
        sys.exit(1)

    report = generate_report(original_path, variants_path)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
