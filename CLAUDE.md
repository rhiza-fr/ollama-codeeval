# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated evaluation of LLM code generation using the HumanEval benchmark. Models are queried via Ollama, generated solutions are executed in a Docker sandbox, and results are scored and reported.

## Common Commands

```bash
# Install dependencies
uv sync --all-extras --dev

# Run tests
uv run pytest tests

# Run a single test
uv run pytest tests/test_dummy.py::test_dummy_test

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Run an evaluation
uv run python src/ollama_codeeval/runner.py [model1,model2]
```

## Dependencies

- **Package manager:** uv
- **Formatter/linter:** ruff
- **Python:** 3.12+ (see `.python-version`)
- **Build system:** hatchling
- **Vendored modules:** `pocketflow.py` and `autocontext.py` are vendored in `src/ollama_codeeval/` (originally from PocketFlowTyped and ollama-think-autocontext)

## Architecture

### Evaluation Pipeline (PocketFlow node graph)

`agent.py` defines a node-based flow using PocketFlow:

```
FormatOriginalQuestionNode → GenerateNode → ExecuteNode → RespondNode
                                                ↓ error
                                             FixNode → ExecuteNode (retry)
                                                ↓ stuck
                                          FixHarderNode → ExecuteNode (retry)
```

- **GenerateNode**: Sends prompts to Ollama via `ollama-think-autocontext` (auto-growing context window)
- **ExecuteNode**: Runs generated code + HumanEval tests in a Docker sandbox (`python-sandbox` image), caches results with `diskcache`
- **FixNode / FixHarderNode**: Re-prompts the model with error feedback for self-correction (max 5 iterations)
- **RespondNode**: Captures final pass/fail result

### Key Modules

- `runner.py` — Entry point. Iterates models × tasks, orchestrates `run_flow()`, writes JSONL results to `output/`
- `agent.py` — PocketFlow graph definition and execution. Holds global `Sandbox` and `Cache` instances
- `sandbox.py` — Docker-based sandboxed code execution via `llm-sandbox`
- `pytest_wrapper.py` — Wraps generated code with HumanEval tests for pytest, cleans tracebacks
- `apply_solution.py` — Extracts and merges LLM output into the original prompt stub
- `data.py` — Loads HumanEval dataset from `data/*.jsonl.gz` or Hugging Face
- `report/` — HTML/Plotly report generation from evaluation results

### Prompt Convention

Generated prompts instruct the model to output only a `def` function — no markdown, no explanations. The `apply_solution` module handles extracting the function from potentially noisy LLM output.

## CI

Gitea Actions runs on push/PR: syncs dependencies with uv, runs `pytest tests`.
