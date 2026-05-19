# ollama-codeeval

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automated evaluation of LLM code generation using the [HumanEval](https://github.com/openai/human-eval) benchmark. Models are queried via [Ollama](https://ollama.com), generated solutions are executed in a Docker sandbox, and results are scored and reported.

[View published results](https://ai.rhiza.fr/humaneval/)

## Features

- Evaluates any Ollama-compatible model against 164 HumanEval coding tasks
- Docker-sandboxed code execution for safe evaluation
- Automatic self-correction: models retry with error feedback (up to 5 iterations, with temperature escalation)
- Automatic `ruff format` + `ruff check --fix` on generated code before execution
- Support for thinking/reasoning models
- Auto-growing context window via `ollama-think-autocontext`
- **Multi-tier cascade evaluation**: route tasks through fast small models first, escalate only failures to larger models — saves time and cost while maintaining accuracy
- **Cascade optimizer**: greedy algorithm that finds the optimal model sequence and per-tier iteration budget under VRAM and time-per-task constraints
- HTML reports with per-model statistics, error breakdowns, and response time distributions
- Interactive Plotly charts, performance envelope analysis
- **Model selector page**: filter models by VRAM budget and urgency (τ), ranked by yield score
- Contamination detection: generate isomorphic variants and score training-data leakage
- **Rewrite datasets**: prompt-rephrased versions of HumanEval to measure prompt sensitivity
- VRAM benchmarking: measure per-model memory usage and tokens/sec across context sizes
- Result caching with `diskcache` to avoid redundant API calls

## Requirements

- Python 3.12+
- [Docker](https://docs.docker.com/get-docker/) (for sandboxed code execution)
- [Ollama](https://ollama.com) running locally or on a remote host
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/rhiza-fr/ollama-codeeval.git
cd ollama-codeeval

# Install dependencies
uv sync --all-extras --dev
```

## Docker sandbox setup

The evaluation runs generated code inside a Docker container for safety. Build the sandbox image:

```bash
docker build -t python-sandbox -f Dockerfile .
```

## Usage

### Run an evaluation

```bash
# Evaluate a single model
ollama-codeeval eval qwen3:4b

# Evaluate multiple models
ollama-codeeval eval "qwen3:4b,gemma4:26b"

# With thinking mode enabled
ollama-codeeval eval qwen3:4b --think

# Custom Ollama host
ollama-codeeval eval qwen3:4b --host http://remote-host:11434

# Concurrent workers (speeds up small models on capable hardware)
ollama-codeeval eval qwen3:4b --workers 4

# Evaluate a specific dataset (rewrite, variants, etc.)
ollama-codeeval eval qwen3:4b --dataset humaneval-rewritten-qwen3-14b

# Custom output directory and tag suffix
ollama-codeeval eval qwen3:4b --output-dir results --tag v2

# Multi-tier cascade evaluation (fast models first, escalate failures)
ollama-codeeval eval qwen3:4b --cascade

# Clear cache before running
ollama-codeeval eval qwen3:4b --clear-cache
```

Results are written as JSONL files to the output directory (default: `output/`).

### Generate reports

```bash
ollama-codeeval report
```

This generates HTML reports in `output/html/` with:

| Page | Description |
|------|-------------|
| **Index** | Summary table comparing all evaluated models |
| **Model detail** | Per-model pages with iteration-level results |
| **Tasks** | Per-task table with difficulty signals (description sparsity, error patterns, rewrite sensitivity) |
| **Context window** | Explosion rate, token usage, per-task breakdown |
| **Cascade** | Cascade evaluation results with tier-by-tier breakdowns |
| **Model selector** | Filter models by VRAM budget and urgency (τ), ranked by yield score |
| **Setup** | Agent flow diagram, configuration, and machine system profile |

Charts include:
- Pass rate comparisons
- Error type breakdowns
- Response time and token length distributions
- Performance envelope analysis
- **Yield score**: success probability within a patience threshold (τ)
- **Dumbbell charts** for rewrite comparison
- **Combined progression** charts tracking pass rate across attempts, iterations, and rewrites

## Cascade evaluation

The cascade flow routes each task through a sequence of models — start with a fast small model, escalate only failures to larger ones. This dramatically reduces total evaluation time for a given accuracy target.

```bash
# CLI (uses default cascade: qwen3:4b → qwen2.5-coder:3b → qwen3:14b)
ollama-codeeval eval qwen3:4b --cascade
```

### Optimize a cascade

Given per-model JSONL result files, the optimizer finds the best tier sequence and iteration budgets:

```bash
# Under a 15-second target and 28GB VRAM budget
uv run python scripts/optimize_cascade.py --target-time 15 --vram-budget 28000
```

The optimizer considers:
- Per-model pass rate and average time
- Model load times and VRAM footprint
- Iteration budget per tier
- Yield probability within the time target


## VRAM benchmarking

Measure per-model VRAM usage and tokens/second at various context sizes:

```bash
uv run python scripts/vram_bench.py --models "qwen3:4b,llama3.1:8b" --contexts "2048,4096,8192,16384"
```

Results are written to `output/vram_bench_*.json` and consumed by the cascade optimizer and model selector report page.

## Rewrite datasets

Rewrite datasets are alternative versions of the 164 HumanEval tasks where the problem description has been rephrased by a different LLM, while the function signature and tests remain identical. Running the same model against both the original and a rewritten dataset measures *prompt sensitivity*: how much performance depends on how a problem is described rather than the underlying skill.

### Pre-built datasets

Three rewrite datasets are included in `data/`:

| File | Rewrite model |
|------|--------------|
| `humaneval-rewritten-gpt-oss20b.jsonl.gz` | `gpt-oss:20b` |
| `humaneval-rewritten-ministral-314b.jsonl.gz` | `ministral:314b` |
| `humaneval-rewritten-qwen34b.jsonl.gz` | `qwen3:4b` |

### Generate a new rewrite dataset

```bash
# Rewrite prompts using any Ollama model
uv run python scripts/generate_rewrites.py --model qwen3:14b
```

This writes `data/humaneval-rewritten-qwen3-14b.jsonl.gz` and caches each rewrite in `diskcache`, so interrupted runs resume from where they left off.

### Run evaluation on a rewrite dataset (CLI)

```bash
# The --dataset flag loads any .jsonl.gz file from data/
ollama-codeeval eval qwen3:4b --dataset humaneval-rewritten-qwen3-14b

# NOTE: rewrite report pages are generated but are explicitly excluded from overall reports - small tweak to re-link if required
```

## Contamination detection

Training-data contamination can inflate benchmark scores. The contamination module generates isomorphic variants of HumanEval tasks and measures how much a model's pass rate drops on tasks it has never seen.

### Generate variants

```bash
# Generate variant dataset (requires qwen3-coder and gpt-oss:20b via Ollama)
uv run python -m ollama_codeeval.contamination.generate_variants

# Limit to N tasks for testing
uv run python -m ollama_codeeval.contamination.generate_variants --limit 20
```

This produces `data/variants.jsonl.gz`. Each variant is an isomorphic task in a different domain: the original solution is morphed, new inputs are fuzzed through a sandbox oracle, and a fresh docstring is written from the examples only.

### Run evaluation on variants

```bash
ollama-codeeval eval qwen3:4b --dataset variants
```

### Score contamination

```bash
uv run python -m ollama_codeeval.contamination.contamination_report \
    output/results_original.jsonl \
    output/results_variant.jsonl
```

The score is `P(pass_original ∧ fail_variant) − P(fail_original ∧ pass_variant)`. A high positive score suggests the model recognised the original tasks from training data.

## Project structure

```
src/ollama_codeeval/
  cli.py                  - Typer CLI entry point
  runner.py               - Orchestrates model × task evaluation
  agent.py                - PocketFlow node graph (generate → ruff → execute → fix → respond)
  cascade_agent.py        - Cascade flow: routes tasks through fast models first, escalates failures
  cascade_optimizer.py    - Greedy cascade optimization: model sequence + budget under constraints
  cascade_report.py       - Cascade reporting: tables, retrospective, pruning
  prompts.py              - Prompt templates and builders for agent nodes
  sandbox.py              - Docker-based sandboxed code execution
  pytest_wrapper.py       - Wraps generated code with HumanEval tests
  apply_solution.py       - Extracts and merges LLM output into prompt stubs
  code_formatter.py       - Runs ruff format/fix locally before execution
  data.py                 - Loads HumanEval dataset (downloads on first use)
  config.py               - Configuration defaults (host, timeouts, limits)
  jsonl_io.py             - JSONL read/write helpers
  discover_model_stats.py - Fetches and caches model metadata from Ollama API
  system_profile.py       - One-off system profiling (CPU, GPU, Docker, Ollama, Python info)
  kde.py                  - KDE chart generation (matplotlib/scipy)
  pocketflow.py           - Node graph engine (could be replaced)
  autocontext.py          - Auto-growing context to help thinking models finish
  contamination/          - Contamination detection tooling
    generate_variants.py    - Generates isomorphic variant dataset
    contamination_report.py - Scores contamination from two result files
    dataset_diff_report.py  - Diff-based quality analysis between datasets
    oracle.py               - Sandbox oracle for input/output pair generation
    variant_utils.py         - Scoring and utility functions
    prompts.py               - LLM prompt templates for variant generation
  report/                 - HTML/Plotly report generation
    _shared.py              - Shared helpers for all report pages
    html_report.py          - Top-level report builder
    html_index.py           - Index page (model comparison table)
    html_model.py           - Per-model detail page
    html_tasks.py           - Tasks table with difficulty signals
    html_context.py         - Context window explosion report
    html_violin.py          - Violin/distribution charts
    html_rewrite.py         - Rewrite analysis page (dumbbell charts, progression)
    html_cascade.py         - Cascade evaluation report page
    html_yield.py           - Yield analysis page
    html_selector.py        - Model selector page (VRAM + τ filtering)
    html_setup.py           - Setup report page (flow diagram, config, machine profile)
    yield_score.py          - Log-Time Yield score computation
    analyse_difficulty.py   - Per-task difficulty signal computation
    analyse_envelope.py     - Performance envelope analysis
    metrics.py              - Error classification and summary metrics
    results_parser.py       - JSONL result parsing

scripts/
  generate_rewrites.py     - Generate rewrite datasets
  vram_bench.py            - VRAM and tokens/s benchmarking across context sizes
  optimize_cascade.py      - Cascade optimizer CLI
  analyse_bench.py         - Analyse VRAM benchmark results
  compare_fixharder.py     - Compare fix-harder strategies

tests/
  test_agent_nodes.py      - Agent node unit tests
  test_sandbox.py          - Sandbox execution tests
  test_pocketflow.py       - PocketFlow sync node tests
  test_pocketflow_async.py - PocketFlow async node tests
  test_autocontext.py      - Autocontext behaviour tests
  test_runner_eval.py      - Runner evaluation logic tests
  test_apply_solution.py   - apply_solution module tests
  test_helpers.py          - Pure helper function tests
  contamination/           - Contamination module tests
```

## Evaluation flow

Each HumanEval task is processed through a PocketFlow node graph defined in `agent.py`. The flow retries with error feedback up to 5 iterations:

```
FormatOriginalQuestionNode
  │
  ▼
GenerateNode ──fail──► RespondNode
  │ default
  ▼
RuffFixNode ◄─────────────────────┐
  │ clean       │ lint_error       │
  ▼             ▼                  │
ExecuteNode   LintFixNode ────────┘
  │ ok    │ error    │ fail
  ▼       ▼         ▼
Respond  FixNode   RespondNode
Node      │ default    │ stuck
          │            ▼
          │       FixHarderNode
          │        │ default    │ verystuck
          │        │            ▼
          └────────┴──► RuffFixNode ──► ... (loop)
                                  RespondNode
```

### Cascade flow

When `--cascade` is enabled, the cascade agent (`cascade_agent.py`) replaces the single-model flow with a multi-tier routing:

```
For each task:
  Tier 1: CascadeGenerateNode (fast model, e.g. qwen3:4b)
           │ pass → mark complete
           ▼ fail
  Tier 2: CascadeGenerateNode (medium model, e.g. qwen2.5-coder:3b)
           │ pass → mark complete
           ▼ fail
  Tier 3: CascadeGenerateNode (large model, e.g. qwen3:14b)
           │ pass/fail → mark complete
```

Each tier uses the same FixNode/ExecuteNode loop as the single-model flow. Failed tasks cascade to the next tier.

### Node responsibilities

| Node | Description |
|------|-------------|
| **FormatOriginalQuestionNode** | Builds the initial prompt from the HumanEval task |
| **GenerateNode** | Sends the prompt to Ollama and captures the response |
| **RuffFixNode** | Runs `ruff format` + `ruff check --fix` locally (no Docker) to auto-fix lint issues |
| **LintFixNode** | Asks the LLM to fix remaining lint errors that ruff can't auto-fix |
| **ExecuteNode** | Runs the generated code + tests in a Docker sandbox (cached with `diskcache`) |
| **FixNode** | Re-prompts the LLM with test error feedback, includes previous attempts to prevent repetition |
| **FixHarderNode** | More aggressive re-prompt demanding a fundamentally different algorithm or strategy |
| **RespondNode** | Captures the final pass/fail result |
| **CascadeGenerateNode** | Like GenerateNode but uses `shared['current_model']` for multi-tier routing |

### Temperature escalation

During self-correction iterations, `FixNode` and `FixHarderNode` progressively increase the sampling temperature to encourage diverse solutions. The temperature starts at the base value (default `0.3`) and increases by `0.15` per iteration, capped at `1.0`. This helps the model escape local minima when repeated attempts produce the same incorrect approach.


## License

[MIT](LICENSE)
