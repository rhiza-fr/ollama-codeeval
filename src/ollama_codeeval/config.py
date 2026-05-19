"""Central configuration defaults for ollama-codeeval."""

from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"

SANDBOX_LANG = "python"
SANDBOX_IMAGE = "python-sandbox"
SANDBOX_TIMEOUT = 10.0

EXECUTION_CACHE_DIR = ".execution_cache"

OUTPUT_BASE = Path("output")
OUTPUT_HTML = OUTPUT_BASE / "html"

MODEL_TEMPERATURE = 0.0
MAX_ITERATIONS = 5
