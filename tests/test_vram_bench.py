# tests/test_vram_bench.py
import importlib.util
from pathlib import Path

# Load module without executing __main__
_spec = importlib.util.spec_from_file_location(
    "vram_bench",
    Path(__file__).parent.parent / "scripts" / "vram_bench.py",
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Could not load vram_bench module spec")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_is_generative_skips_embed_in_name():
    model = {"name": "nomic-embed-text:latest", "details": {"families": []}}
    assert mod.is_generative(model) is False


def test_is_generative_skips_base_tag():
    model = {"name": "llama3:base", "details": {"families": ["llama"]}}
    assert mod.is_generative(model) is False


def test_is_generative_skips_bert_family():
    model = {"name": "some-model:latest", "details": {"families": ["bert"]}}
    assert mod.is_generative(model) is False


def test_is_generative_skips_nomic_family():
    model = {"name": "some-model:latest", "details": {"families": ["nomic"]}}
    assert mod.is_generative(model) is False


def test_is_generative_skips_clip_family():
    model = {"name": "some-model:latest", "details": {"families": ["clip"]}}
    assert mod.is_generative(model) is False


def test_is_generative_allows_normal_model():
    model = {"name": "qwen2.5-coder:7b", "details": {"families": ["qwen2"]}}
    assert mod.is_generative(model) is True


def test_is_generative_missing_families_key():
    model = {"name": "llama3:latest", "details": {}}
    assert mod.is_generative(model) is True


def test_parse_nvidia_smi_output():
    raw = "4096 MiB\n2048 MiB\n"
    result = mod.parse_nvidia_smi(raw)
    assert result == {"GPU-0": 4096, "GPU-1": 2048}


def test_parse_nvidia_smi_single_gpu():
    raw = "8192 MiB\n"
    result = mod.parse_nvidia_smi(raw)
    assert result == {"GPU-0": 8192}


def test_parse_nvidia_smi_empty():
    result = mod.parse_nvidia_smi("")
    assert result == {}


def test_vram_delta():
    before = {"GPU-0": 2000, "GPU-1": 500}
    after = {"GPU-0": 6821, "GPU-1": 500}
    assert mod.vram_delta(before, after) == {"GPU-0": 4821, "GPU-1": 0}


def test_vram_delta_clamps_negative():
    before = {"GPU-0": 5000, "GPU-1": 500}
    after = {"GPU-0": 4800, "GPU-1": 600}  # GPU-0 went down (noise), GPU-1 went up
    result = mod.vram_delta(before, after)
    assert result == {"GPU-0": 0, "GPU-1": 100}


def test_calc_toks_per_sec_normal():
    # duration is in nanoseconds per Ollama API
    assert mod.calc_toks_per_sec(100, 2_000_000_000) == 50.0


def test_calc_toks_per_sec_zero_duration():
    assert mod.calc_toks_per_sec(100, 0) is None


def test_calc_toks_per_sec_zero_count():
    assert mod.calc_toks_per_sec(0, 1_000_000_000) is None
