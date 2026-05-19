"""One-off system profiling: collect CPU, GPU, Docker, Ruff, Ollama, Python info.

Run manually before evaluation or as part of a setup step.  Output is written
to output/system_profile.json and consumed by the setup report page.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

from ollama_codeeval.config import OUTPUT_BASE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], timeout: float = 15.0) -> str | None:
    """Run a command and return stripped stdout; None on any failure."""
    try:
        cp = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
            errors="replace",
        )
        return cp.stdout.strip() or None
    except Exception:
        return None


def _run_multi(args: list[str], timeout: float = 15.0) -> dict[str, str | int]:
    """Run a command and return {'stdout': ..., 'stderr': ..., 'rc': ...}."""
    try:
        cp = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, encoding="utf-8",
            errors="replace",
        )
        return {"stdout": cp.stdout.strip(), "stderr": cp.stderr.strip(), "rc": cp.returncode}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "rc": -1}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_cpu() -> dict:
    """CPU information from Python platform module and (Windows) wmic."""
    info: dict = {
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "cpu_count_logical": getattr(platform, "_os", None),
    }
    try:
        info["cpu_count_logical"] = getattr(platform, "_syscmd_uname")().processor
    except Exception:
        pass

    if sys.platform == "win32":
        wmic = _run(["cmd", "/c", "wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/format:csv"])
        if wmic:
            # Strip the Node column (first CSV field) which contains the machine name
            lines = wmic.splitlines()
            if lines:
                header = lines[0]
                # Remove first field (Node) from header and data lines
                parts = header.split(",", 1)
                lines[0] = parts[1] if len(parts) > 1 else header
                for i in range(1, len(lines)):
                    parts = lines[i].split(",", 1)
                    lines[i] = parts[1] if len(parts) > 1 else lines[i]
                info["wmic"] = "\n".join(lines)
    else:
        cpuinfo = _run(["cat", "/proc/cpuinfo"])
        if cpuinfo:
            # Keep only unique model name lines
            lines = [ln for ln in cpuinfo.splitlines() if "model name" in ln]
            info["model_names"] = list(dict.fromkeys(lines))

    # Try platform.uname() directly
    try:
        u = platform.uname()
        info["system"] = u.system
        info["release"] = u.release
        # On Windows, uname().version can contain PII — use getwindowsversion instead
        if sys.platform == "win32":
            try:
                wv = sys.getwindowsversion()
                info["version"] = f"{wv.major}.{wv.minor}.{wv.build}"
            except Exception:
                info["version"] = u.release
        else:
            info["version"] = u.version
    except Exception:
        pass

    return info


def probe_gpu() -> dict:
    """GPU information via nvidia-smi; returns empty dict if unavailable."""
    res = _run_multi(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                       "--format=csv,noheader"])
    if res["rc"] != 0:
        return {"available": False, "error": res.get("stderr") or "nvidia-smi not found"}
    lines = [ln for ln in str(res["stdout"]).splitlines() if ln.strip()]
    gpus = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 4:
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "memory": parts[2],
                "driver": parts[3],
            })
    return {"available": True, "gpus": gpus}


def probe_docker() -> dict:
    """Docker version and whether the sandbox image exists."""
    version = _run(["docker", "--version"]) or "not available"
    info_output = _run(["docker", "info", "--format", "{{.OSType}} {{.ServerVersion}} {{.KernelVersion}}"])
    sandbox = _run(["docker", "image", "inspect", "python-sandbox", "--format", "{{.RepoTags}}"])
    return {
        "version": version,
        "info": info_output or "not available",
        "sandbox_image": "present" if sandbox else "not found",
    }


def probe_ruff() -> dict:
    """Ruff version."""
    ver = _run(["ruff", "--version"]) or "not available"
    return {"version": ver}


def probe_ollama_version() -> dict:
    """Ollama CLI and API version."""
    cli_ver = _run(["ollama", "--version"]) or "not available"
    return {"cli_version": cli_ver}


def probe_ollama_api() -> dict:
    """Ollama API version (requires running Ollama server)."""
    from urllib.error import URLError
    from urllib.request import urlopen

    from ollama_codeeval.config import OLLAMA_HOST
    try:
        with urlopen(f"{OLLAMA_HOST}/api/version") as r:  # nosec B310
            data = json.loads(r.read())
        return {"available": True, "version": data.get("version", "unknown")}
    except (URLError, json.JSONDecodeError) as e:
        return {"available": False, "error": str(e)}


def probe_python() -> dict:
    """Python version info — executable path reduced to basename for privacy."""
    exe = Path(sys.executable).name
    return {
        "version": sys.version,
        "executable": exe,
        "implementation": platform.python_implementation(),
    }


# ---------------------------------------------------------------------------
# Collect & write
# ---------------------------------------------------------------------------

def collect() -> dict:
    """Run all probes and return a single system-profile dict.

    This function is intentionally lightweight so it can be called from tests
    or other automation without filesystem side-effects.
    """
    return {
        "cpu": probe_cpu(),
        "gpu": probe_gpu(),
        "docker": probe_docker(),
        "ruff": probe_ruff(),
        "ollama": {
            "cli": probe_ollama_version(),
            "api": probe_ollama_api(),
        },
        "python": probe_python(),
    }


def write_profile(output_dir: Path | None = None) -> Path:
    """Collect system profile and write to output/system_profile.json.

    Returns the path that was written.
    """
    profile = collect()
    if output_dir is None:
        output_dir = OUTPUT_BASE
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "system_profile.json"
    dest.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"System profile written to {dest}")
    return dest


if __name__ == "__main__":
    write_profile()
