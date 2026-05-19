"""Reads a JSONL file into a list of dictionaries. Each line is a JSON object.
Args: path - File path (str or Path). Returns: List of dicts."""

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Reads a JSONL (JSON Lines) file and returns a list of dictionaries.

    Each line in the file is expected to be a valid JSON object. This function
    is compatible with both string paths and pathlib.Path objects.

    Args:
        path: The file path to the JSONL file (can be a str or Path object).

    Returns:
        A list of dictionaries, where each dictionary represents a JSON object.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    """
    Appends a dictionary as a new line to a JSONL (JSON Lines) file.

    This function is compatible with both string paths and pathlib.Path objects.

    Args:
        path: The file path to the JSONL file (can be a str or Path object).
        data: The dictionary to be appended as a new JSON line.
    """
    with open(path, "a", encoding="utf-8") as f:
        json_record = json.dumps(data)
        f.write(json_record + "\n")
