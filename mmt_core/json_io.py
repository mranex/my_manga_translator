"""Shared atomic JSON file helpers for cache metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .image_io import ensure_path


def write_json_atomic(
    path: Path | str,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    target_path = ensure_path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp")
    serialized = json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii)
    temp_path.write_text(serialized, encoding="utf-8")
    os.replace(temp_path, target_path)
    return target_path


__all__ = ["write_json_atomic"]
