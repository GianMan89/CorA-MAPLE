"""Stable identifiers, atomic writes, and provenance (no training)."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

def repository_root() -> Path:
    """Locate repository assets for editable installs or an explicit resource root."""
    explicit = os.environ.get("CORA_MAPLE_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs" / "base.yaml").exists():
        return candidate
    current = Path.cwd().resolve()
    for parent in (current, *current.parents):
        if (parent / "configs" / "base.yaml").exists():
            return parent
    return candidate

ROOT = repository_root()

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def seed_from(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(canonical(parts).encode()).digest()[:4], "little")

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def source_hash() -> str:
    records = {}
    for folder in (ROOT / "src", ROOT / "vendor"):
        for path in sorted(folder.rglob("*.py")):
            records[str(path.relative_to(ROOT))] = file_hash(path)
    return digest(records)

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "PyYAML", "filelock", "matplotlib", "joblib", "threadpoolctl"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version, "platform": platform.platform(), "packages": versions}
