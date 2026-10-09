#!/usr/bin/env python3
"""Verify distributed file hashes without importing the training package."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    failures = []
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(item["path"])
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            failures.append(item["path"])
    print(json.dumps({"files": len(manifest["files"]), "mismatches": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
