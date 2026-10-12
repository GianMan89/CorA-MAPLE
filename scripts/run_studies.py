#!/usr/bin/env python3
"""Run configured studies in dependency order, resume completed jobs, then report."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "primary": "repeated_cv.yaml",
    "pool_sizes": "pool_sizes.yaml",
    "variability_k1": "diagnostics/variability_k1.yaml",
    "short_schedule": "diagnostics/short_schedule.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", choices=list(CONFIG), default=["primary", "pool_sizes"])
    parser.add_argument("--workspace", default="runs/tep_cv10x5_cleanft_v2_1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("Worker and thread counts must be positive.")
    workspace = str(Path(args.workspace).resolve())
    for study in args.studies:
        subprocess.run([sys.executable, "-m", "cora_maple", "run", "--config", str(ROOT / "configs" / CONFIG[study]),
                        "--workspace", workspace, "--workers", str(args.workers), "--device", args.device,
                        "--threads", str(args.threads)], check=True)
        subprocess.run([sys.executable, "-m", "cora_maple", "report", "--workspace", workspace,
                        "--study", study], check=True)


if __name__ == "__main__":
    main()
