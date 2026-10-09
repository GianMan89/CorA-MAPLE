#!/usr/bin/env python3
"""Create source partitions and job specifications without fitting models."""
from pathlib import Path
import argparse
import json
from cora_maple.common import ROOT, atomic_json
from cora_maple.config import load_config
from cora_maple.protocol import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="runs/tep_cv10x5")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    seen, report = set(), {}
    for filename in ["repeated_cv.yaml", "pool_sizes.yaml"]:
        plan = prepare(load_config(ROOT / "configs" / filename), workspace)
        added = set(plan["tasks"]) - seen
        report[plan["study"]] = {"tasks": plan["counts"], "additional_tasks": {
            kind: sum(plan["tasks"][i]["kind"] == kind for i in added) for kind in ("pretrain", "fit", "baseline")},
            "shared_tasks": len(set(plan["tasks"]) & seen)}
        seen.update(plan["tasks"])
    report["unique_tasks"] = len(seen)
    atomic_json(workspace / "workload_plan.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
