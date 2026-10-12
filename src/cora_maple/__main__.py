"""Command-line entry points; model fitting requires an explicit 'run'."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil

# Set before CUDA may be initialized. No model is constructed here.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from . import __version__
from .common import ROOT, atomic_json
from .config import load_config, PROTOCOL


def main(argv=None):
    parser = argparse.ArgumentParser(description="CorA-MAPLE training, evaluation and reproducible reporting.")
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = subs.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--workspace", default="runs/tep_cv10x5_cleanft_v2_1")
        if name == "run":
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--device", default="cpu")
            p.add_argument("--threads", type=int, default=1)
            p.add_argument("--only-repeat", type=int)
            p.add_argument("--only-fold", type=int)
            p.add_argument("--max-tasks", type=int)
    for name in ("status", "audit", "report"):
        p = subs.add_parser(name)
        p.add_argument("--workspace", default="runs/tep_cv10x5_cleanft_v2_1")
        p.add_argument("--study", required=True)
    p = subs.add_parser("export-paper")
    p.add_argument("--workspace", default="runs/tep_cv10x5_cleanft_v2_1")
    p.add_argument("--destination", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    if args.command in ("prepare", "run"):
        config_path = Path(args.config)
        if not config_path.exists():
            config_path = ROOT / config_path
        cfg = load_config(config_path)
        if args.command == "prepare":
            from .protocol import prepare
            plan = prepare(cfg, workspace)
            result = {"status": "planned", "study": plan["study"],
                      "counts": plan["counts"], "outer_splits": plan["n_splits_total"], "workspace": str(workspace)}
        else:
            from .runner import run_study
            result = run_study(cfg, workspace, args.workers, args.device, args.threads,
                               args.only_repeat, args.only_fold, args.max_tasks)
    elif args.command == "status":
        from .report import load_study
        from .runner import completed
        plan, _ = load_study(workspace, args.study)
        from .runner import task_directory
        count = sum(completed(workspace, t, verify=False) for t in plan["tasks"].values())
        active, failed = [], []
        for task in plan["tasks"].values():
            directory = task_directory(workspace, task["id"])
            if (directory/"completed.json").exists():
                continue
            if (directory/"failure.json").exists():
                failed.append({"method":task["method"],"repeat":task["repeat"],"fold":task["fold"],
                               **json.loads((directory/"failure.json").read_text())})
            elif (directory/"progress.json").exists():
                active.append({"method":task["method"],"repeat":task["repeat"],"fold":task["fold"],
                               **json.loads((directory/"progress.json").read_text())})
        result = {"study": args.study, "completed_tasks": count, "planned_tasks": len(plan["tasks"]),
                  "completion_percent": round(100*count/max(1,len(plan["tasks"])),2),"counts": plan["counts"],
                  "latest_unfinished_progress":active,"failed_tasks":failed,
                  "progress_note":"Progress is the last recorded update; an interrupted task can leave a stale progress record."}
    elif args.command in ("audit", "report"):
        from .report import audit, report
        result = audit(workspace, args.study) if args.command == "audit" else report(workspace, args.study)
    else:
        from .report import report
        for study in ("primary", "pool_sizes"):
            report(workspace, study)  # Fails closed if ANY required job is missing.
        destination = Path(args.destination).resolve()
        files = [(workspace / "reports" / "primary" / "tables" / n, destination / "tables" / n)
                 for n in ("robustness_all_table.tex", "robustness_main_table.tex", "ablation_factorial_table.tex")]
        files.append((workspace / "reports" / "pool_sizes" / "tables" / "unlabelled_pool_table_compact.tex",
                      destination / "tables" / "unlabelled_pool_table_compact.tex"))
        extra = workspace / "reports" / "pool_sizes" / "tables" / "unlabelled_pool_by_group.tex"
        if extra.exists():
            files.append((extra, destination / "tables" / extra.name))
        files.append((workspace / "reports" / "primary" / "numeric_facts.json", destination / "numeric_facts.json"))
        files.append((workspace / "reports" / "primary" / "paper_numeric_summary.csv", destination / "paper_numeric_summary.csv"))
        for study in ("primary", "pool_sizes"):
            files.append((workspace / "reports" / study / "ASSET_PROVENANCE.json", destination / f"{study}_provenance.json"))
        files.extend((p, destination / "figures" / p.name) for p in (workspace / "reports" / "primary" / "figures").iterdir())
        if not args.overwrite and any(dst.exists() for _, dst in files):
            raise FileExistsError("Destination assets exist. Use a new folder or --overwrite after reviewing them.")
        for src, dst in files:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        atomic_json(destination / "RESULTS_STATUS.json", {"protocol": PROTOCOL, "complete": True,
                    "source": "completed_cv10x5_tasks", "n_splits": json.loads((workspace / "studies/primary/config.json").read_text())["cv"]["n_splits"],
                    "n_repeats": json.loads((workspace / "studies/primary/config.json").read_text())["cv"]["n_repeats"]})
        result = {"destination": str(destination), "files": len(files)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
