"""Task-boundary resumable execution; preparation/reporting never train models."""
from __future__ import annotations
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
from filelock import FileLock
from threadpoolctl import threadpool_limits
from sklearn.metrics import f1_score, accuracy_score
from afc_excerpt_ssl.data import load_processed
from .common import atomic_json, file_hash, source_hash, environment
from .config import data_path, METHODS
from .protocol import prepare
from .transforms import ensure_test_cache


def task_directory(workspace: Path, ident: str) -> Path:
    return workspace / "tasks" / ident


def completed(workspace: Path, task: dict, verify: bool = True) -> bool:
    directory = task_directory(workspace, task["id"])
    path = directory / "completed.json"
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if result["task"] != task:
        raise RuntimeError(f"Task identity mismatch at {path}")
    for name, checksum in result["files"].items():
        file = directory / name
        if not file.exists() or (verify and file_hash(file) != checksum):
            raise RuntimeError(f"Completed task file is missing or modified: {file}")
    return True


@lru_cache(maxsize=2)
def worker_data(path: str, horizon: int):
    return load_processed(path, horizon)


def score_predictions(y, predicted, n_classes):
    return {"macro_f1": float(f1_score(y, predicted, labels=np.arange(n_classes), average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, predicted))}


def run_task(workspace_s: str, cfg: dict, task: dict, part: dict, device: str, threads: int) -> dict:
    """Execute one dependency-resolved training or evaluation task."""
    workspace = Path(workspace_s)
    directory = task_directory(workspace, task["id"])
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".task.lock")), threadpool_limits(limits=threads):
        if completed(workspace, task):
            return {"id": task["id"], "status": "reused"}
        atomic_json(directory / "task.json", task)
        start = time.perf_counter()
        try:
            context = json.loads((workspace / "context.json").read_text())
            if context["source_sha256"] != source_hash() or context["data_sha256"] != file_hash(data_path(cfg)):
                raise RuntimeError("Source or dataset changed after plan creation")
            data = worker_data(str(data_path(cfg)), cfg["data"]["observation_samples"])
            # Keep training inputs and held-out test inputs separate even internally.
            from .learning import pretrain_model, fit_classifier, predict_neural
            if task["kind"] == "pretrain":
                pretrain_model(data.X[task["ssl_indices"]], cfg, task, directory, device, threads)
            else:
                labels = np.asarray(task["label_indices"])
                if task["kind"] == "fit":
                    dependency = task.get("dependency")
                    encoder_path = task_directory(workspace, dependency) / "encoder.pt" if dependency else None
                    if dependency and not (task_directory(workspace, dependency) / "completed.json").exists():
                        raise RuntimeError("Pretraining dependency has not completed")
                    model = fit_classifier(data.X[labels], data.y[labels], len(data.classes), cfg, task,
                                           directory, encoder_path, device, threads)
                    predict = lambda X: predict_neural(model, X, device, cfg["test"]["inference_batch"])
                else:
                    import joblib
                    from .baselines import make_baseline, predict_baseline
                    model = make_baseline(task["method"], cfg["baselines"], task["initialization_seed"])
                    model.fit(data.X[labels], data.y[labels])
                    # Joblib is for trusted, locally generated models only.
                    joblib.dump(model, directory / "baseline.joblib", compress=3)
                    predict = lambda X: predict_baseline(model, X.astype(np.float32))
                    atomic_json(directory / "training.json", {"seconds": time.perf_counter() - start, "labelled_recordings": len(labels)})
                atomic_json(directory / "model_info.json", {"classes": data.classes.tolist(),
                    "channels": data.tags.tolist(), "observation_samples": cfg["data"]["observation_samples"],
                    "sample_period_minutes": cfg["data"]["sample_period_minutes"], "method": task["method"]})
                cache_path = ensure_test_cache(data, part, cfg, workspace)
                with np.load(cache_path, allow_pickle=False) as cache:
                    y = cache["y"]
                    predictions, rows = [], []
                    for key, X in zip(cache["conditions"], cache["X"]):
                        op, severity, draw = str(key).split("|")
                        predicted = np.asarray(predict(X), dtype=np.int16)
                        if predicted.shape != y.shape or not np.isin(predicted, np.arange(len(data.classes))).all():
                            raise AssertionError("Invalid predicted classes/shape")
                        predictions.append(predicted)
                        rows.append({"condition": str(key), "operator": op, "severity": float(severity), "draw": int(draw),
                                     **score_predictions(y, predicted, len(data.classes))})
                    np.savez_compressed(directory / "predictions.npz", predictions=np.stack(predictions),
                                        conditions=cache["conditions"], y=y, ids=cache["ids"], classes=cache["classes"])
                    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
                atomic_json(directory / "evaluation.json", {"test_cache": str(cache_path.relative_to(workspace)),
                    "test_cache_sha256": file_hash(cache_path), "n_test": len(part["test"]), "n_conditions": len(rows)})
            if context["source_sha256"] != source_hash() or context["data_sha256"] != file_hash(data_path(cfg)):
                raise RuntimeError("Source or dataset changed during task execution")
            outputs = {p.name: file_hash(p) for p in directory.iterdir()
                       if p.is_file() and p.suffix in (".json", ".csv", ".npz", ".pt", ".joblib")
                       and p.name not in ("completed.json", "failure.json")}
            atomic_json(directory / "completed.json", {"task": task, "files": outputs,
                "elapsed_seconds": time.perf_counter() - start, "device": device, "threads": threads})
            (directory / "failure.json").unlink(missing_ok=True)
            return {"id": task["id"], "status": "completed", "seconds": time.perf_counter() - start}
        except Exception as exc:
            atomic_json(directory / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            raise


def run_study(cfg: dict, workspace: Path, workers: int = 1, device: str = "cpu", threads: int = 1,
              only_repeat: int | None = None, only_fold: int | None = None, max_tasks: int | None = None) -> dict:
    if max_tasks is not None and max_tasks < 0:
        raise ValueError("max_tasks must be nonnegative")
    if not (device == "cpu" or device == "cuda" or device.startswith("cuda:")):
        raise ValueError("Experiment device must be cpu or cuda[:index]")
    if workers < 1 or threads < 1:
        raise ValueError("Worker and thread counts must be positive")
    if device.startswith("cuda") and workers != 1:
        raise ValueError("Use one worker per GPU invocation. Shard by --only-repeat/--only-fold for multiple devices.")
    plan = prepare(cfg, workspace)
    context = json.loads((workspace / "context.json").read_text())
    if context["source_sha256"] != source_hash():
        raise RuntimeError("Source changed since plan creation")
    runtime = {"environment": environment(), "device_family": "cuda" if device.startswith("cuda") else device,
               "threads_per_worker": threads}
    with FileLock(str(workspace / ".runtime.lock")):
        path = workspace / "execution_environment.json"
        if path.exists() and json.loads(path.read_text()) != runtime:
            raise ValueError("Do not mix environments, CPU/GPU results, or thread policies in one workspace")
        if not path.exists():
            atomic_json(path, runtime)
    tasks = [t for t in plan["tasks"].values()
             if (only_repeat is None or t["repeat"] == only_repeat) and (only_fold is None or t["fold"] == only_fold)]
    if not tasks:
        raise ValueError("Selection contains no tasks")
    parts = {(p["repeat"], p["fold"]): p for p in plan["parts"]}
    done_count = 0
    outcome = []
    # All selected pretraining dependencies finish before any selected classifier fit.
    for phase in ("pretrain", "evaluation"):
        pending = [t for t in tasks if ((t["kind"] == "pretrain") == (phase == "pretrain")) and not completed(workspace, t)]
        pending.sort(key=lambda t: (t["repeat"], t["fold"], t.get("pool_size", 0), t["method"], t.get("k", 0)))
        if max_tasks is not None:
            pending = pending[:max(0, max_tasks - done_count)]
        if workers == 1:
            for task in pending:
                result = run_task(str(workspace), cfg, task, parts[(task["repeat"], task["fold"])], device, threads)
                outcome.append(result)
                done_count += 1
                print(f"{result['status']}: {task['method']} r={task['repeat']} f={task['fold']} k={task.get('k','-')} N_U={task['pool_size']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
                futures = {executor.submit(run_task, str(workspace), cfg, t, parts[(t["repeat"], t["fold"])], device, threads): t for t in pending}
                for future in as_completed(futures):
                    result = future.result()
                    task = futures[future]
                    outcome.append(result)
                    done_count += 1
                    print(f"{result['status']}: {task['method']} r={task['repeat']} f={task['fold']} k={task.get('k','-')} N_U={task['pool_size']}", flush=True)
        if max_tasks is not None and done_count >= max_tasks:
            break
    status = {"newly_executed_tasks": done_count, "study": plan["study"],
              "complete": all(completed(workspace, t, verify=False) for t in plan["tasks"].values())}
    atomic_json(workspace / "studies" / plan["study"] / "last_run_status.json", status)
    return status
