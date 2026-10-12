"""Plan repeated stratified CV before any model sees data.

Preparing a plan hashes input matrices and selects roles; it does not fit models,
construct test corruptions, or calculate experimental scores.
"""
from __future__ import annotations
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from filelock import FileLock
from afc_excerpt_ssl.data import Dataset, load_processed, nested_labels
from .common import atomic_json, digest, file_hash, seed_from, source_hash, environment
from .config import METHODS, BASELINES, data_path, validate_config, pretraining_method, training_settings


def partitions(data: Dataset, cv: dict) -> list[dict]:
    """Exactly one test appearance/source/repetition; duplicates stay together."""
    if len(set(data.ids.tolist())) != len(data.ids):
        raise ValueError("This dataset must contain one matrix per uniquely identified source recording")
    n_splits, repeats = cv["n_splits"], cv["n_repeats"]
    unique = len(np.unique(data.hashes)) == len(data.hashes)
    if unique:
        iterator = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=repeats, random_state=cv["partition_seed"]).split(np.zeros(len(data.y)), data.y)
        generated = ((i // n_splits, i % n_splits, tr, te) for i, (tr, te) in enumerate(iterator))
        splitter_name = "RepeatedStratifiedKFold"
    else:
        def groups():
            for rep in range(repeats):
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_from(cv["partition_seed"], "split", rep))
                for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(data.y)), data.y, data.hashes)):
                    yield rep, fold, tr, te
        generated = groups()
        splitter_name = "RepeatedStratifiedGroupKFold_exact_input_groups"
    out = []
    for rep, fold, tr, te in generated:
        if set(data.hashes[tr]) & set(data.hashes[te]):
            raise AssertionError("Duplicate observation crosses train and test roles")
        if set(data.y[tr]) != set(data.y) or set(data.y[te]) != set(data.y):
            raise ValueError("Every fold must represent each class in both train and test")
        _, first = np.unique(data.hashes[tr], return_index=True)
        out.append({"repeat": rep, "fold": fold, "train": tr.tolist(), "test": te.tolist(),
                    "ssl_full": sorted(tr[first].tolist()), "splitter": splitter_name})
    for rep in range(repeats):
        counts = np.bincount([i for p in out if p["repeat"] == rep for i in p["test"]], minlength=len(data.y))
        if not np.all(counts == 1):
            raise AssertionError("Each source must occur once in the test set per repetition")
    return out


def pool_indices(part: dict, n: int | str, draw: int, cv: dict) -> list[int]:
    """Nested class-blind permutations; labelled IDs are never passed here."""
    full = np.asarray(part["ssl_full"], dtype=int)
    if n == "full":
        return full.tolist()
    if n > len(full):
        raise ValueError(f"Requested {n} pretraining sources, only {len(full)} available")
    rng = np.random.default_rng(seed_from(cv["pool_seed"], part["repeat"], part["fold"], draw))
    return sorted(rng.permutation(full)[:n].tolist())


def build_plan(cfg: dict, data: Dataset, context_id: str) -> dict:
    validate_config(cfg)
    study, cv = cfg["study"], cfg["cv"]
    parts = partitions(data, cv)
    tasks = {}
    evaluations = []
    def add(spec: dict) -> str:
        ident = spec["kind"] + "_" + digest(spec)[:24]
        tasks[ident] = {"id": ident, **spec}
        return ident
    for part in parts:
        r, f = part["repeat"], part["fold"]
        for pool_n in study["pool_sizes"]:
            pool_draws = [0] if pool_n == "full" else study["pool_draws"]
            for pool_draw in pool_draws:
                ssl = pool_indices(part, pool_n, pool_draw, cv)
                pool_tag = "full" if pool_n == "full" else str(pool_n)
                for model_rep in study["model_seeds"]:
                    init_seed = seed_from(cv["training_seed"], r, f, model_rep, "encoder")
                    pretrain_ids = {}
                    for method in study["methods"]:
                        if method in BASELINES or not METHODS[method]["pretrain"]:
                            continue
                        pretrain_ids[method] = add({
                            "kind": "pretrain", "context": context_id, "method": pretraining_method(method),
                            "repeat": r, "fold": f, "model_rep": model_rep,
                            "pool": pool_tag, "pool_draw": pool_draw,
                            "pool_size": len(ssl), "ssl_indices": ssl, "initialization_seed": init_seed,
                            "crop_seed": seed_from(cv["training_seed"], r, f, model_rep, "crop"),
                            "mask_seed": seed_from(cv["training_seed"], r, f, model_rep, "mask"),
                            "corruption_seed": seed_from(cv["training_seed"], r, f, model_rep, "pretrain_corruption"),
                            "dropout_seed": seed_from(cv["training_seed"], r, f, model_rep, "pretrain_dropout"),
                        })
                    for label_draw in study["label_draws"]:
                        labels = nested_labels(data, np.asarray(part["train"]), study["label_budgets"],
                            seed_from(cv["label_seed"], r, f, label_draw))
                        for k, labelled in labels.items():
                            for method in study["methods"]:
                                isbase = method in BASELINES
                                # Classical references do not have stochastic neural initialization.
                                if isbase and model_rep != min(study["model_seeds"]):
                                    continue
                                if (isbase or not METHODS[method]["pretrain"]) and pool_n != "full":
                                    raise ValueError("Pool sensitivity compares pretrained methods only")
                                spec = {
                                    "kind": "baseline" if isbase else "fit", "context": context_id,
                                    "method": method, "repeat": r, "fold": f, "model_rep": 0 if isbase else model_rep,
                                    "label_draw": label_draw, "k": int(k), "label_indices": labelled.tolist(),
                                    "pool": pool_tag, "pool_draw": pool_draw, "pool_size": len(ssl),
                                    "initialization_seed": seed_from(cv["training_seed"], r, f, 0, "encoder") if isbase else init_seed,
                                }
                                if not isbase:
                                    # Label-draw changes only the selected sources; all training RNG seeds
                                    # stay fixed when model_rep is fixed (crossed variability diagnostic).
                                    spec.update({
                                        "dependency": pretrain_ids.get(method),
                                        "head_seed": seed_from(cv["training_seed"], r, f, model_rep, k, "head"),
                                        "batch_seed": seed_from(cv["training_seed"], r, f, model_rep, k, "batch"),
                                        "corruption_seed": seed_from(cv["training_seed"], r, f, model_rep, k, "supervised_corruption"),
                                        "dropout_seed": seed_from(cv["training_seed"], r, f, model_rep, k, "supervised_dropout"),
                                    })
                                ident = add(spec)
                                evaluations.append(ident)
    evaluations = sorted(set(evaluations))
    # Check roles again independently of method expansion.
    lookup = {(p["repeat"], p["fold"]): p for p in parts}
    for task in tasks.values():
        part = lookup[(task["repeat"], task["fold"])]
        inds = task.get("ssl_indices", task.get("label_indices", []))
        if not set(inds) <= set(part["train"]) or set(inds) & set(part["test"]):
            raise AssertionError("A planned fit includes held-out data")
    counts = {kind: sum(t["kind"] == kind for t in tasks.values()) for kind in ("pretrain", "fit", "baseline")}
    return {"protocol": cfg["protocol"], "study": study["name"], "context": context_id,
            "status": "planned", "counts": counts, "n_splits_total": len(parts),
            "tasks": tasks, "evaluations": evaluations, "parts": parts}


def prepare(cfg: dict, workspace: Path) -> dict:
    """Create or verify a study plan. No model construction or evaluation."""
    workspace = workspace.resolve()
    path = data_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    data = load_processed(path, cfg["data"]["observation_samples"])
    if cfg["data"].get("expected_recordings") and len(data.X) != cfg["data"]["expected_recordings"]:
        raise ValueError("Unexpected number of recordings")
    if cfg["data"].get("expected_channels") and data.X.shape[1] != cfg["data"]["expected_channels"]:
        raise ValueError("Unexpected channel count")
    common = deepcopy(cfg)
    common.pop("study")
    identity = {"config": common, "data_sha256": file_hash(path), "source_sha256": source_hash()}
    context_id = digest(identity)
    workspace.mkdir(parents=True, exist_ok=True)
    with FileLock(str(workspace / ".prepare.lock")):
        context_path = workspace / "context.json"
        if context_path.exists():
            old = json.loads(context_path.read_text())
            if old["id"] != context_id:
                raise ValueError("Workspace has different source/data/protocol settings. Use a new workspace; old results will not be overwritten.")
        else:
            atomic_json(context_path, {"id": context_id, **identity, "planning_environment": environment(),
                                      "data_shape": list(data.X.shape), "classes": data.classes.tolist()})
            atomic_json(workspace / "dataset_index.json", {"ids": data.ids.tolist(), "hashes": data.hashes.tolist(),
                                                           "labels": data.y.tolist(), "classes": data.classes.tolist(), "tags": data.tags.tolist()})
        plan = build_plan(cfg, data, context_id)
        study_dir = workspace / "studies" / cfg["study"]["name"]
        plan_path = study_dir / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("Study plan already exists with a different specification")
        else:
            atomic_json(plan_path, plan)
            atomic_json(study_dir / "config.json", cfg)
        # All studies share identical source-level partitions and test draws.
        atomic_json(workspace / "splits.json", plan["parts"])
    return plan
