"""Repeated-CV aggregation and manuscript assets from completed tasks."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, source_hash, file_hash
from .config import METHODS, BASELINES, PROPOSED, PROTOCOL, validate_config, operator_groups, OPERATOR_LABELS
from .runner import completed, task_directory, score_predictions
from .transforms import test_conditions

METRICS = ["macro_f1", "accuracy"]
SPLIT_KEYS = ["repeat", "fold", "method", "k", "pool", "pool_size", "operator", "severity"]
REPEAT_KEYS = ["repeat", "method", "k", "pool", "pool_size", "operator", "severity"]
SUMMARY_KEYS = ["method", "k", "pool", "pool_size", "operator", "severity"]
PAIRS = []
for scope in ("excerpt", "full"):
    PAIRS += [(f"maple_{scope}", f"scratch_{scope}"),
              (f"cora_maple_{scope}", f"maple_{scope}"),
              (f"cora_maple_{scope}", f"scratch_{scope}")]
PAIRS += [(p, baseline) for p in PROPOSED for baseline in BASELINES]
PAIRS += [("cora_maple_full", "cora_maple_excerpt")]


def load_study(workspace: Path, study: str):
    folder = workspace / "studies" / study
    plan = json.loads((folder / "plan.json").read_text())
    cfg = json.loads((folder / "config.json").read_text())
    # Never silently filter a saved study to obtain apparent completeness.
    validate_config(cfg)
    if plan.get("protocol") != PROTOCOL:
        raise ValueError("Stored plan does not match the clean-only protocol; incompatible results cannot complete this study")
    allowed = set(METHODS) | set(BASELINES)
    if any(task["method"] not in allowed for task in plan["tasks"].values()):
        raise ValueError("Stored plan contains unsupported methods")
    if any(ident not in plan["tasks"] for ident in plan["evaluations"]):
        raise ValueError("Stored plan is missing an evaluation task")
    return plan, cfg


def audit(workspace: Path, study: str, recompute: bool = True) -> dict:
    """Validate stored predictions, role assignments and condition coverage."""
    plan, cfg = load_study(workspace, study)
    index = json.loads((workspace / "dataset_index.json").read_text())
    parts = {(p["repeat"], p["fold"]): p for p in plan["parts"]}
    missing, checked_rows = [], 0
    expected_conditions = 1 + len(cfg["test"]["operators"]) * sum(s > 0 for s in cfg["test"]["severities"]) * cfg["test"]["draws"]
    for ident, task in plan["tasks"].items():
        part = parts[(task["repeat"], task["fold"])]
        chosen = task.get("ssl_indices", task.get("label_indices", []))
        if not set(chosen) <= set(part["train"]):
            raise AssertionError("Training/test role violation")
        if set(index["hashes"][i] for i in part["train"]) & set(index["hashes"][i] for i in part["test"]):
            raise AssertionError("Exact inputs cross training and test")
        if not completed(workspace, task):
            missing.append(ident)
            continue
        if task["kind"] == "pretrain":
            continue
        folder = task_directory(workspace, ident)
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            if z["ids"].tolist() != [index["ids"][i] for i in part["test"]]:
                raise AssertionError("Prediction IDs do not match held-out source order")
            if z["y"].tolist() != [index["labels"][i] for i in part["test"]]:
                raise AssertionError("Prediction labels mismatch")
            expected_keys = [f"{op}|{s:g}|{d}" for op, s, d in test_conditions(cfg)]
            if z["conditions"].tolist() != expected_keys:
                raise AssertionError("Prediction condition identities/order do not match the planned suite")
            if z["classes"].tolist() != index["classes"]:
                raise AssertionError("Prediction class-name mapping mismatch")
            if "lengths" not in z.files or z["lengths"].shape != (expected_conditions, len(part["test"])):
                raise AssertionError("Missing observation-validity metadata")
            if np.any(z["lengths"] < 1) or np.any(z["lengths"] > cfg["data"]["observation_samples"]):
                raise AssertionError("Invalid supplied observation lengths")
            if not np.isin(z["predictions"], np.arange(len(index["classes"]))).all():
                raise AssertionError("Invalid predicted classes")
            if z["predictions"].shape != (expected_conditions, len(part["test"])):
                raise AssertionError("Incomplete condition coverage")
            if recompute:
                metrics = pd.read_csv(folder / "metrics.csv").set_index("condition")
                for condition, predicted in zip(z["conditions"], z["predictions"]):
                    scores = score_predictions(z["y"], predicted, len(z["classes"]))
                    for metric, val in scores.items():
                        if not np.isclose(val, metrics.loc[str(condition), metric], atol=1e-12, rtol=0):
                            raise AssertionError(f"Metric mismatch: {ident}, {condition}, {metric}")
                    checked_rows += 1
    report = {"study": study, "complete": not missing, "planned_tasks": len(plan["tasks"]),
              "missing_tasks": missing, "recomputed_metric_rows": checked_rows,
              "scope": "Task hashes, planned source roles, IDs and saved-prediction arithmetic; not retraining or statistical superiority."}
    atomic_json(workspace / "studies" / study / "audit.json", report)
    return report


def aggregate_frames(rows: pd.DataFrame):
    """MC -> model/label/pool -> fold -> repetition; no fold-independence CI."""
    fit_keys = ["task_id", "repeat", "fold", "method", "k", "pool", "pool_size", "pool_draw", "label_draw", "model_rep", "operator", "severity"]
    per_fit = rows.groupby(fit_keys, as_index=False, dropna=False)[METRICS].mean()
    per_split = per_fit.groupby(SPLIT_KEYS, as_index=False, dropna=False)[METRICS].mean()
    per_repeat = per_split.groupby(REPEAT_KEYS, as_index=False, dropna=False)[METRICS].mean()
    summary = per_repeat.groupby(SUMMARY_KEYS, as_index=False, dropna=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"), n_repeats=("repeat", "nunique"))
    return per_fit, per_split, per_repeat, summary


def paired_differences(per_fit: pd.DataFrame):
    frames = []
    keys = ["repeat", "fold", "k", "pool", "pool_size", "pool_draw", "label_draw", "model_rep", "operator", "severity"]
    methods = set(per_fit["method"])
    for a, b in PAIRS:
        if not {a, b} <= methods:
            continue
        joined = per_fit.loc[per_fit.method == a, keys + ["macro_f1"]].merge(
            per_fit.loc[per_fit.method == b, keys + ["macro_f1"]], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        if joined.empty:
            continue
        joined["difference"] = joined.macro_f1_a - joined.macro_f1_b
        joined["method_a"], joined["method_b"] = a, b
        frames.append(joined)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    per_pair = pd.concat(frames, ignore_index=True)
    base = ["method_a", "method_b", "k", "pool", "pool_size", "operator", "severity"]
    split_mean = per_pair.groupby(base + ["repeat", "fold"], as_index=False).difference.mean()
    repeats = split_mean.groupby(base + ["repeat"], as_index=False).difference.mean()
    summary = repeats.groupby(base, as_index=False).agg(mean_difference=("difference", "mean"),
                sd_difference=("difference", "std"), minimum_repeat_difference=("difference", "min"),
                maximum_repeat_difference=("difference", "max"), positive_repetitions=("difference", lambda x: int((x > 0).sum())),
                n_repeats=("difference", "count"))
    return per_pair, summary


def augment_robustness_summaries(per_fit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = per_fit[per_fit.operator == "clean"][["task_id", "macro_f1"]].rename(columns={"macro_f1": "clean_macro_f1"})
    drops = per_fit[per_fit.operator != "clean"].merge(clean, on="task_id", validate="many_to_one")
    drops["absolute_drop"] = drops.clean_macro_f1 - drops.macro_f1
    records = []
    clean_lookup = clean.set_index("task_id").clean_macro_f1.to_dict()
    for (task, op), frame in drops.groupby(["task_id", "operator"]):
        frame = frame.sort_values("severity")
        s = np.r_[0., frame.severity.to_numpy()]
        scores = np.r_[clean_lookup[task], frame.macro_f1.to_numpy()]
        row = frame.iloc[0].drop(labels=["severity", "macro_f1", "accuracy", "clean_macro_f1", "absolute_drop"]).to_dict()
        row.update(operator=op, severity_auc=float(np.trapezoid(scores, s) / s[-1]))
        records.append(row)
    return drops, pd.DataFrame(records)


def method_label(name: str) -> str:
    return METHODS[name]["label"] if name in METHODS else name


def latex_name(name: str) -> str:
    return r"\emph{" + method_label(name).replace("_", r"\_") + "}"


def value(row: pd.Series) -> str:
    return f"${row.macro_f1_mean:.3f}{{\\pm}}{row.macro_f1_sd:.3f}$"


def lookup(summary, method, k, op, severity, pool="full"):
    frame = summary[(summary.method == method) & (summary.k == k) & (summary.operator == op)
                    & (np.isclose(summary.severity, severity)) & (summary.pool.astype(str) == str(pool))]
    if len(frame) != 1:
        raise ValueError(f"Expected one result for {method}, k={k}, op={op}, s={severity}, pool={pool}; got {len(frame)}")
    return frame.iloc[0]


def make_assets(summary: pd.DataFrame, cfg: dict, folder: Path):
    from .assets import make_assets as write_assets
    write_assets(summary, cfg, folder)


def plot_profiles(summary, cfg, out):
    from .plotting import render_profiles
    render_profiles(summary, cfg, out)


def report(workspace: Path, study: str) -> dict:
    validation = audit(workspace, study, recompute=True)
    if not validation["complete"]:
        raise RuntimeError(f"Study incomplete: {len(validation['missing_tasks'])} tasks missing. No publication assets generated.")
    plan, cfg = load_study(workspace, study)
    frames = []
    for ident in plan["evaluations"]:
        task = plan["tasks"][ident]
        frame = pd.read_csv(task_directory(workspace, ident) / "metrics.csv")
        for key in ("repeat", "fold", "method", "k", "pool", "pool_size", "pool_draw", "label_draw", "model_rep"):
            frame[key] = task[key]
        frame["task_id"] = ident
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    per_fit, per_split, per_repeat, summary = aggregate_frames(rows)
    if not (summary.n_repeats == cfg["cv"]["n_repeats"]).all():
        raise AssertionError("Unexpected repetition coverage")
    folder = workspace / "reports" / study
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in (("metrics_by_draw", rows), ("per_fit", per_fit), ("per_split", per_split),
                        ("per_repeat", per_repeat), ("summary", summary)):
        frame.to_csv(folder / (name + ".csv"), index=False)
    paired, paired_summary = paired_differences(per_fit)
    paired.to_csv(folder / "paired_per_fit.csv", index=False)
    paired_summary.to_csv(folder / "paired_summary.csv", index=False)
    drops, auc = augment_robustness_summaries(per_fit)
    drops.to_csv(folder / "absolute_drops.csv", index=False)
    auc.to_csv(folder / "severity_auc_per_fit.csv", index=False)
    # Preserve dispersion concealed by within-split averaging, including crossed diagnostics.
    if cfg["study"]["kind"] == "variability":
        fixed = ["repeat", "fold", "method", "k", "pool", "operator", "severity"]
        init_sd = per_fit.groupby(fixed + ["label_draw"], as_index=False).agg(
            mean=("macro_f1", "mean"), sd_across_initializations=("macro_f1", "std"), count=("macro_f1", "count"))
        label_sd = per_fit.groupby(fixed + ["model_rep"], as_index=False).agg(
            mean=("macro_f1", "mean"), sd_across_label_draws=("macro_f1", "std"), count=("macro_f1", "count"))
        init_sd.to_csv(folder / "fixed_labels_initialization_variability.csv", index=False)
        label_sd.to_csv(folder / "fixed_initialization_label_variability.csv", index=False)
    if cfg["study"]["kind"] == "pool":
        match = ["repeat", "fold", "method", "k", "operator", "severity"]
        full = per_split[per_split.pool == "full"][match + ["macro_f1"]].rename(columns={"macro_f1": "full_macro_f1"})
        pool_pairs = per_split[per_split.pool != "full"].merge(full, on=match, validate="many_to_one")
        pool_pairs["difference_to_full"] = pool_pairs.macro_f1 - pool_pairs.full_macro_f1
        pool_pairs.to_csv(folder / "pool_to_full_per_split.csv", index=False)
    write_exposure_report(workspace, plan, cfg, folder)
    write_group_summaries(per_split, cfg, folder)
    write_input_effects(workspace, plan, folder)
    make_assets(summary, cfg, folder)
    atomic_json(folder / "config.json", cfg)
    numerical_files = ("metrics_by_draw.csv", "per_fit.csv", "per_split.csv", "per_repeat.csv", "summary.csv", "config.json")
    atomic_json(folder / "REPORT_STATUS.json", {"protocol": cfg["protocol"], "complete": True, "audit": validation, "source": "workspace_tasks",
        "configuration": "config.json", "files": {name: file_hash(folder/name) for name in numerical_files}})
    return {"study": study, "output": str(folder), "metric_rows": len(rows), "n_repeat_means": cfg["cv"]["n_repeats"]}


def write_exposure_report(workspace: Path, plan: dict, cfg: dict, folder: Path) -> None:
    """Export available data, sampled duration and source-coverage diagnostics."""
    records = []
    for ident, task in plan['tasks'].items():
        if task['kind'] != 'pretrain':
            continue
        statistics = json.loads((task_directory(workspace, ident) / 'training.json').read_text())
        records.append({'task_id': ident, 'repeat': task['repeat'], 'fold': task['fold'],
                        'method': task['method'], 'pool': task['pool'],
                        'pool_draw': task['pool_draw'], 'model_rep': task['model_rep'],
                        'available_recordings': task['pool_size'],
                        'available_history_hours': task['pool_size'] * cfg['data']['observation_samples'] * cfg['data']['sample_period_minutes'] / 60,
                        **statistics})
    if records:
        pd.DataFrame(records).to_csv(folder / 'pretraining_exposure.csv', index=False)


def write_group_summaries(per_split: pd.DataFrame, cfg: dict, folder: Path) -> None:
    """Operator-balanced groups are averaged within each fold, then repetitions.

    This is not a pooled industrial prevalence model. Group membership is defined
    by the proposed methods' planned exposure; no performance-based filtering.
    """
    groups = operator_groups(cfg)
    groups["all_corruptions"] = cfg["test"]["operators"]
    split_frames = []
    base = ["repeat", "fold", "method", "k", "pool", "pool_size", "severity"]
    for name in ("trained", "test_only", "all_corruptions"):
        ops = groups[name]
        if not ops:
            continue
        q = per_split[per_split.operator.isin(ops)]
        counts = q.groupby(base, dropna=False).operator.nunique()
        if not (counts == len(ops)).all():
            raise ValueError(f"Incomplete operator-balanced group: {name}")
        g = q.groupby(base, as_index=False)[METRICS].mean()
        g["group"] = name
        split_frames.append(g)
    if not split_frames:
        return
    g = pd.concat(split_frames, ignore_index=True)
    g.to_csv(folder/"group_per_split.csv", index=False)
    keys = ["method", "k", "pool", "pool_size", "severity", "group"]
    r = g.groupby(keys+["repeat"], as_index=False)[METRICS].mean()
    r.to_csv(folder/"group_per_repeat.csv", index=False)
    su = r.groupby(keys, as_index=False).agg(macro_f1_mean=("macro_f1", "mean"),
          macro_f1_sd=("macro_f1", "std"), accuracy_mean=("accuracy", "mean"),
          accuracy_sd=("accuracy", "std"), n_repeats=("repeat", "nunique"))
    su.to_csv(folder/"group_summary.csv", index=False)


def write_input_effects(workspace: Path, plan: dict, folder: Path) -> None:
    """One source-level perturbation audit per split, never duplicated per model."""
    paths = set()
    for ident in plan["evaluations"]:
        p = task_directory(workspace, ident)/"evaluation.json"
        if p.exists():
            info = json.loads(p.read_text())
            paths.add((workspace/info["test_cache"]).with_suffix(".effects.csv"))
    frames = [pd.read_csv(p) for p in sorted(paths)]
    if frames:
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(folder/"test_input_effects_per_source.csv", index=False)
        keys = ["repeat", "fold", "operator", "severity", "draw"]
        metrics = [c for c in raw.columns if c not in keys+["source_id"]]
        raw.groupby(keys, as_index=False)[metrics].mean().to_csv(folder/"test_input_effects_per_draw.csv", index=False)
