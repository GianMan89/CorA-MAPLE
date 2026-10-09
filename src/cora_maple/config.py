"""Validated, explicit experiment configurations and ablation definitions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import yaml
from .common import ROOT

# 'reconstruction_variance' removes only the latent term; it keeps the remaining
# coefficients of CorA-MAPLE. It is NOT the pure reconstruction-only control.
METHODS = {
    "scratch": {"label": "Scratch", "pretrain": False, "augmentation": "none"},
    "scratch_aug": {"label": "Scratch+Aug", "pretrain": False, "augmentation": "bank"},
    "scratch_aug_online": {"label": "Scratch+AugOnline", "pretrain": False, "augmentation": "online"},
    "scratch_long": {"label": "Scratch-1200", "pretrain": False, "augmentation": "none", "finetune_steps": 1200},
    "scratch_aug_long": {"label": "Scratch+Aug-1200", "pretrain": False, "augmentation": "bank", "finetune_steps": 1200},
}
for suffix, scope in (("excerpt", "excerpt"), ("full", "full")):
    letter = "E" if scope == "excerpt" else "F"
    for stem, label, objective, corruption in (
        ("recon", "Recon", "reconstruction", False),
        ("cora_recon", "CorA-Recon", "reconstruction", True),
        ("cora_recon_var", "CorA-Recon+Var", "reconstruction_variance", True),
        ("maple", "MAPLE", "latent", False),
        ("cora_maple", "CorA-MAPLE", "latent", True),
    ):
        METHODS[f"{stem}_{suffix}"] = {
            "label": f"{label}-{letter}", "pretrain": True, "scope": scope,
            "objective": objective, "corruption": corruption, "augmentation": "none",
        }
METHODS["cora_maple_excerpt_matched"] = {
    **METHODS["cora_maple_excerpt"], "label": "CorA-MAPLE-E-TimeMatch", "exposure_matched": True,
}
BASELINES = ("EAC-1NN", "MBW-LR", "JAC-1NN", "WDI-1NN", "ACM-SVM")
OPERATORS = ("missing", "spurious", "timing", "burst", "tag_dropout", "mixed")
PROPOSED = ("cora_maple_excerpt", "cora_maple_full")
PRIMARY_NEURAL = (
    "scratch", "scratch_aug", "recon_excerpt", "maple_excerpt", "maple_full", *PROPOSED,
)

def merge(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out

def load_config(path: str | Path, _seen: set | None = None) -> dict:
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError("Circular configuration inheritance")
    seen.add(path)
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("Configuration must be a mapping")
    parent = content.pop("extends", None)
    cfg = merge(load_config(path.parent / parent, seen), content) if parent else content
    # Only fully resolved study configurations require experiment-specific checks.
    if "study" in cfg:
        validate_config(cfg)
    return cfg

def validate_config(cfg: dict) -> None:
    required = ("protocol", "data", "cv", "model", "objective", "training", "corruption", "test", "baselines", "study")
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing configuration section: {key}")
    if cfg["protocol"] != "cora_maple_cv10x5_v1":
        raise ValueError("Unknown protocol identifier")
    cv, model, train, obj = (cfg[k] for k in ("cv", "model", "training", "objective"))
    if cv["n_splits"] < 2 or cv["n_repeats"] < 1:
        raise ValueError("Invalid CV dimensions")
    q = model["patch_size"]
    horizon = cfg["data"]["observation_samples"]
    if q < 1 or horizon < q or horizon % q:
        raise ValueError("Observation length must be divisible by patch_size")
    if cfg["data"]["sample_period_minutes"] <= 0:
        raise ValueError("Sampling period must be positive")
    if model["n_heads"] != 4 or model["feedforward_multiplier"] != 2 or model["activation"] != "gelu":
        raise ValueError("This encoder implements four heads, FFN width 2d, and GELU")
    if model["d_model"] % 4 or model["d_model"] < 4:
        raise ValueError("d_model must be a positive multiple of four")
    if not 0 <= model["dropout"] < 1 or model["n_layers"] < 1:
        raise ValueError("Invalid encoder configuration")
    lengths = train["crop_lengths"]
    if not lengths or len(lengths) != len(set(lengths)) or any(l < q or l > horizon or l % q for l in lengths):
        raise ValueError("Unique excerpt lengths must fit observation and patch grid")
    if train["pretrain_batch"] < 2 or train["finetune_batch"] < 1:
        raise ValueError("Variance loss requires pretraining batch >= 2")
    if not 0 < train["mask_ratio"] < 1 or min(train["pretrain_steps"], train["finetune_steps"]) < 1:
        raise ValueError("Invalid mask or update budget")
    if not 0 <= obj["ema"] < 1 or min(obj["raw_weight"], obj["var_weight"]) < 0:
        raise ValueError("Invalid loss coefficient")
    if min(obj["gamma"], obj["epsilon"]) <= 0:
        raise ValueError("Variance threshold and stabilizer must be positive")
    for key in ("pretrain_lr", "finetune_lr", "gradient_clip_norm"):
        if train[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if train["weight_decay"] < 0:
        raise ValueError("Weight decay must be nonnegative")
    cor = cfg["corruption"]
    if not 0 <= cor["identity_probability"] <= 1:
        raise ValueError("identity_probability must be in [0,1]")
    names, probs = cor["operators"], cor["probabilities"]
    if not names or len(names) != len(probs) or len(names) != len(set(names)):
        raise ValueError("Specify unique corruption operators and matching probabilities")
    if not set(names) <= set(OPERATORS) or any(p < 0 for p in probs) or not math.isclose(sum(probs), 1., abs_tol=1e-10):
        raise ValueError("Invalid operator distribution")
    if not 0 <= cor["severity_min"] < cor["severity_max"] < 1:
        raise ValueError("Invalid training severity support")
    if min(cor["timing_max_shift_minutes"], cor["burst_half_width_minutes"]) <= 0:
        raise ValueError("Time-scale parameters must be positive")
    immutable = {
        "initial_active_policy": "act", "repair": "retained_AFC_RobustBench_TraceRepair",
        "mixed_order": ["missing", "spurious"], "mixed_repair_between_stages": True,
        "tag_budget": "floor_severity_times_all_channels", "event_budget": "floor_severity_times_current_event_count",
    }
    if any(cor.get(key) != val for key, val in immutable.items()):
        raise ValueError("Changing retained event/repair semantics requires a new implemented protocol, not just a YAML edit")
    if len(train["adam_betas"]) != 2 or any(not 0 <= b < 1 for b in train["adam_betas"]) or train["adam_epsilon"] <= 0:
        raise ValueError("Invalid AdamW settings")
    if train["augmentation_bank_size"] < 1 or train["log_interval"] < 1:
        raise ValueError("Invalid augmentation bank size/log interval")
    test = cfg["test"]
    if test["draws"] < 1 or test["inference_batch"] < 1 or not test["operators"] or not set(test["operators"]) <= set(OPERATORS):
        raise ValueError("Invalid test scenario definition")
    if 0 not in test["severities"] or any(not 0 <= s < 1 for s in test["severities"]):
        raise ValueError("Test grid must include zero and stay in [0,1)")
    if len(test["severities"]) != len(set(test["severities"])) or len(test["operators"]) != len(set(test["operators"])):
        raise ValueError("Duplicate test conditions")
    study = cfg["study"]
    if study["kind"] not in ("primary", "pool", "variability", "diagnostic"):
        raise ValueError("Unknown study kind")
    if not study["methods"] or len(study["methods"]) != len(set(study["methods"])) or not set(study["methods"]) <= set(METHODS) | set(BASELINES):
        raise ValueError("Unknown/empty method selection")
    for name in ("label_budgets", "label_draws", "model_seeds", "pool_draws"):
        vals = study[name]
        minimum = 1 if name == "label_budgets" else 0
        if not vals or len(vals) != len(set(vals)) or min(vals) < minimum:
            raise ValueError(f"Invalid {name}")
    sizes = study["pool_sizes"]
    if not sizes or len(sizes) != len(set(sizes)) or any(s != "full" and (not isinstance(s, int) or s < 1) for s in sizes):
        raise ValueError("pool_sizes must be positive counts or 'full'")
    if study["kind"] == "pool" and study["label_budgets"] != [1]:
        raise ValueError("The prespecified pool study fixes k=1")

def data_path(cfg: dict) -> Path:
    path = Path(cfg["data"]["path"])
    return path if path.is_absolute() else ROOT / path

def encoder_kwargs(cfg: dict) -> dict:
    return {k: cfg["model"][k] for k in ("d_model", "patch_size", "n_layers", "dropout")}
