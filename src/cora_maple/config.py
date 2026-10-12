"""Three pretraining conditions per E/F schedule; clean-only supervised fitting."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import yaml
from .common import ROOT

PROTOCOL = "cora_maple_clean_ft_cv10x5_v2_1"
BASELINES = ("EAC-1NN", "MBW-LR", "JAC-1NN", "WDI-1NN", "ACM-SVM")
# No burst, duplicate tag-dropout, ad-hoc reordering or new perturbation mechanism.
OPERATORS = ("false_full", "missing_full", "timing", "late_start", "missing", "spurious")
OPERATOR_LABELS = {
    "false_full": "Full-channel false alarms", "missing_full": "Full-channel loss",
    "timing": "Timestamp uncertainty", "late_start": "Delayed observation",
    "missing": "Missing event notifications", "spurious": "Spurious event notifications",
}
METHODS = {}
for suffix, letter in (("excerpt", "E"), ("full", "F")):
    for stem, label, pretrain, corruption in (
        ("scratch", "Scratch", False, False),
        ("maple", "MAPLE", True, False),
        ("cora_maple", "CorA-MAPLE", True, True),
    ):
        METHODS[f"{stem}_{suffix}"] = {
            "label": f"{label}-{letter}", "scope": suffix, "pretrain": pretrain,
            "objective": "latent", "corruption": corruption,
            "augmentation": "none",
            "pretraining_state": "corrupted" if corruption else "clean" if pretrain else "none",
        }
PROPOSED = ("cora_maple_excerpt", "cora_maple_full")
PRIMARY_NEURAL = tuple(METHODS)


def merge(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for key, value in update.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else deepcopy(value)
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
    if "study" in cfg:
        validate_config(cfg)
    return cfg


def training_settings(cfg: dict, method: str) -> dict:
    """All three pretraining conditions within a scope share supervised update/LR settings."""
    train = deepcopy(cfg["training"])
    schedule = train.pop("schedules")[METHODS[method]["scope"]]
    return merge(train, schedule)


def pretraining_method(method: str) -> str:
    """One SSL checkpoint per scope/state, shared by all labelled adaptation budgets."""
    m = METHODS[method]
    if not m["pretrain"]:
        raise ValueError("Scratch has no SSL checkpoint")
    return ("cora_maple_" if m["corruption"] else "maple_") + m["scope"]


def _positive(v, name, allow_zero=False):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or (v < 0 if allow_zero else v <= 0):
        raise ValueError(f"Invalid {name}: {v!r}")


def _distribution(stage: dict, name: str, allowed: set):
    names, p = stage["operators"], stage["probabilities"]
    if not names or len(names) != len(p) or len(names) != len(set(names)) or not set(names) <= allowed:
        raise ValueError(f"Invalid {name} operator list")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in p) or not math.isclose(sum(p), 1., abs_tol=1e-10):
        raise ValueError(f"Invalid {name} probabilities")
    if not 0 <= stage["severity_min"] < stage["severity_max"] <= .30:
        raise ValueError(f"{name} must keep all training severities in [0, 0.30]")
    if not 0 <= stage.get("ceiling_probability", 0) <= 1:
        raise ValueError("Invalid ceiling probability")


def validate_config(cfg: dict) -> None:
    for key in ("protocol", "data", "cv", "model", "objective", "training", "corruption", "test", "baselines", "study", "reporting"):
        if key not in cfg:
            raise ValueError(f"Missing configuration section: {key}")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Protocol mismatch. The report or workspace must match this experiment configuration.")
    cv, m, t, obj = (cfg[k] for k in ("cv", "model", "training", "objective"))
    if not isinstance(cv["n_splits"],int) or not isinstance(cv["n_repeats"],int) or cv["n_splits"] < 2 or cv["n_repeats"] < 1:
        raise ValueError("Invalid CV dimensions")
    if any(not isinstance(cv[k], int) or cv[k] < 0 for k in ("partition_seed", "label_seed", "training_seed", "pool_seed")):
        raise ValueError("Seeds must be nonnegative integers")
    q = m["patch_size"]
    horizon = cfg["data"]["observation_samples"]
    if not isinstance(q, int) or q < 1 or not isinstance(horizon, int) or horizon < q:
        raise ValueError("Invalid sample/patch length")
    _positive(cfg["data"]["sample_period_minutes"], "sampling period")
    if m["n_heads"] != 4 or m["feedforward_multiplier"] != 2 or m["activation"] != "gelu":
        raise ValueError("Encoder implements four heads, FFN width 2d and GELU")
    if m["d_model"] < 4 or m["d_model"] % 4 or not 0 <= m["dropout"] < 1 or m["n_layers"] < 1:
        raise ValueError("Invalid encoder configuration")
    if m.get("padding_policy") != "explicit_valid_lengths":
        raise ValueError("Explicit length-aware attention, pooling and losses are required")
    lengths = t["crop_lengths"]
    if not lengths or len(lengths) != len(set(lengths)) or any(not isinstance(x, int) or x < q or x > horizon or x % q for x in lengths):
        raise ValueError("Unique excerpt lengths must fit the observation and patch grid")
    if t["pretrain_batch"] < 2 or t["finetune_batch"] < 1 or not 0 < t["mask_ratio"] < 1:
        raise ValueError("Invalid training batch or patch-mask fraction")
    for scope in ("excerpt", "full"):
        schedule = t["schedules"][scope]
        for key in ("pretrain_steps", "finetune_steps"):
            if not isinstance(schedule[key], int) or schedule[key] < 1:
                raise ValueError(f"Invalid {scope} {key}")
    for key in ("pretrain_lr", "encoder_lr", "head_lr", "gradient_clip_norm", "adam_epsilon"):
        _positive(t[key], key)
    for key in ("pretrain_weight_decay", "finetune_weight_decay"):
        _positive(t[key], key, allow_zero=True)
    if len(t["adam_betas"]) != 2 or any(not 0 <= b < 1 for b in t["adam_betas"]):
        raise ValueError("Invalid AdamW betas")
    if not 0 < t["lr_min_factor"] <= 1 or min(t["pretrain_warmup"], t["finetune_warmup"]) < 0 or t["log_interval"] < 1:
        raise ValueError("Invalid LR schedule/log interval")
    if t.get("supervised_objective") != "clean_cross_entropy":
        raise ValueError("The classifier objective must be clean_cross_entropy")
    if "paired_clean_weight" in t or "supervised" in cfg["corruption"]:
        raise ValueError("Supervised corruption settings are not part of the clean-only protocol")
    if not 0 <= obj["ema"] < 1 or min(obj["raw_weight"], obj["var_weight"]) < 0 or min(obj["gamma"], obj["epsilon"]) <= 0:
        raise ValueError("Invalid pretraining objective")
    c = cfg["corruption"]
    if c["event_initial_state"] != "known_snapshot":
        raise ValueError("Unknown event reconstruction observation model")
    if c["delay_policy"] != "crop_rebase_mask_padding":
        raise ValueError("Delay must not invent a suffix of observed inactivity")
    _positive(c["timing_max_shift_minutes"], "timing shift")
    _distribution(c["pretraining"], "pretraining", set(OPERATORS) - {"late_start", "missing", "spurious"})
    if not 0 <= c["pretraining"]["identity_probability"] <= 1:
        raise ValueError("Invalid SSL identity probability")
    # Delayed observations and event missing/spurious are test-only mechanisms.
    tst = cfg["test"]
    if tst["draws"] < 1 or tst["inference_batch"] < 1 or not tst["operators"] or not set(tst["operators"]) <= set(OPERATORS):
        raise ValueError("Invalid test operator set")
    if len(tst["operators"]) != len(set(tst["operators"])):
        raise ValueError("Duplicate test operators")
    if sorted(tst["severities"]) != tst["severities"] or len(tst["severities"]) != len(set(tst["severities"])) or 0 not in tst["severities"] or any(not 0 <= s <= .5 for s in tst["severities"]):
        raise ValueError("Test severity grid must be sorted, unique, contain zero, and not exceed 0.50")
    study = cfg["study"]
    if study["kind"] not in ("primary", "pool", "variability", "diagnostic"):
        raise ValueError("Unknown study kind")
    if not study["methods"] or len(study["methods"]) != len(set(study["methods"])) or not set(study["methods"]) <= set(METHODS) | set(BASELINES):
        raise ValueError("Unknown/empty/duplicate method list")
    for key in ("label_budgets", "label_draws", "model_seeds", "pool_draws"):
        a = study[key]
        if not a or len(a) != len(set(a)) or any(not isinstance(v, int) or v < (1 if key == "label_budgets" else 0) for v in a):
            raise ValueError(f"Invalid {key}")
    sizes = study["pool_sizes"]
    if not sizes or len(sizes) != len(set(sizes)) or any(s != "full" and (not isinstance(s, int) or s < 1) for s in sizes):
        raise ValueError("pool_sizes must be positive counts or full")
    if study["kind"] == "pool" and study["label_budgets"] != [1]:
        raise ValueError("Prespecified pool study fixes k=1")
    if study["kind"] in ("primary", "pool") and cfg["reporting"]["headline_k"] not in study["label_budgets"]:
        raise ValueError("The headline label budget must be in the study")
    if cfg["reporting"]["headline_severity"] not in tst["severities"] or cfg["reporting"]["figure_operator"] not in tst["operators"]:
        raise ValueError("Report conditions must belong to the test grid")


def data_path(cfg: dict) -> Path:
    p = Path(cfg["data"]["path"])
    return p if p.is_absolute() else ROOT / p


def encoder_kwargs(cfg: dict) -> dict:
    return {key: cfg["model"][key] for key in ("d_model", "patch_size", "n_layers", "dropout")}


def operator_groups(cfg: dict) -> dict[str, list[str]]:
    """Groups describe CorA-MAPLE's pretraining exposure, not that of every ablation.

    All methods use clean-only supervised fitting. Delay is therefore test-only,
    alongside missing/spurious notifications; no operator is supervised-trained.
    """
    pre = set(cfg["corruption"]["pretraining"]["operators"])
    tests = cfg["test"]["operators"]
    return {"pretraining_only": [o for o in tests if o in pre],
            "trained": [o for o in tests if o in pre],
            "test_only": [o for o in tests if o not in pre]}
