"""Published state/event perturbations with explicit boundary and padding semantics.

The underlying event operators are attributed AFC-RobustBench code. The adapter
conditions event reconstruction on the recorded initial snapshot and excludes
delayed-window padding from observed samples.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
from filelock import FileLock
from afc_robustness.perturbations import MissingEvents, SpuriousEvents, TimingUncertainty
from afc_robustness.representations import series_to_episode
from afc_robustness.domain import AlarmEpisode, EventType
from .common import atomic_json, seed_from, digest, file_hash


@dataclass(frozen=True)
class Observation:
    values: np.ndarray
    valid_length: int
    operator: str
    severity: float


def reconstruct_snapshot(episode: AlarmEpisode, initial: np.ndarray, length: int, dt: float) -> np.ndarray:
    """Forward-only notification interpretation conditioned on a trusted t=0 snapshot.

    A leading RTN on an inactive tag is redundant, not evidence of past activity.
    A leading RTN on an initially active tag is valid. Notifications are not
    imputed. RTN loss may legitimately prolong active state. Notifications clipped
    to t=0 by timing corruption affect sample 1 onward; they never overwrite the
    independently observed initial snapshot. Stable event ordering breaks ties.
    """
    initial = np.asarray(initial, dtype=np.uint8)
    if initial.shape != (episode.n_tags,) or length < 1 or dt <= 0:
        raise ValueError("Invalid initial snapshot / reconstruction grid")
    result = np.repeat(initial[:, None], length, axis=1)
    active = initial.astype(bool).copy()
    for event in sorted(episode.events, key=lambda e: (e.timestamp, e.order)):
        state = event.event_type == EventType.ACT
        if bool(active[event.tag]) == state:
            continue
        active[event.tag] = state
        start = max(1, int(np.ceil(event.timestamp / dt - 1e-10)))
        if start < length:
            result[event.tag, start:] = int(state)
    return result


class Corruptor:
    """Label-independent transformations; every random state is caller-supplied."""
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt = float(cfg["data"]["sample_period_minutes"])
        self.c = cfg["corruption"]
        self.policy = self.c["event_initial_state"]
        self.ops = {"missing": MissingEvents(), "spurious": SpuriousEvents(),
                    "timing": TimingUncertainty(max_shift=self.c["timing_max_shift_minutes"])}

    def apply_with_length(self, x: np.ndarray, operator: str, severity: float,
                          rng: np.random.Generator) -> Observation:
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] < 1 or not np.isin(x, [0, 1]).all():
            raise ValueError("Expected channels x samples binary observation")
        if not np.isfinite(severity) or not 0 <= severity <= .5:
            raise ValueError("Implemented evaluation support is 0 <= severity <= 0.5")
        C, T = x.shape
        original = x.astype(np.float32, copy=True)
        if operator not in {"clean", "false_full", "missing_full", "late_start", *self.ops}:
            raise ValueError(f"Unknown perturbation: {operator}")
        if severity == 0 or operator == "clean":
            return Observation(original, T, operator, float(severity))
        if operator in ("false_full", "missing_full"):
            number = int(np.floor(severity * C))
            if number:
                selected = rng.choice(C, size=number, replace=False)
                original[selected, :] = 1 if operator == "false_full" else 0
            return Observation(original, T, operator, float(severity))
        if operator == "late_start":
            # ETFA late detection by time cropping, without an extra deactivation rule.
            count = min(int(np.floor(severity * T)), T - 1)
            remaining = T - count
            result = np.zeros_like(original)
            result[:, :remaining] = original[:, count:]
            return Observation(result, remaining, operator, float(severity))
        if self.policy == "known_snapshot":
            # The snapshot is not a notification and therefore not in the event budget.
            ep = series_to_episode(original, dt=self.dt, initial_active_policy="ignore")
            transformed = self.ops[operator].apply(ep, severity, rng)
            result = reconstruct_snapshot(transformed, original[:, 0], T, self.dt)
        else:
            raise ValueError(f"Unknown event reconstruction policy {self.policy}")
        return Observation(result.astype(np.float32), T, operator, float(severity))

    def apply(self, x, operator, severity, rng):
        """Matrix interface; consumers of delayed windows must also retain valid_length."""
        return self.apply_with_length(x, operator, severity, rng).values

    def training_view(self, x, rng, stage="pretraining") -> Observation:
        if stage != "pretraining":
            raise ValueError("Supervised corruption is disabled: only pretraining views are supported")
        params = self.c["pretraining"]
        if rng.random() < params["identity_probability"]:
            return Observation(np.asarray(x, dtype=np.float32).copy(), x.shape[1], "clean", 0.)
        op = str(rng.choice(params["operators"], p=params["probabilities"]))
        s = float(rng.uniform(params["severity_min"], params["severity_max"]))
        if s > .30:
            raise AssertionError("Training exceeded the frozen 30% corruption ceiling")
        if op == "late_start":
            raise AssertionError("Delay is test-only: latent patch targets must remain aligned")
        return self.apply_with_length(x, op, s, rng)

    def training(self, x, rng):
        return self.training_view(x, rng, "pretraining").values


def test_conditions(cfg):
    yield "clean", 0., 0
    for op in cfg["test"]["operators"]:
        for s in cfg["test"]["severities"]:
            if s == 0:
                continue
            for draw in range(cfg["test"]["draws"]):
                yield op, float(s), draw


def input_effects(x: np.ndarray, observation: Observation) -> dict:
    """Measure actual changes, distinguishing excluded history from changed states."""
    T = x.shape[-1]
    L = observation.valid_length
    ref = x[:, T - L:] if observation.operator == "late_start" else x[:, :L]
    observed = observation.values[:, :L]
    before = np.any(ref, axis=1)
    after = np.any(observed, axis=1)
    return {"valid_samples": int(L), "lost_samples": int(T-L),
            "changed_state_cells": int(np.count_nonzero(ref != observed)),
            "added_active_cells": int(np.count_nonzero((ref == 0) & (observed == 1))),
            "removed_active_cells": int(np.count_nonzero((ref == 1) & (observed == 0))),
            "lost_active_tags": int(np.count_nonzero(before & ~after)),
            "gained_active_tags": int(np.count_nonzero(~before & after)),
            "unchanged_observation": int(L == T and np.array_equal(x, observation.values))}


def ensure_test_cache(data, part, cfg, workspace: Path) -> Path:
    """Source-paired test inputs, validity lengths and observed input effects.

    Caches are shared by *all* neural models, baselines, label and pool budgets.
    Cache identities include snapshot and padding semantics, source identities,
    test settings and input hashes; incompatible cached observations are rejected.
    """
    import json
    import pandas as pd
    workspace = Path(workspace)
    directory = workspace / "cache" / "test"
    directory.mkdir(parents=True, exist_ok=True)
    r, f = part["repeat"], part["fold"]
    condition_list = list(test_conditions(cfg))
    identity = {"protocol": cfg["protocol"], "test": cfg["test"], "corruption": cfg["corruption"],
                "part": part, "sampling_period": cfg["data"]["sample_period_minutes"],
                "ids": data.ids[part["test"]].tolist(), "hashes": data.hashes[part["test"]].tolist()}
    ident = digest(identity)
    path = directory / f"r{r:02d}_f{f:02d}_{ident[:16]}.npz"
    meta_path = path.with_suffix(".json")
    effects_path = path.with_suffix(".effects.csv")
    with FileLock(str(path) + ".lock"):
        if path.exists() and meta_path.exists() and effects_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["identity"] != identity or meta["sha256"] != file_hash(path) or meta["effects_sha256"] != file_hash(effects_path):
                raise RuntimeError("Paired test cache identity/hash mismatch")
            return path
        indices = np.asarray(part["test"])
        clean = data.X[indices]
        observations = np.empty((len(condition_list), *clean.shape), dtype=np.uint8)
        lengths = np.empty((len(condition_list), len(indices)), dtype=np.int32)
        effect_rows = []
        corruptor = Corruptor(cfg)
        for j, (op, s, draw) in enumerate(condition_list):
            for i, source_index in enumerate(indices):
                rng = np.random.default_rng(seed_from(cfg["test"]["seed"], r, f, str(data.ids[source_index]), op, s, draw))
                result = corruptor.apply_with_length(clean[i], op, s, rng)
                observations[j, i] = result.values
                lengths[j, i] = result.valid_length
                effect_rows.append({"repeat": r, "fold": f, "source_id": str(data.ids[source_index]),
                                    "operator": op, "severity": s, "draw": draw,
                                    **input_effects(clean[i], result)})
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, X=observations, lengths=lengths,
                                conditions=np.asarray([f"{op}|{s:g}|{draw}" for op, s, draw in condition_list]),
                                ids=data.ids[indices], y=data.y[indices], classes=data.classes)
        os.replace(tmp, path)
        pd.DataFrame(effect_rows).to_csv(effects_path, index=False)
        atomic_json(meta_path, {"identity": identity, "sha256": file_hash(path),
                               "effects_sha256": file_hash(effects_path), "conditions": condition_list,
                               "shape": list(observations.shape), "test_indices": indices.tolist()})
    return path
