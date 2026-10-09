"""Configurable event corruption using the retained, unmodified operators."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from filelock import FileLock
from afc_robustness.perturbations import MissingEvents, SpuriousEvents, TimingUncertainty, BurstSpuriousEvents, TagDropout, MixedPerturbation
from afc_robustness.repair import TraceRepair
from afc_robustness.representations import series_to_episode, episode_to_series
from .common import atomic_json, seed_from, digest, file_hash


class Corruptor:
    """A new view per call; independent RNG passed explicitly by the caller."""
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dt = float(cfg["data"]["sample_period_minutes"])
        c = cfg["corruption"]
        self.identity_probability = c["identity_probability"]
        self.names = tuple(c["operators"])
        self.probabilities = np.asarray(c["probabilities"], dtype=float)
        self.low, self.high = c["severity_min"], c["severity_max"]
        self.ops = {
            "missing": MissingEvents(), "spurious": SpuriousEvents(),
            "timing": TimingUncertainty(max_shift=c["timing_max_shift_minutes"]),
            "burst": BurstSpuriousEvents(half_width=c["burst_half_width_minutes"]),
            "tag_dropout": TagDropout(),
            "mixed": MixedPerturbation((MissingEvents(), SpuriousEvents()), name="missing_then_spurious"),
        }
        self.repair = TraceRepair()

    def apply(self, x: np.ndarray, operator: str, severity: float, rng: np.random.Generator) -> np.ndarray:
        if operator not in self.ops or not 0 <= severity < 1:
            raise ValueError("Unknown operator or invalid severity")
        if severity == 0:
            return x.astype(np.float32, copy=True)
        events = series_to_episode(x, dt=self.dt, initial_active_policy="act")
        changed = self.ops[operator].apply(events, float(severity), rng)
        repaired = self.repair.apply(changed)
        result = episode_to_series(repaired, dt=self.dt, max_length=x.shape[-1]).astype(np.float32)
        if result.shape != x.shape:
            raise AssertionError("Corruption changed observation dimensions")
        return result

    def training(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.identity_probability:
            return x.astype(np.float32, copy=True)
        op = str(rng.choice(self.names, p=self.probabilities))
        return self.apply(x, op, float(rng.uniform(self.low, self.high)), rng)


def supervised_batch(X: np.ndarray, indices: np.ndarray, mode: str, corruptor: Corruptor,
                     rng: np.random.Generator, bank: np.ndarray | None = None) -> np.ndarray:
    """Construct clean, fixed-bank, or freshly corrupted labelled minibatches.

    Minibatch source selection uses a separate RNG in learning.py, so selecting
    online vs bank augmentation cannot change the labelled-example order.
    """
    if mode == "none":
        return X[indices].astype(np.float32, copy=True)
    if mode == "online":
        return np.stack([corruptor.training(X[i], rng) for i in indices])
    if mode == "bank" and bank is not None:
        return bank[int(rng.integers(len(bank))), indices].astype(np.float32, copy=True)
    raise ValueError(f"Invalid augmentation mode/bank: {mode}")


def test_conditions(cfg: dict) -> list[tuple[str, float, int]]:
    conditions = [("clean", 0., 0)]
    for op in cfg["test"]["operators"]:
        for severity in sorted(cfg["test"]["severities"]):
            if severity > 0:
                conditions.extend((op, float(severity), d) for d in range(cfg["test"]["draws"]))
    return conditions


def ensure_test_cache(data, part: dict, cfg: dict, workspace: Path) -> Path:
    """One immutable paired suite per split, shared by all studies and methods."""
    r, f = part["repeat"], part["fold"]
    cache_dir = workspace / "cache" / "test"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"repeat_{r:02d}_fold_{f:02d}.npz"
    meta_path = path.with_suffix(".json")
    condition_list = test_conditions(cfg)
    identity = digest({"test_indices": part["test"], "test": cfg["test"], "corruption": cfg["corruption"],
                       "dt": cfg["data"]["sample_period_minutes"], "repeat": r, "fold": f,
                       "hashes": data.hashes[part["test"]].tolist()})
    with FileLock(str(path) + ".lock"):
        if path.exists() and meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            if meta["identity"] != identity or meta["sha256"] != file_hash(path):
                raise RuntimeError("Paired test cache identity/hash mismatch; remove only the corrupted cache files")
            return path
        corruptor = Corruptor(cfg)
        indices = np.asarray(part["test"])
        clean = data.X[indices]
        array = np.empty((len(condition_list), *clean.shape), dtype=np.uint8)
        for j, (op, s, draw) in enumerate(condition_list):
            if op == "clean":
                array[j] = clean
                continue
            for i, source_index in enumerate(indices):
                rng = np.random.default_rng(seed_from(cfg["test"]["seed"], r, f, str(data.ids[source_index]), op, s, draw))
                array[j, i] = corruptor.apply(clean[i], op, s, rng)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, X=array, conditions=np.asarray([f"{op}|{s:g}|{d}" for op, s, d in condition_list]),
                                ids=data.ids[indices], y=data.y[indices], classes=data.classes)
        os.replace(tmp, path)
        atomic_json(meta_path, {"identity": identity, "sha256": file_hash(path), "conditions": condition_list,
                               "shape": list(array.shape), "test_indices": indices.tolist()})
    return path
