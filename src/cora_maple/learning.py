"""Clean-only classifier fine-tuning and masked latent pretraining.

All three pretraining conditions share the model and within-scope schedules. Sampling,
corruption, initialization, dropout and label-order streams are separate. Test
results are never used for LR scheduling, checkpoint selection or early stopping.
"""
from __future__ import annotations
import os
import random
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from afc_excerpt_ssl.models import PatchEncoder, Classifier, LatentPretrainer
from afc_excerpt_ssl.data import sample_crops
from .config import METHODS, encoder_kwargs, training_settings
from .transforms import Corruptor
from .common import atomic_json


def configure_randomness(seed: int, threads: int, device: str) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; install a matching PyTorch build or use cpu")
        torch.cuda.set_device(torch.device("cuda:0" if device == "cuda" else device))
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def cosine_learning_rate(step: int, total: int, peak: float, warmup: int, floor: float) -> float:
    """Linear warmup followed by cosine decay, with LR bounded below by floor*peak."""
    warmup = min(warmup, max(0, total - 1))
    if warmup and step < warmup:
        return peak * (.03 + .97 * (step + 1) / warmup)
    phase = min(1., max(0., (step - warmup) / max(1, total - warmup)))
    return peak * (floor + (1 - floor) * .5 * (1 + np.cos(np.pi * phase)))


class ControlledPretrainer(LatentPretrainer):
    def __init__(self, encoder: PatchEncoder, objective: dict):
        super().__init__(encoder, ema=objective["ema"], raw_weight=objective["raw_weight"], var_weight=objective["var_weight"])
        self.gamma, self.epsilon = objective["gamma"], objective["epsilon"]

    def loss(self, clean, view, mask, objective="latent", lengths=None):
        if objective != "latent":
            raise ValueError("Only the complete MAPLE pretraining objective belongs to this design")
        h = self.student(view, mask, lengths=lengths)
        raw = self.student.patchify(clean, lengths)
        valid_cells = self.student.cell_mask(clean, lengths)
        cells = F.binary_cross_entropy_with_logits(self.decoder(h), raw, reduction="none")
        positive = valid_cells * raw
        negative = valid_cells * (1 - raw)
        rec = .5 * ((cells * positive).sum() / positive.sum().clamp_min(1)
                    + (cells * negative).sum() / negative.sum().clamp_min(1))
        pooled = self.student.mean_tokens(h, view, lengths)
        std = torch.sqrt(pooled.var(0, unbiased=False) + self.epsilon)
        var = F.relu(self.gamma - std).mean()
        with torch.no_grad():
            target = self.teacher(clean, lengths=lengths)
        prediction = F.layer_norm(self.predictor(h), (h.shape[-1],))
        weights = self.student.patch_weights(clean, lengths)
        # Per-input normalization avoids overweighting a longer input when variable
        # supplied lengths are used. Equal-length minibatches reduce to mean squared error.
        per_patch = ((prediction - target) ** 2).mean(-1)
        latent = ((per_patch * weights).sum(1) / weights.sum(1).clamp_min(1e-12)).mean()
        total = latent + self.raw_weight * rec + self.var_weight * var
        return total, {"latent": latent.detach().item(), "reconstruction": rec.detach().item(),
                       "variance": var.detach().item(), "embedding_std_mean": std.detach().mean().item()}


def make_mask(batch: int, patches: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    if patches < 1 or batch < 1:
        raise ValueError("A minibatch needs real patches")
    out = np.zeros((batch, patches), dtype=bool)
    count = min(patches, max(1, round(patches * ratio)))
    for row in out:
        row[rng.choice(patches, size=count, replace=False)] = True
    return out


def cpu_state(module: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _log(directory, records, filename, stage, step, total, start):
    """Live monitoring is task-local and is not used for checkpoint selection."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(directory / filename, index=False)
    atomic_json(directory / "progress.json", {"stage": stage, "step": step, "total_steps": total,
                                             "elapsed_seconds": time.perf_counter() - start,
                                             "last_loss": records[-1].get("loss", records[-1].get("minibatch_loss"))})


def pretrain_model(X: np.ndarray, cfg: dict, task: dict, directory: Path, device: str, threads: int) -> None:
    configure_randomness(task["initialization_seed"], threads, device)
    method, train = METHODS[task["method"]], training_settings(cfg, task["method"])
    if not method["pretrain"]:
        raise ValueError("Scratch cannot be pretrained")
    model = ControlledPretrainer(PatchEncoder(X.shape[1], **encoder_kwargs(cfg)), cfg["objective"]).to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=train["pretrain_lr"],
                                 weight_decay=train["pretrain_weight_decay"], betas=tuple(train["adam_betas"]), eps=train["adam_epsilon"])
    crops_rng = np.random.default_rng(task["crop_seed"])
    mask_rng = np.random.default_rng(task["mask_seed"])
    noise_rng = np.random.default_rng(task["corruption_seed"])
    torch.manual_seed(task["dropout_seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(task["dropout_seed"])
    corruptor = Corruptor(cfg)
    lengths = [X.shape[-1]] if method["scope"] == "full" else train["crop_lengths"]
    batch, q = train["pretrain_batch"], cfg["model"]["patch_size"]
    steps = train["pretrain_steps"]
    records, sample_steps, empty = [], 0, 0
    source_draw_counts = np.zeros(len(X), dtype=np.int64)
    sample_coverage = np.zeros((len(X), X.shape[-1]), dtype=bool)
    exposures = Counter()
    severity_max = 0.
    start = time.perf_counter()
    for step in range(steps):
        length = int(crops_rng.choice(lengths))
        reference, source_indices, starts = sample_crops(X, batch, length, crops_rng)
        np.add.at(source_draw_counts, source_indices, 1)
        for src, first in zip(source_indices, starts):
            sample_coverage[src, first:first+length] = True
        if method["corruption"]:
            views = [corruptor.training_view(x, noise_rng, "pretraining") for x in reference]
            if any(v.valid_length != length for v in views):
                raise AssertionError("SSL student and reference must retain the same patch coordinates")
            view = np.stack([v.values for v in views])
            for v in views:
                exposures[v.operator] += 1
                severity_max = max(severity_max, v.severity)
        else:
            view = reference.copy()
            exposures["clean"] += batch
        mask = make_mask(batch, (length+q-1)//q, train["mask_ratio"], mask_rng)
        lr = cosine_learning_rate(step, steps, train["pretrain_lr"], train["pretrain_warmup"], train["lr_min_factor"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, components = model.loss(torch.from_numpy(reference).to(device), torch.from_numpy(view).to(device),
                                      torch.from_numpy(mask).to(device), method["objective"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite pretraining loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        model.update_teacher()
        sample_steps += batch * length
        empty += int((reference.sum((1, 2)) == 0).sum())
        if step == 0 or (step+1) % train["log_interval"] == 0 or step+1 == steps:
            records.append({"step": step+1, "loss": loss.detach().item(), "sample_steps": sample_steps,
                            "learning_rate": lr, **components})
            _log(directory, records, "pretraining.csv", "pretraining", step+1, steps, start)
    stats = {"updates": steps, "source_recordings": len(X), "input_draws": batch*steps,
             "sample_steps": sample_steps, "sample_minutes": sample_steps*cfg["data"]["sample_period_minutes"],
             "patch_tokens": sample_steps//q, "empty_inputs": empty, "seconds": time.perf_counter()-start,
             "objective": method["objective"], "corruption": method["corruption"],
             "processed_history_hours": sample_steps*cfg["data"]["sample_period_minutes"]/60,
             "distinct_sources_visited": int(np.count_nonzero(source_draw_counts)),
             "distinct_sample_steps_visited": int(sample_coverage.sum()),
             "sample_coverage_fraction": float(sample_coverage.mean()), "scope": method["scope"],
             "operator_exposures": dict(exposures), "max_training_severity_observed": severity_max,
             "event_initial_state": cfg["corruption"]["event_initial_state"],
             "schedule": {key: train[key] for key in ("pretrain_steps", "finetune_steps", "pretrain_lr", "pretrain_warmup", "lr_min_factor")}}
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/"source_coverage.npz", draw_counts=source_draw_counts,
                        sample_coverage=sample_coverage, source_indices=np.asarray(task["ssl_indices"]))
    torch.save({"state_dict": cpu_state(model.student), "model": encoder_kwargs(cfg), "n_tags": X.shape[1],
                "task": task, "statistics": stats, "protocol": cfg["protocol"]}, directory/"encoder.pt")
    atomic_json(directory/"training.json", stats)


def fit_classifier(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: dict, task: dict, directory: Path,
                   encoder_path: Path | None, device: str, threads: int) -> Classifier:
    configure_randomness(task["initialization_seed"], threads, device)
    train, method = training_settings(cfg, task["method"]), METHODS[task["method"]]
    if method["augmentation"] != "none" or train.get("supervised_objective") != "clean_cross_entropy":
        raise ValueError("Only clean-only classifier fitting belongs to this release")
    if (encoder_path is not None) != method["pretrain"]:
        raise ValueError("Classifier dependency disagrees with the selected pretraining condition")
    encoder = PatchEncoder(X.shape[1], **encoder_kwargs(cfg))
    if encoder_path is not None:
        checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)
        if checkpoint.get("protocol") != cfg["protocol"]:
            raise ValueError("A pretrained checkpoint from another protocol cannot be reused")
        encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.manual_seed(task["head_seed"])
    model = Classifier(encoder, n_classes).to(device).train()
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": train["encoder_lr"]},
                                  {"params": model.head.parameters(), "lr": train["head_lr"]}],
                                 weight_decay=train["finetune_weight_decay"], betas=tuple(train["adam_betas"]), eps=train["adam_epsilon"])
    batch_rng = np.random.default_rng(task["batch_seed"])
    torch.manual_seed(task["dropout_seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(task["dropout_seed"])
    steps = train["finetune_steps"]
    history, start = [], time.perf_counter()
    target_y = torch.as_tensor(y, dtype=torch.long, device=device)
    clean_X = torch.from_numpy(X.astype(np.float32, copy=False)).to(device)
    exposures = Counter()
    severity_max = 0.
    corrupted_steps = clean_steps = 0
    for step in range(steps):
        indices = batch_rng.choice(len(X), size=min(train["finetune_batch"], len(X)), replace=False)
        batch = X[indices].astype(np.float32, copy=True)
        labels = target_y[torch.as_tensor(indices, device=device)]
        for group, peak in zip(optimizer.param_groups, (train["encoder_lr"], train["head_lr"])):
            group["lr"] = cosine_learning_rate(step, steps, peak, train["finetune_warmup"], train["lr_min_factor"])
        optimizer.zero_grad(set_to_none=True)
        clean_logits = model(torch.from_numpy(batch).to(device))
        clean_ce = F.cross_entropy(clean_logits, labels)
        loss, corrupt_ce = clean_ce, torch.zeros((), device=device)
        clean_steps += len(batch)*batch.shape[-1]
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite fine-tuning loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step+1) % train["log_interval"] == 0 or step == steps-1:
            model.eval()
            with torch.no_grad():
                logits = model(clean_X)
                clean_loss = F.cross_entropy(logits, target_y).item()
                accuracy = (logits.argmax(1) == target_y).float().mean().item()
            history.append({"step": step+1, "minibatch_loss": loss.item(), "clean_ce": clean_ce.item(),
                            "corrupted_ce": corrupt_ce.item(), "clean_train_loss": clean_loss,
                            "clean_train_accuracy": accuracy, "encoder_lr": optimizer.param_groups[0]["lr"],
                            "head_lr": optimizer.param_groups[1]["lr"]})
            _log(directory, history, "finetuning.csv", "supervised", step+1, steps, start)
            model.train()
    model.eval()
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": cpu_state(model), "model": encoder_kwargs(cfg), "n_tags": X.shape[1],
                "n_classes": n_classes, "task": task, "protocol": cfg["protocol"]}, directory/"classifier.pt")
    atomic_json(directory/"training.json", {"seconds": time.perf_counter()-start, "updates": steps,
                "labelled_recordings": len(X), "augmentation": method["augmentation"], "scope": method["scope"],
                "train_accuracy": accuracy, "train_loss": clean_loss, "operator_exposures": dict(exposures),
                "max_training_severity_observed": severity_max, "clean_sample_steps": clean_steps,
                "corrupted_sample_steps": corrupted_steps, "event_initial_state": cfg["corruption"]["event_initial_state"],
                "schedule": {key: train[key] for key in ("finetune_steps", "encoder_lr", "head_lr", "finetune_warmup", "lr_min_factor")}})
    return model


@torch.no_grad()
def predict_neural(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 128, lengths=None) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        x = torch.from_numpy(X[start:start+batch_size].astype(np.float32)).to(device)
        valid = None if lengths is None else torch.as_tensor(lengths[start:start+batch_size], device=device, dtype=torch.long)
        outputs.append(model(x, lengths=valid).argmax(1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int16)
