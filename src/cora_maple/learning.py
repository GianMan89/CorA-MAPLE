"""Pretraining and supervised adaptation with independent random streams."""
from __future__ import annotations
import copy
import os
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from afc_excerpt_ssl.models import PatchEncoder, Classifier, LatentPretrainer
from afc_excerpt_ssl.data import sample_crops
from .config import METHODS, encoder_kwargs
from .transforms import Corruptor, supervised_batch
from .common import seed_from, atomic_json


def configure_randomness(seed: int, threads: int, device: str) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is unavailable; install a matching PyTorch build or use --device cpu")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


class ControlledPretrainer(LatentPretrainer):
    """Retained network with explicit compound/reconstruction/no-latent objectives."""
    def __init__(self, encoder: PatchEncoder, objective: dict):
        super().__init__(encoder, ema=objective["ema"], raw_weight=objective["raw_weight"], var_weight=objective["var_weight"])
        self.gamma = objective["gamma"]
        self.epsilon = objective["epsilon"]

    def loss(self, clean, view, mask, objective="latent"):
        if objective not in ("latent", "reconstruction", "reconstruction_variance"):
            raise ValueError("Unknown pretraining objective")
        h = self.student(view, mask)
        raw_targets = self.student.patchify(clean)
        cells = F.binary_cross_entropy_with_logits(self.decoder(h), raw_targets, reduction="none")
        rec = .5 * ((cells * raw_targets).sum() / raw_targets.sum().clamp_min(1)
                    + (cells * (1 - raw_targets)).sum() / (1 - raw_targets).sum().clamp_min(1))
        pooled = h.mean(1)
        std = torch.sqrt(pooled.var(0, unbiased=False) + self.epsilon)
        var = F.relu(self.gamma - std).mean()
        latent = h.new_zeros(())
        if objective == "latent":
            with torch.no_grad():
                target = self.teacher(clean)
            prediction = F.layer_norm(self.predictor(h), (h.shape[-1],))
            latent = F.mse_loss(prediction, target)
            total = latent + self.raw_weight * rec + self.var_weight * var
        elif objective == "reconstruction":
            total = rec  # Genuine reconstruction-only control, no variance term.
        else:
            total = self.raw_weight * rec + self.var_weight * var  # Only L_lat removed.
        return total, {"latent": latent.detach().item(), "reconstruction": rec.detach().item(),
                       "variance": var.detach().item(), "embedding_std_mean": std.detach().mean().item()}


def make_mask(batch: int, patches: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((batch, patches), dtype=bool)
    count = max(1, round(patches * ratio))
    for row in out:
        row[rng.choice(patches, size=count, replace=False)] = True
    return out


def cpu_state(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def pretrain_model(X: np.ndarray, cfg: dict, task: dict, directory: Path, device: str, threads: int) -> None:
    configure_randomness(task["initialization_seed"], threads, device)
    method, train = METHODS[task["method"]], cfg["training"]
    model = ControlledPretrainer(PatchEncoder(X.shape[1], **encoder_kwargs(cfg)), cfg["objective"]).to(device).train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=train["pretrain_lr"],
                                 weight_decay=train["weight_decay"], betas=tuple(train["adam_betas"]), eps=train["adam_epsilon"])
    crops_rng = np.random.default_rng(task["crop_seed"])
    mask_rng = np.random.default_rng(task["mask_seed"])
    noise_rng = np.random.default_rng(task["corruption_seed"])
    torch.manual_seed(task["dropout_seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(task["dropout_seed"])
    corruptor = Corruptor(cfg)
    lengths = [X.shape[-1]] if method["scope"] == "full" else train["crop_lengths"]
    batch, q = train["pretrain_batch"], cfg["model"]["patch_size"]
    target_samples = train["pretrain_steps"] * batch * X.shape[-1] if method.get("exposure_matched") else None
    records, sample_steps, step, empty = [], 0, 0, 0
    source_draw_counts = np.zeros(len(X), dtype=np.int64)
    sample_coverage = np.zeros((len(X), X.shape[-1]), dtype=bool)
    start = time.perf_counter()
    while (sample_steps < target_samples) if target_samples is not None else (step < train["pretrain_steps"]):
        length = int(crops_rng.choice(lengths))
        if target_samples is not None:
            length = min(length, (target_samples - sample_steps) // batch)
        if length < q or length % q:
            raise AssertionError("Invalid final exposure-matched crop")
        reference, source_indices, starts = sample_crops(X, batch, length, crops_rng)
        np.add.at(source_draw_counts, source_indices, 1)
        for source_index, first in zip(source_indices, starts):
            sample_coverage[source_index, first:first + length] = True
        view = np.stack([corruptor.training(x, noise_rng) for x in reference]) if method["corruption"] else reference.copy()
        mask = make_mask(batch, length // q, train["mask_ratio"], mask_rng)
        optimizer.zero_grad(set_to_none=True)
        loss, components = model.loss(torch.from_numpy(reference).to(device), torch.from_numpy(view).to(device),
                                      torch.from_numpy(mask).to(device), method["objective"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite pretraining loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        if method["objective"] == "latent":
            model.update_teacher()
        step += 1
        sample_steps += batch * length
        empty += int((reference.sum((1, 2)) == 0).sum())
        if step == 1 or step % train["log_interval"] == 0 or (target_samples is None and step == train["pretrain_steps"]) or (target_samples is not None and sample_steps == target_samples):
            records.append({"step": step, "loss": loss.detach().item(), "sample_steps": sample_steps, **components})
    stats = {"updates": step, "source_recordings": len(X), "input_draws": batch * step,
             "sample_steps": sample_steps, "sample_minutes": sample_steps * cfg["data"]["sample_period_minutes"],
             "patch_tokens": sample_steps // q, "empty_inputs": empty, "seconds": time.perf_counter() - start,
             "objective": method["objective"], "corruption": method["corruption"],
             "processed_history_hours": sample_steps * cfg["data"]["sample_period_minutes"] / 60,
             "distinct_sources_visited": int(np.count_nonzero(source_draw_counts)),
             "distinct_sample_steps_visited": int(sample_coverage.sum()),
             "sample_coverage_fraction": float(sample_coverage.mean())}
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "source_coverage.npz", draw_counts=source_draw_counts, sample_coverage=sample_coverage, source_indices=np.asarray(task["ssl_indices"]))
    torch.save({"state_dict": cpu_state(model.student), "model": encoder_kwargs(cfg), "n_tags": X.shape[1],
                "task": task, "statistics": stats}, directory / "encoder.pt")
    pd.DataFrame(records).to_csv(directory / "pretraining.csv", index=False)
    atomic_json(directory / "training.json", stats)


def fit_classifier(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: dict, task: dict, directory: Path,
                   encoder_path: Path | None, device: str, threads: int) -> Classifier:
    configure_randomness(task["initialization_seed"], threads, device)
    encoder = PatchEncoder(X.shape[1], **encoder_kwargs(cfg))
    if encoder_path is not None:
        checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.manual_seed(task["head_seed"])
    model = Classifier(encoder, n_classes).to(device).train()
    train, method = cfg["training"], METHODS[task["method"]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["finetune_lr"], weight_decay=train["weight_decay"],
                                 betas=tuple(train["adam_betas"]), eps=train["adam_epsilon"])
    batch_rng = np.random.default_rng(task["batch_seed"])
    noise_rng = np.random.default_rng(task["corruption_seed"])
    corruptor = Corruptor(cfg)
    bank = None
    if method["augmentation"] == "bank":
        bank_rng = np.random.default_rng(seed_from(task["corruption_seed"], "bank"))
        bank = np.stack([np.stack([corruptor.training(x, bank_rng) for x in X]) for _ in range(train["augmentation_bank_size"])])
    torch.manual_seed(task["dropout_seed"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(task["dropout_seed"])
    steps = method.get("finetune_steps", train["finetune_steps"])
    history, start = [], time.perf_counter()
    target_y = torch.as_tensor(y, dtype=torch.long, device=device)
    clean_X = torch.from_numpy(X).to(device)
    for step in range(steps):
        indices = batch_rng.choice(len(X), size=min(train["finetune_batch"], len(X)), replace=False)
        batch = supervised_batch(X, indices, method["augmentation"], corruptor, noise_rng, bank)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(torch.from_numpy(batch).to(device)), target_y[torch.as_tensor(indices, device=device)])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite fine-tuning loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % train["log_interval"] == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                logits = model(clean_X)
                clean_loss = F.cross_entropy(logits, target_y).item()
                accuracy = (logits.argmax(1) == target_y).float().mean().item()
            history.append({"step": step + 1, "minibatch_loss": loss.item(), "clean_train_loss": clean_loss,
                            "clean_train_accuracy": accuracy})
            model.train()
    model.eval()
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": cpu_state(model), "model": encoder_kwargs(cfg), "n_tags": X.shape[1],
                "n_classes": n_classes, "task": task}, directory / "classifier.pt")
    pd.DataFrame(history).to_csv(directory / "finetuning.csv", index=False)
    atomic_json(directory / "training.json", {"seconds": time.perf_counter() - start, "updates": steps,
                "labelled_recordings": len(X), "augmentation": method["augmentation"],
                "train_accuracy": accuracy, "train_loss": clean_loss})
    return model


@torch.no_grad()
def predict_neural(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 128) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(X), batch_size):
        x = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
        outputs.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(outputs).astype(np.int16)
