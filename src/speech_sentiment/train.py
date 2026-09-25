"""Train heads on MELD official train split; select epochs on dev macro-F1."""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from . import TASK_LABELS
from .features import SYSTEM_FEATURES, feature_path, load_vector
from .meld import read_manifest
from .model import SentimentHead


def sample_hash(rows: list[dict]) -> str:
    payload = json.dumps([(row["id"], row["label"]) for row in rows], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_split(
    manifest: Path, cache_dir: Path, system: str, *, feature_dirs: dict[str, Path] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rows = read_manifest(manifest)
    if any(row["split"] != manifest.stem for row in rows):
        raise ValueError(f"Manifest split does not match {manifest.name}")
    kinds = SYSTEM_FEATURES[system]
    vectors = []
    for row in rows:
        vectors.append(np.concatenate([
            load_vector(feature_path((feature_dirs or {}).get(kind, cache_dir), kind, row))
            for kind in kinds
        ]))
    dims = {vector.shape for vector in vectors}
    if len(dims) != 1:
        raise ValueError(f"Feature dimensions vary within {manifest}: {dims}")
    return np.stack(vectors).astype(np.float32), np.asarray([r["label"] for r in rows], dtype=np.int64), rows


def train(
    data_dir: Path, cache_dir: Path, output: Path, system: str, *, seed: int = 42,
    epochs: int = 50, patience: int = 7, batch_size: int = 64,
    learning_rate: float = 1e-3, hidden_dim: int = 128, dropout: float = 0.2,
    device: str = "cpu", task: str = "sentiment", feature_dirs: dict[str, Path] | None = None,
) -> dict:
    if system not in SYSTEM_FEATURES:
        raise ValueError(system)
    if task not in TASK_LABELS:
        raise ValueError(task)
    label_names = TASK_LABELS[task]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    x_train, y_train, train_rows = load_split(data_dir / "train.jsonl", cache_dir, system, feature_dirs=feature_dirs)
    x_val, y_val, val_rows = load_split(data_dir / "val.jsonl", cache_dir, system, feature_dirs=feature_dirs)
    if any(row.get("task", "sentiment") != task for row in train_rows + val_rows):
        raise ValueError(f"Manifest task does not match {task}")
    if len(x_train) != len(train_rows) or len(x_val) != len(val_rows):
        raise AssertionError("Feature/label mismatch")
    counts = np.bincount(y_train, minlength=len(label_names))
    if np.any(counts == 0):
        raise ValueError(f"Training split missing a {task} class: {counts.tolist()}")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x = torch.from_numpy((x_train - mean) / scale).to(device)
    val_x = torch.from_numpy((x_val - mean) / scale).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    weights = torch.from_numpy((len(y_train) / (len(label_names) * counts)).astype(np.float32)).to(device)
    model = SentimentHead(x_train.shape[1], hidden_dim, dropout, len(label_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(seed)
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_y), generator=generator)
        losses = []
        for indices in permutation.split(batch_size):
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            predicted = model(val_x).argmax(dim=-1).cpu().numpy()
        macro_f1 = float(f1_score(y_val, predicted, average="macro", labels=list(range(len(label_names))), zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": macro_f1})
        if macro_f1 > best_f1 + 1e-6:
            best_f1, best_epoch, stale = macro_f1, epoch, 0
            torch.save({
                "system": system, "state_dict": model.cpu().state_dict(), "mean": mean.tolist(),
                "scale": scale.tolist(), "hidden_dim": hidden_dim, "dropout": dropout,
                "seed": seed, "best_epoch": epoch, "val_macro_f1": macro_f1,
                "train_class_counts": counts.tolist(), "dataset": "MELD", "task": task,
                "label_names": list(label_names),
                "train_sample_hash": sample_hash(train_rows), "val_sample_hash": sample_hash(val_rows),
            }, output)
            model.to(device)
        else:
            stale += 1
            if stale >= patience:
                break
    result = {"checkpoint": str(output), "system": system, "task": task, "best_epoch": best_epoch, "val_macro_f1": best_f1, "history": history}
    output.with_suffix(".train.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
