"""Train six-class CREMA-D classifier heads and test on the fixed 200 clips.

Pretrained encoders stay frozen. This script writes separate artifacts under
cache/crema_d_supervised/ and runs/crema_d_supervised/; the prior MELD and
CREMA-D cross-dataset outputs are untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speech_sentiment import EMOTION_LABELS  # noqa: E402
from speech_sentiment.features import SYSTEM_FEATURES, extract, feature_path, load_vector  # noqa: E402
from speech_sentiment.model import SentimentHead  # noqa: E402
from speech_sentiment.train import sample_hash  # noqa: E402

DATA = ROOT / "data/crema_d_supervised"
CACHE = ROOT / "cache/crema_d_supervised"
OUT = ROOT / "runs/crema_d_supervised"
LABELS = tuple(EMOTION_LABELS[:6])
SYSTEMS = ("audio", "fusion", "mimi")
SEED, EPOCHS, PATIENCE, BATCH_SIZE = 42, 50, 7, 64
HIDDEN_DIM, DROPOUT, LEARNING_RATE = 128, 0.2, 1e-3


def read_manifest(split: str) -> list[dict]:
    path = DATA / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {"train": 1000, "val": 200, "test": 200}[split]
    if len(rows) != expected or any(r["split"] != split or r.get("dataset") != "CREMA-D" for r in rows):
        raise ValueError(f"Invalid {split} manifest: expected {expected} CREMA-D rows")
    if any(not isinstance(r["label"], int) or r["label"] not in range(6) for r in rows):
        raise ValueError(f"{split} must use numeric labels 0..5 in {LABELS}")
    for row in rows:
        path = Path(row["audio_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != row["audio_sha256"]:
            raise ValueError(f"Audio changed for {row['id']}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_features(rows: list[dict], system: str) -> np.ndarray:
    vectors = [np.concatenate([load_vector(feature_path(CACHE, kind, row)) for kind in SYSTEM_FEATURES[system]]) for row in rows]
    shapes = {x.shape for x in vectors}
    if len(shapes) != 1:
        raise ValueError(f"{system}: inconsistent feature shapes {shapes}")
    return np.stack(vectors).astype(np.float32)


def train_head(system: str, train_rows: list[dict], val_rows: list[dict], device: str) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
    x_train, x_val = load_features(train_rows, system), load_features(val_rows, system)
    y_train = np.asarray([r["label"] for r in train_rows], dtype=np.int64)
    y_val = np.asarray([r["label"] for r in val_rows], dtype=np.int64)
    counts = np.bincount(y_train, minlength=len(LABELS))
    if (counts == 0).any():
        raise ValueError(f"Training split is missing a class: {counts.tolist()}")
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    tx = torch.from_numpy((x_train - mean) / scale).to(device)
    vx = torch.from_numpy((x_val - mean) / scale).to(device)
    ty = torch.from_numpy(y_train).to(device)
    class_weights = torch.from_numpy((len(ty) / (len(LABELS) * counts)).astype(np.float32)).to(device)
    model = SentimentHead(tx.shape[1], HIDDEN_DIM, DROPOUT, len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(SEED)
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    checkpoint_path = OUT / "checkpoints" / f"{system}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(ty), generator=generator)
        losses = []
        for indices in order.split(BATCH_SIZE):
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tx[indices]), ty[indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_pred = model(vx).argmax(dim=-1).cpu().numpy()
        val_macro = float(f1_score(y_val, val_pred, labels=list(range(6)), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_macro})
        if val_macro > best_f1 + 1e-6:
            best_f1, best_epoch, stale = val_macro, epoch, 0
            torch.save({
                "system": system, "task": "emotion", "dataset": "CREMA-D", "label_names": list(LABELS),
                "state_dict": model.cpu().state_dict(), "mean": mean.tolist(), "scale": scale.tolist(),
                "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT, "seed": SEED,
                "best_epoch": epoch, "val_macro_f1": val_macro, "train_class_counts": counts.tolist(),
                "train_sample_hash": sample_hash(train_rows), "val_sample_hash": sample_hash(val_rows),
                "training_data": "CREMA-D 1000-example speaker-exclusive train subset; frozen encoder features",
            }, checkpoint_path)
            model.to(device)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    result = {
        "system": system, "dataset": "CREMA-D", "train_count": len(train_rows), "val_count": len(val_rows),
        "best_epoch": best_epoch, "val_macro_f1": best_f1, "train_class_counts": counts.tolist(),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path), "history": history,
    }
    atomic_json(OUT / system / "train_metrics.json", result)
    print(f"{system}: selected epoch {best_epoch}; validation Macro-F1 {best_f1:.3f}", flush=True)
    return result


def apply_checkpoint(checkpoint_path: Path, x: np.ndarray, device: str) -> tuple[np.ndarray, float, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    model = SentimentHead(len(mean), checkpoint["hidden_dim"], checkpoint["dropout"], len(checkpoint["label_names"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    start = time.perf_counter()
    with torch.inference_mode():
        probs = torch.softmax(model(torch.from_numpy((x - mean) / scale).to(device)), dim=-1).cpu().numpy()
    batch_seconds = time.perf_counter() - start
    return probs, batch_seconds, batch_seconds / len(x)


def evaluate_head(system: str, rows: list[dict], device: str, train_hash: str, val_hash: str) -> dict:
    x = load_features(rows, system)
    checkpoint_path = OUT / "checkpoints" / f"{system}.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(ckpt["label_names"]) != LABELS or ckpt["train_sample_hash"] != train_hash or ckpt["val_sample_hash"] != val_hash:
        raise ValueError(f"{system} checkpoint data/label provenance does not match the CREMA-D manifests")
    probabilities, batch_latency_seconds, batch_throughput_seconds_per_utterance = apply_checkpoint(checkpoint_path, x, device)
    y_true = np.asarray([r["label"] for r in rows], dtype=np.int64)
    y_pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(6)), zero_division=0)
    per_class = [{"label": label, "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(LABELS)]
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(6))).tolist()
    distribution = Counter(LABELS[index] for index in y_pred)
    result = {
        "dataset": "CREMA-D", "split": "fixed 200-example test", "system": system,
        "evaluation_type": "CREMA-D trained classifier head; frozen pretrained encoder; test speakers held out",
        "count": len(rows), "label_names": list(LABELS), "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(6)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(6)), average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)), "per_class": per_class,
        "confusion_matrix": matrix, "confusion_matrix_labels": list(LABELS),
        "prediction_distribution": [{"label": label, "count": distribution[label], "percentage": distribution[label] / len(rows) * 100} for label in LABELS],
        "test_sample_hash": sample_hash(rows), "test_manifest_sha256": hashlib.sha256((DATA / "test.jsonl").read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "classifier_batch_latency_seconds": batch_latency_seconds,
        "classifier_batch_throughput_seconds_per_utterance": batch_throughput_seconds_per_utterance,
        "latency_note": "Classifier-only wall time for one batch of 200 precomputed feature vectors; throughput per utterance is batch time divided by 200, not single-utterance latency. Audio encoder/ASR runtime is excluded.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    directory = OUT / system
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "metrics.json", result)
    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "gold", "prediction", "probabilities_json", "audio_sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, probs, pred in zip(rows, probabilities, y_pred):
            writer.writerow({"sample_id": row["id"], "gold": LABELS[row["label"]], "prediction": LABELS[int(pred)], "probabilities_json": json.dumps(dict(zip(LABELS, probs.tolist()))), "audio_sha256": row["audio_sha256"]})
    with (directory / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(per_class)
    save_confusion(directory / "confusion_matrix.png", matrix)
    return result


def save_confusion(path: Path, matrix: list[list[int]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(np.asarray(matrix), cmap="Blues")
    fig.colorbar(image, ax=ax, label="Count")
    ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(6), LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title("CREMA-D held-out test — CREMA-D trained head")
    for i, line in enumerate(matrix):
        for j, value in enumerate(line):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=[*SYSTEMS, "all"], default="all")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train_rows, val_rows, test_rows = read_manifest("train"), read_manifest("val"), read_manifest("test")
    train_hash, val_hash = sample_hash(train_rows), sample_hash(val_rows)
    config = json.loads((DATA / "subset_config.json").read_text(encoding="utf-8"))
    speaker_sets = {split: set(config["speakers"][split]) for split in ("train", "val", "test")}
    if any(speaker_sets[a] & speaker_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Speaker leakage across CREMA-D splits")
    for rows, split in ((train_rows, "train"), (val_rows, "val"), (test_rows, "test")):
        if any(r["speaker_id"] not in speaker_sets[split] for r in rows):
            raise ValueError(f"Speaker leakage detected in {split}")
    print("Verified 1,000/200/200 samples; disjoint speakers and fixed test manifest.", flush=True)
    manifests = [DATA / f"{split}.jsonl" for split in ("train", "val", "test")]
    systems = SYSTEMS if args.system == "all" else (args.system,)
    OUT.mkdir(parents=True, exist_ok=True)
    for system in systems:
        print(f"\nExtracting/caching frozen {system} features...", flush=True)
        extraction = extract(manifests, CACHE, system, args.device)
        atomic_json(OUT / system / "feature_extraction.json", extraction)
        train_head(system, train_rows, val_rows, args.device)
        result = evaluate_head(system, test_rows, args.device, train_hash, val_hash)
        print(f"{system}: TEST Macro-F1 {result['macro_f1']:.3f}; Weighted-F1 {result['weighted_f1']:.3f}; Accuracy {result['accuracy']:.1%}", flush=True)

    summaries = {}
    for system in systems:
        current = json.loads((OUT / system / "metrics.json").read_text(encoding="utf-8"))
        previous_path = ROOT / "runs/crema_d" / system / "metrics.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else None
        summaries[system] = {
            "previous_meld_trained": (
                {k: previous[k] for k in ("macro_f1", "weighted_f1", "accuracy")}
                if previous is not None else None
            ),
            "crema_d_trained": {k: current[k] for k in ("macro_f1", "weighted_f1", "accuracy")},
        }
    atomic_json(OUT / "comparison.json", {"dataset": "CREMA-D", "test_manifest": str((DATA / "test.jsonl").relative_to(ROOT)), "systems": summaries})
    has_previous = any(value["previous_meld_trained"] is not None for value in summaries.values())
    lines = ["# CREMA-D-trained heads on the fixed test set", "", "The encoders remained frozen. Only the classifier heads were trained on 1,000 CREMA-D train utterances; validation Macro-F1 on 200 speaker-disjoint validation utterances selected the checkpoint. The fixed 200 test utterances use the official speaker-disjoint test split.", "", "Classifier-only test latency uses cached frozen features and excludes encoder/ASR time.", ""]
    if has_previous:
        lines += ["| System | MELD-trained Macro-F1 | CREMA-D-trained Macro-F1 | MELD-trained Accuracy | CREMA-D-trained Accuracy |", "|---|---:|---:|---:|---:|"]
    else:
        lines += ["| System | CREMA-D-trained Macro-F1 | Weighted-F1 | Accuracy |", "|---|---:|---:|---:|"]
    for system, data in summaries.items():
        old, new = data["previous_meld_trained"], data["crema_d_trained"]
        if old is not None:
            lines.append(f"| {system} | {old['macro_f1']:.3f} | {new['macro_f1']:.3f} | {old['accuracy']:.1%} | {new['accuracy']:.1%} |")
        else:
            lines.append(f"| {system} | {new['macro_f1']:.3f} | {new['weighted_f1']:.3f} | {new['accuracy']:.1%} |")
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    packages = {}
    for name in ("torch", "numpy", "scikit-learn", "matplotlib", "soundfile"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    run_config = {
        "dataset": "CREMA-D", "systems": list(systems), "labels": list(LABELS),
        "train_count": 1000, "validation_count": 200, "test_count": 200,
        "encoders": "frozen pretrained encoders; train only the simple MLP classifier head",
        "train_split": "official speaker-disjoint TFDS 1.0.0 train; stratified 1000 subset",
        "validation_split": "official speaker-disjoint TFDS 1.0.0 validation; stratified 200 subset",
        "test_manifest": str((DATA / "test.jsonl").relative_to(ROOT)),
        "test_manifest_sha256": hashlib.sha256((DATA / "test.jsonl").read_bytes()).hexdigest(),
        "seed": SEED, "epochs": EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
        "checkpoint_selection": "validation Macro-F1; test labels not used for model selection",
        "device": args.device, "python_version": platform.python_version(), "platform": platform.platform(),
        "package_versions": packages, "feature_cache": str(CACHE.relative_to(ROOT)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "old_cross_dataset_outputs_preserved": True,
    }
    atomic_json(OUT / "run_config.json", run_config)
    print(f"\nSaved CREMA-D-trained runs under {OUT}", flush=True)


if __name__ == "__main__":
    main()
