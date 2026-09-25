"""Metrics on one official MELD split, plus optional live end-to-end benchmark."""

from __future__ import annotations

import json
import platform
import resource
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import TASK_LABELS
from .inference import Predictor
from .model import SentimentHead
from .train import load_split, sample_hash


def evaluate(checkpoint_path: Path, manifest: Path, cache_dir: Path, output: Path, *, device: str = "cpu") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    system = checkpoint["system"]
    task = checkpoint.get("task", "sentiment")
    label_names = tuple(checkpoint.get("label_names", TASK_LABELS[task]))
    label_ids = list(range(len(label_names)))
    x, y, rows = load_split(manifest, cache_dir, system)
    if any(row.get("task", "sentiment") != task for row in rows):
        raise ValueError(f"Checkpoint task {task} does not match manifest")
    for split, key in (("train", "train_sample_hash"), ("val", "val_sample_hash")):
        if rows[0]["split"] == split and checkpoint.get(key) != sample_hash(rows):
            raise ValueError(f"Checkpoint {split} samples differ from manifest")
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    if x.shape[1] != len(mean):
        raise ValueError("Checkpoint and cache dimensions differ")
    model = SentimentHead(x.shape[1], checkpoint["hidden_dim"], checkpoint["dropout"], len(label_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(torch.from_numpy((x - mean) / scale).to(device)), dim=-1).cpu().numpy()
    predicted = probabilities.argmax(axis=1)
    report = {
        "dataset": "MELD", "split": rows[0]["split"], "system": system, "task": task,
        "label_names": list(label_names),
        "count": len(rows), "macro_f1": float(f1_score(y, predicted, average="macro", labels=label_ids, zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, average="weighted", labels=label_ids, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "per_class": classification_report(y, predicted, labels=label_ids, target_names=label_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predicted, labels=label_ids).tolist(),
        "confusion_matrix_order": list(label_names),
        "head_parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "train_sample_hash": checkpoint.get("train_sample_hash"),
        "val_sample_hash": checkpoint.get("val_sample_hash"),
        "test_sample_hash": sample_hash(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    predictions = output.with_suffix(".predictions.jsonl")
    with predictions.open("w", encoding="utf-8") as handle:
        for row, label, probs in zip(rows, predicted, probabilities):
            handle.write(json.dumps({"id": row["id"], "gold": row["label"], "predicted": int(label), "probabilities": probs.tolist(), "reference_transcript": row.get("transcript")}, ensure_ascii=False) + "\n")
    return report


def benchmark(checkpoint_path: Path, manifest: Path, *, device: str = "cpu", limit: int = 200) -> dict:
    from .meld import read_manifest

    rows = read_manifest(manifest)[:limit]
    if len(rows) < 2:
        raise ValueError("Benchmark needs at least two clips")
    predictor = Predictor(checkpoint_path, device)
    if any(row.get("task", "sentiment") != predictor.task for row in rows):
        raise ValueError(f"Checkpoint task {predictor.task} does not match benchmark manifest")
    # Warm up the same loaded pipeline before timing; this run is excluded.
    predictor.predict(Path(rows[0]["audio_path"]))
    timing_rows = []
    for row in rows:
        result = predictor.predict(Path(row["audio_path"]))
        timing_rows.append({"id": row["id"], "audio_seconds": row["duration_seconds"], **result["timings"]})
    durations = np.asarray([item["audio_seconds"] for item in timing_rows])
    end_to_end = np.asarray([item["end_to_end_seconds"] for item in timing_rows])
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak_rss *= 1024
    known_parameters = 0
    known_model_state_bytes = 0
    for encoder in (predictor.audio, predictor.text, predictor.mimi):
        if encoder is not None:
            module = getattr(encoder.model, "model", encoder.model)
            if isinstance(module, torch.nn.Module):
                known_parameters += sum(p.numel() for p in module.parameters())
                known_model_state_bytes += sum(t.numel() * t.element_size() for t in module.state_dict().values())
    known_parameters += sum(p.numel() for p in predictor.head.parameters())
    known_model_state_bytes += sum(t.numel() * t.element_size() for t in predictor.head.state_dict().values())
    return {
        "dataset": "MELD", "system": predictor.system, "task": predictor.task, "device": device,
        "batch_size": 1, "warmup_clips": 1, "measured_clips": len(timing_rows),
        "mean_end_to_end_seconds": float(end_to_end.mean()),
        "median_end_to_end_seconds": float(np.median(end_to_end)),
        "rtf_total": float(end_to_end.sum() / durations.sum()),
        "peak_process_rss_bytes": int(peak_rss),
        "known_model_parameters": known_parameters,
        "known_model_state_bytes": known_model_state_bytes,
        "asr_parameter_count_included": False,
        "asr_parameter_count_note": "ASR parameters are not included in known_model_parameters.",
        "stage_means_seconds": {k: float(np.mean([item[k] for item in timing_rows if k in item])) for k in timing_rows[0] if k.endswith("_seconds") and k != "audio_seconds"},
        "per_utterance": timing_rows,
    }
