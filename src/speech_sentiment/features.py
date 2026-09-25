"""Per-utterance feature cache. ASR output, never MELD reference text, feeds MiniLM."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .encoders import ContinuousMimi, Emotion2Vec, LocalASR, TextEncoder
from .meld import read_manifest

SYSTEM_FEATURES = {"audio": ("audio",), "text": ("text",), "fusion": ("audio", "text"), "mimi": ("mimi",)}


def feature_path(cache_dir: Path, kind: str, row: dict) -> Path:
    return cache_dir / kind / row["split"] / f"{row['id']}.npy"


def asr_path(cache_dir: Path, row: dict) -> Path:
    return cache_dir / "asr" / row["split"] / f"{row['id']}.json"


def write_vector(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    try:
        np.save(tmp, value.astype(np.float32))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_vector(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"Invalid feature: {path}")
    return value


def extract(manifests: list[Path], cache_dir: Path, system: str, device: str = "cpu", force: bool = False) -> dict:
    if system not in SYSTEM_FEATURES:
        raise ValueError(system)
    rows = [row for path in manifests for row in read_manifest(path)]
    if any(row["split"] not in ("train", "val", "test") for row in rows):
        raise ValueError("Only official MELD splits are accepted")
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / "encoder_config.json"
    config = {"dataset": "MELD", "emotion2vec": Emotion2Vec.model_id, "asr": LocalASR.model_id, "text": TextEncoder.model_id, "mimi": ContinuousMimi.model_id}
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cache model IDs differ; choose a new --cache-dir")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    needed = SYSTEM_FEATURES[system]
    audio_encoder = asr = text_encoder = mimi_encoder = None
    timings = {"audio": 0.0, "asr": 0.0, "text": 0.0, "mimi": 0.0}
    done = 0
    for row in rows:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if "audio" in needed and (force or not feature_path(cache_dir, "audio", row).exists()):
            audio_encoder = audio_encoder or Emotion2Vec(device)
            start = time.perf_counter()
            value = audio_encoder.encode(audio_path)
            timings["audio"] += time.perf_counter() - start
            write_vector(feature_path(cache_dir, "audio", row), value)
        if "text" in needed:
            transcript_file = asr_path(cache_dir, row)
            if force or not transcript_file.exists():
                asr = asr or LocalASR(device, "int8" if device == "cpu" else "float16")
                start = time.perf_counter()
                transcript = asr.transcribe(audio_path)
                timings["asr"] += time.perf_counter() - start
                transcript_file.parent.mkdir(parents=True, exist_ok=True)
                transcript_file.write_text(json.dumps({"text": transcript}, ensure_ascii=False))
            if force or not feature_path(cache_dir, "text", row).exists():
                transcript = json.loads(transcript_file.read_text())["text"]
                text_encoder = text_encoder or TextEncoder(device)
                start = time.perf_counter()
                value = text_encoder.encode(transcript)
                timings["text"] += time.perf_counter() - start
                write_vector(feature_path(cache_dir, "text", row), value)
        if "mimi" in needed and (force or not feature_path(cache_dir, "mimi", row).exists()):
            mimi_encoder = mimi_encoder or ContinuousMimi(device)
            start = time.perf_counter()
            value = mimi_encoder.encode(audio_path)
            timings["mimi"] += time.perf_counter() - start
            write_vector(feature_path(cache_dir, "mimi", row), value)
        done += 1
    result = {"system": system, "utterances": done, "feature_seconds": timings}
    (cache_dir / f"extract_{system}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
