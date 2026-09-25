import json

import numpy as np

from speech_sentiment.evaluate import evaluate
from speech_sentiment.features import asr_path, extract, feature_path, load_vector, write_vector
from speech_sentiment.train import train


def _manifest(path, split, count=12):
    rows = []
    for i in range(count):
        audio = path.parent / f"{split}_{i}.wav"
        audio.touch()
        rows.append({"id": f"{split}_{i}", "split": split, "audio_path": str(audio), "transcript": "gold reference must not be used", "label": i % 3, "duration_seconds": 1.0})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_asr_text_is_used_instead_of_meld_reference(tmp_path, monkeypatch):
    manifest = tmp_path / "train.jsonl"
    rows = _manifest(manifest, "train", 1)
    observed = []
    transcriptions = []

    class Audio:
        model_id = "fake-audio"
        def __init__(self, device): pass
        def encode(self, path): return np.ones(2, dtype=np.float32)

    class ASR:
        model_id = "fake-asr"
        def __init__(self, device, compute_type): pass
        def transcribe(self, path):
            transcriptions.append(path)
            return "words from audio"

    class Text:
        model_id = "fake-text"
        def __init__(self, device): pass
        def encode(self, value):
            observed.append(value)
            return np.ones(3, dtype=np.float32)

    class Mimi:
        model_id = "fake-mimi"

    for name, fake in (("Emotion2Vec", Audio), ("LocalASR", ASR), ("TextEncoder", Text), ("ContinuousMimi", Mimi)):
        monkeypatch.setattr(f"speech_sentiment.features.{name}", fake)
    extract([manifest], tmp_path / "cache", "fusion")
    assert observed == ["words from audio"]
    assert json.loads((tmp_path / "cache/asr/train/train_0.json").read_text())["text"] == "words from audio"
    assert load_vector(feature_path(tmp_path / "cache", "text", rows[0])).shape == (3,)
    extract([manifest], tmp_path / "cache", "fusion")
    assert len(transcriptions) == 1  # rerun reuses fusion's ASR and MiniLM cache
    assert observed == ["words from audio"]


def test_existing_text_cache_is_reused_by_fusion(tmp_path, monkeypatch):
    manifest = tmp_path / "train.jsonl"
    row = _manifest(manifest, "train", 1)[0]
    cache = tmp_path / "cache"
    transcript_path = asr_path(cache, row)
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(json.dumps({"text": "cached ASR"}))
    write_vector(feature_path(cache, "text", row), np.ones(3, dtype=np.float32))

    class Audio:
        model_id = "fake-audio"
        def __init__(self, device): pass
        def encode(self, path): return np.ones(2, dtype=np.float32)

    class MustNotRun:
        model_id = "unused"
        def __init__(self, *args): raise AssertionError("cached text branch was recomputed")

    class Mimi:
        model_id = "fake-mimi"

    for name, fake in (("Emotion2Vec", Audio), ("LocalASR", MustNotRun), ("TextEncoder", MustNotRun), ("ContinuousMimi", Mimi)):
        monkeypatch.setattr(f"speech_sentiment.features.{name}", fake)
    extract([manifest], cache, "fusion")
    assert load_vector(feature_path(cache, "audio", row)).shape == (2,)


def test_audio_train_and_evaluate_on_cached_features(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    cache = tmp_path / "cache"
    for split in ("train", "val", "test"):
        rows = _manifest(data / f"{split}.jsonl", split, 18)
        for row in rows:
            label = row["label"]
            write_vector(feature_path(cache, "audio", row), np.asarray([label * 3.0, 1.0], dtype=np.float32))
    checkpoint = tmp_path / "audio.pt"
    outcome = train(data, cache, checkpoint, "audio", epochs=80, patience=25, batch_size=6)
    assert checkpoint.exists()
    assert outcome["best_epoch"] >= 1
    report = evaluate(checkpoint, data / "test.jsonl", cache, tmp_path / "test.json")
    assert report["count"] == 18
    assert len(report["confusion_matrix"]) == 3
    assert report["macro_f1"] > 0.5


def test_text_only_train_and_evaluate_on_cached_features(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    cache = tmp_path / "cache"
    for split in ("train", "val", "test"):
        rows = _manifest(data / f"{split}.jsonl", split, 18)
        for row in rows:
            label = row["label"]
            vector = np.asarray([label * 3.0, float(label == 1), float(label == 2)], dtype=np.float32)
            write_vector(feature_path(cache, "text", row), vector)
    checkpoint = tmp_path / "text.pt"
    outcome = train(data, cache, checkpoint, "text", epochs=30, patience=8, batch_size=6)
    assert checkpoint.exists()
    assert outcome["best_epoch"] >= 1
    report = evaluate(checkpoint, data / "test.jsonl", cache, tmp_path / "text_test.json")
    assert report["system"] == "text"
    assert report["label_names"] == ["negative", "neutral", "positive"]
    assert report["count"] == 18
    assert len(report["confusion_matrix"]) == 3
