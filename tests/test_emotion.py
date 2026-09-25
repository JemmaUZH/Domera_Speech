import json

import numpy as np
import torch

from speech_sentiment import EMOTION_LABELS
from speech_sentiment.evaluate import evaluate
from speech_sentiment.features import feature_path, write_vector
from speech_sentiment.train import train


def test_seven_class_head_uses_separate_labels_and_cache(tmp_path):
    data = tmp_path / "emotion"
    data.mkdir()
    cache = tmp_path / "cache"
    for split in ("train", "val", "test"):
        rows = []
        for label in range(7):
            for repeat in range(3):
                row = {"id": f"{split}_{label}_{repeat}", "split": split, "task": "emotion", "audio_path": "unused.wav", "label": label, "duration_seconds": 1.0}
                rows.append(row)
                vector = np.eye(7, dtype=np.float32)[label]
                write_vector(feature_path(cache, "audio", row), vector)
        (data / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = tmp_path / "emotion" / "audio.pt"
    train(data, cache, checkpoint, "audio", task="emotion", epochs=8, patience=4)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["label_names"] == list(EMOTION_LABELS)
    report = evaluate(checkpoint, data / "test.jsonl", cache, tmp_path / "emotion" / "audio_test.json")
    assert report["task"] == "emotion"
    assert report["count"] == 21
    assert report["confusion_matrix_order"] == list(EMOTION_LABELS)
    assert len(report["confusion_matrix"]) == 7
    assert len(json.loads((tmp_path / "emotion" / "audio_test.predictions.jsonl").read_text().splitlines()[0])["probabilities"]) == 7
    assert 0 <= report["weighted_f1"] <= 1
