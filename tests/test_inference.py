import json

import pytest
import torch

from speech_sentiment.evaluate import benchmark
from speech_sentiment.inference import Predictor
from speech_sentiment.model import SentimentHead


def test_classifier_only_predictor_does_not_load_encoders(tmp_path):
    head = SentimentHead(2, hidden_dim=4, dropout=0.0, num_classes=3)
    checkpoint = tmp_path / "audio.pt"
    torch.save({
        "system": "audio",
        "task": "sentiment",
        "label_names": ["negative", "neutral", "positive"],
        "mean": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "hidden_dim": 4,
        "dropout": 0.0,
        "state_dict": head.state_dict(),
    }, checkpoint)

    predictor = Predictor(checkpoint, load_encoders=False)

    assert predictor.audio is None
    assert predictor.asr is None
    assert predictor.text is None
    assert predictor.mimi is None
    with pytest.raises(RuntimeError, match="only a classifier head"):
        predictor.predict(tmp_path / "unused.wav")


def test_benchmark_marks_asr_parameters_as_not_included(tmp_path, monkeypatch):
    class FakePredictor:
        system = "fusion"
        task = "sentiment"
        device = "cpu"
        asr = object()
        audio = text = mimi = None
        head = torch.nn.Linear(2, 3)

        def __init__(self, checkpoint_path, device):
            pass

        def predict(self, audio_path):
            return {"timings": {"end_to_end_seconds": 0.1}}

    manifest = tmp_path / "test.jsonl"
    rows = [
        {"id": f"test_{index}", "audio_path": str(tmp_path / f"{index}.wav"),
         "label": 0, "split": "test", "duration_seconds": 1.0}
        for index in range(2)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr("speech_sentiment.evaluate.Predictor", FakePredictor)

    result = benchmark(tmp_path / "checkpoint.pt", manifest)

    assert result["asr_parameter_count_included"] is False
    assert "not included" in result["asr_parameter_count_note"]
