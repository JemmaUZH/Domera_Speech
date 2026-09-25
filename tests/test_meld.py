import csv
import json
import wave
from pathlib import Path

import pytest

from speech_sentiment.meld import convert_clip, prepare


def _csv(path: Path, sentiment: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Dialogue_ID", "Utterance_ID", "Utterance", "Speaker", "Sentiment"])
        writer.writeheader()
        writer.writerow({"Dialogue_ID": 0, "Utterance_ID": 1, "Utterance": "reference words", "Speaker": "Rachel", "Sentiment": sentiment})


def test_official_split_and_label_mapping(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for csv_name, clip_dir, sentiment in (
        ("train_sent_emo.csv", "train_splits", "negative"),
        ("dev_sent_emo.csv", "dev_splits_complete", "neutral"),
        ("test_sent_emo.csv", "output_repeated_splits_test", "positive"),
    ):
        _csv(source / csv_name, sentiment)
        directory = source / clip_dir
        directory.mkdir(exist_ok=True)
        (directory / "dia0_utt1.mp4").touch()

    def fake_convert(clip, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return 1.0

    monkeypatch.setattr("speech_sentiment.meld.convert_clip", fake_convert)
    target = tmp_path / "processed"
    summary = prepare(source, target)
    assert [summary[split]["count"] for split in ("train", "val", "test")] == [1, 1, 1]
    for split, label in (("train", 0), ("val", 1), ("test", 2)):
        row = json.loads((target / f"{split}.jsonl").read_text())
        assert row["label"] == label
        assert row["id"] == f"{split}_dia0_utt1"
        assert row["transcript"] == "reference words"


def test_missing_clip_is_reported_and_strict_fails(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for csv_name, clip_dir in (("train_sent_emo.csv", "train_splits"), ("dev_sent_emo.csv", "dev_splits_complete"), ("test_sent_emo.csv", "output_repeated_splits_test")):
        _csv(source / csv_name, "neutral")
        directory = source / clip_dir
        directory.mkdir(exist_ok=True)
        (directory / "dia0_utt1.mp4").touch()
    # Remove one of three clips; the split now has no valid rows.
    (source / "dev_splits_complete" / "dia0_utt1.mp4").unlink()
    monkeypatch.setattr("speech_sentiment.meld.convert_clip", lambda clip, output: 1.0)
    with pytest.raises(RuntimeError, match="val/dia0_utt1"):
        prepare(source, tmp_path / "output", strict=True)


def test_converted_wav_is_rebuilt_when_source_clip_changes(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    output = tmp_path / "audio" / "clip.wav"
    source.write_bytes(b"first source")
    conversions = []

    def fake_ffmpeg(command, check):
        conversions.append(source.read_bytes())
        with wave.open(command[-1], "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 160)

    monkeypatch.setattr("speech_sentiment.meld.subprocess.run", fake_ffmpeg)
    assert convert_clip(source, output) == 0.01
    assert convert_clip(source, output) == 0.01
    source.write_bytes(b"updated source")
    assert convert_clip(source, output) == 0.01
    assert conversions == [b"first source", b"updated source"]
