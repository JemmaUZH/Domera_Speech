import csv
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

from speech_sentiment.subset import prepare_subset, stratified_candidates


def test_stratified_quota_is_deterministic():
    rows = [{"id": str(i), "label": i % 3} for i in range(90)]
    quotas, first = stratified_candidates(rows, 30, 42)
    _, second = stratified_candidates(rows, 30, 42)
    assert quotas == {0: 10, 1: 10, 2: 10}
    assert [[row["id"] for row in first[label]] for label in range(3)] == [[row["id"] for row in second[label]] for label in range(3)]


def test_archive_streaming_prepares_exact_shared_subset(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    source = tmp_path / "MELD.Raw.tar.gz"
    names = {"train": ("train_sent_emo.csv", "train.tar.gz"), "val": ("dev_sent_emo.csv", "dev.tar.gz"), "test": ("test_sent_emo.csv", "test.tar.gz")}
    with tarfile.open(source, "w:gz") as outer:
        for split, (csv_name, archive_name) in names.items():
            with (annotations / csv_name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Dialogue_ID", "Utterance_ID", "Utterance", "Speaker", "Sentiment"])
                writer.writeheader()
                for index in range(30):
                    writer.writerow({"Dialogue_ID": index, "Utterance_ID": 0, "Utterance": f"reference {index}", "Speaker": "Ross", "Sentiment": ("negative", "neutral", "positive")[index % 3]})
            nested = io.BytesIO()
            with tarfile.open(fileobj=nested, mode="w:gz") as inner:
                for index in range(30):
                    payload = b"fake mp4"
                    info = tarfile.TarInfo(f"{split}/dia{index}_utt0.mp4")
                    info.size = len(payload)
                    inner.addfile(info, io.BytesIO(payload))
            payload = nested.getvalue()
            info = tarfile.TarInfo(f"MELD.Raw/{archive_name}")
            info.size = len(payload)
            outer.addfile(info, io.BytesIO(payload))

    def fake_convert(clip, output):
        assert clip.read_bytes() == b"fake mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake wav")
        return 1.0

    monkeypatch.setattr("speech_sentiment.subset.convert_clip", fake_convert)
    output = tmp_path / "processed"
    summary = prepare_subset(source, annotations, output, {"train": 12, "val": 6, "test": 6}, seed=42, reserve_per_class=2)
    assert [summary[split]["count"] for split in names] == [12, 6, 6]
    for split, expected in (("train", 4), ("val", 2), ("test", 2)):
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert Counter(row["label"] for row in rows) == {0: expected, 1: expected, 2: expected}
        assert all(Path(row["audio_path"]).is_file() for row in rows)
    assert len(list((output / "audio").rglob("*.wav"))) == 24
