"""Prepare CREMA-D speaker-disjoint 1,000/200 train/validation sets.

The existing 200-example test manifest is reused read-only. New data and
manifests are isolated in data/crema_d_supervised/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_crema_d import download_one, read_summary  # noqa: E402

EMOTIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness")
OUT = ROOT / "data/crema_d_supervised"
EXISTING_TEST = ROOT / "data/crema_d/test.jsonl"
SEED = 42


def stratified_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"Requested {count} rows from only {len(rows)} examples")
    groups = {label: [r for r in rows if r["label"] == label] for label in EMOTIONS}
    exact = {label: count * len(group) / len(rows) for label, group in groups.items()}
    quotas = {label: math.floor(value) for label, value in exact.items()}
    remainder = count - sum(quotas.values())
    for label in sorted(EMOTIONS, key=lambda k: (-(exact[k] - quotas[k]), EMOTIONS.index(k)))[:remainder]:
        quotas[label] += 1
    rng = random.Random(seed)
    selected = []
    for label in EMOTIONS:
        ordered = sorted(groups[label], key=lambda r: r["id"])
        selected.extend(rng.sample(ordered, quotas[label]))
    return sorted(selected, key=lambda r: r["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "data/crema_d/summaryTable.csv")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not args.summary_csv.is_file():
        raise FileNotFoundError(f"CREMA-D metadata CSV not found: {args.summary_csv}")
    metadata, summary_hash = read_summary(args.summary_csv)
    subset_cfg = json.loads((ROOT / "data/crema_d/subset_config.json").read_text(encoding="utf-8"))
    test_rows = [json.loads(line) for line in EXISTING_TEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(test_rows) != 200:
        raise ValueError("Expected the existing fixed 200-example CREMA-D test set")
    if summary_hash != subset_cfg["summary_table_sha256"]:
        raise ValueError("CREMA-D metadata differs from that used to create the fixed test set")

    speakers = sorted({row["speaker_id"] for row in metadata})
    rng = np.random.RandomState(0)
    rng.shuffle(speakers)
    train_end, val_end = int(0.7 * len(speakers)), int(0.8 * len(speakers))
    train_speakers, val_speakers, test_speakers = set(speakers[:train_end]), set(speakers[train_end:val_end]), set(speakers[val_end:])
    fixed_test_speakers = {row["speaker_id"] for row in test_rows}
    if fixed_test_speakers != test_speakers:
        raise ValueError("Existing test speakers do not match the official TFDS 1.0.0 split")
    assert not (train_speakers & val_speakers or train_speakers & test_speakers or val_speakers & test_speakers)

    train_pool = [row for row in metadata if row["speaker_id"] in train_speakers]
    val_pool = [row for row in metadata if row["speaker_id"] in val_speakers]
    selected = {"train": stratified_sample(train_pool, 1000, SEED), "val": stratified_sample(val_pool, 200, SEED + 1)}
    for split in selected:
        selected_ids = {row["id"] for row in selected[split]}
        if split == "train" and selected_ids & {row["id"] for row in test_rows}:
            raise ValueError("Test ID leakage into train")
        if split == "val" and selected_ids & {row["id"] for row in test_rows}:
            raise ValueError("Test ID leakage into validation")

    OUT.mkdir(parents=True, exist_ok=True)
    output_rows = {}
    for split, items in selected.items():
        audio_dir = OUT / "audio" / split
        completed = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_one, row, audio_dir): row["id"] for row in items}
            for future in as_completed(futures):
                item = future.result()
                completed[item["id"]] = item
                if len(completed) % 50 == 0 or len(completed) == len(items):
                    print(f"{split}: downloaded {len(completed)}/{len(items)}", flush=True)
        records = []
        for item in items:
            downloaded = completed[item["id"]]
            audio_path = Path(downloaded["audio_path"])
            records.append({
                **downloaded, "label": EMOTIONS.index(downloaded["label"]),
                "emotion_label": downloaded["label"], "split": split, "task": "emotion",
                "dataset": "CREMA-D", "duration_seconds": float(sf.info(audio_path).duration),
            })
        output_rows[split] = records
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Numeric IDs are required by the common frozen-feature loader. This is a
    # separate manifest view of the unchanged, existing test examples.
    numeric_test = []
    for row in test_rows:
        numeric_test.append({**row, "emotion_label": row["label"], "label": EMOTIONS.index(row["label"])})
    with (OUT / "test.jsonl").open("w", encoding="utf-8") as handle:
        for row in numeric_test:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def ids_hash(rows: list[dict]) -> str:
        return hashlib.sha256(json.dumps([(r["id"], r["label"]) for r in rows], separators=(",", ":")).encode()).hexdigest()

    config = {
        "dataset": "CREMA-D", "source": "TensorFlow Datasets crema_d 1.0.0 speaker-exclusive split",
        "source_summary_sha256": summary_hash, "selection_seed": SEED,
        "speaker_split_random_state": 0, "speakers": {"train": sorted(train_speakers), "val": sorted(val_speakers), "test": sorted(test_speakers)},
        "sample_counts": {"train": 1000, "val": 200, "test": 200},
        "class_counts": {split: dict(Counter(r["emotion_label"] for r in records)) for split, records in {**output_rows, "test": numeric_test}.items()},
        "label_names": list(EMOTIONS), "label_to_id": {label: i for i, label in enumerate(EMOTIONS)},
        "existing_test_manifest": str(EXISTING_TEST.relative_to(ROOT)),
        "existing_test_manifest_sha256": hashlib.sha256(EXISTING_TEST.read_bytes()).hexdigest(),
        "manifest_sample_hashes": {split: ids_hash(records) for split, records in {**output_rows, "test": numeric_test}.items()},
        "split_speaker_disjoint": True, "test_samples_reused_unchanged": True,
        "selection": "stratified sampling without replacement within official TFDS speaker split",
    }
    (OUT / "subset_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": str(OUT), "sample_counts": config["sample_counts"], "class_counts": config["class_counts"], "speaker_disjoint": True}, indent=2))


if __name__ == "__main__":
    main()
