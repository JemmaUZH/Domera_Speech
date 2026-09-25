"""Apply official MELD emotion labels to the already selected utterances."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from . import EMOTION_LABELS, LABELS
from .meld import CSV_NAMES, find_unique, read_manifest


def prepare_emotion_manifests(selected_dir: Path, annotations_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_to_id = {name: index for index, name in enumerate(EMOTION_LABELS)}
    summary = {"dataset": "MELD", "task": "emotion", "labels": list(EMOTION_LABELS)}
    for split, csv_name in CSV_NAMES.items():
        selected = read_manifest(selected_dir / f"{split}.jsonl")
        if any(row.get("task", "sentiment") != "sentiment" or row["split"] != split for row in selected):
            raise ValueError(f"Expected selected sentiment rows for {split}")
        annotations = {}
        with find_unique(annotations_dir, csv_name).open(newline="", encoding="utf-8-sig") as handle:
            for source in csv.DictReader(handle):
                key = (int(source["Dialogue_ID"]), int(source["Utterance_ID"]))
                if key in annotations:
                    raise ValueError(f"Duplicate MELD annotation in {split}: {key}")
                emotion = source["Emotion"].strip().lower()
                sentiment = source["Sentiment"].strip().lower()
                if emotion not in label_to_id or sentiment not in LABELS:
                    raise ValueError(f"Unknown MELD label for {split}: {key}")
                annotations[key] = (emotion, sentiment)
        output_rows = []
        for row in selected:
            key = (row["dialogue_id"], row["utterance_id"])
            if key not in annotations:
                raise ValueError(f"Selected clip missing annotation: {split} {key}")
            emotion, sentiment = annotations[key]
            if LABELS[row["label"]] != sentiment:
                raise ValueError(f"Sentiment changed for {split} {key}")
            output_rows.append({**row, "sentiment_label": row["label"], "label": label_to_id[emotion], "task": "emotion"})
        if {row["id"] for row in selected} != {row["id"] for row in output_rows}:
            raise AssertionError("Selection changed")
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts = Counter(EMOTION_LABELS[row["label"]] for row in output_rows)
        summary[split] = {"count": len(output_rows), "labels": {name: counts[name] for name in EMOTION_LABELS}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
