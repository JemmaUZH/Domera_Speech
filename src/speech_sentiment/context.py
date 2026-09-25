"""MELD dialogue context reconstruction and context-text feature extraction."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from . import EMOTION_LABELS
from .features import TextEncoder, feature_path, write_vector
from .meld import CSV_NAMES, find_unique, read_manifest


def _read_dialogues(annotations_dir: Path) -> dict[tuple[str, int], list[dict]]:
    dialogues: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for split, filename in CSV_NAMES.items():
        with find_unique(annotations_dir, filename).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"Dialogue_ID", "Utterance_ID", "Utterance", "Speaker", "Emotion"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing MELD dialogue fields in {filename}")
            seen = set()
            for source_index, raw in enumerate(reader):
                dialogue_id = int(raw["Dialogue_ID"])
                utterance_id = int(raw["Utterance_ID"])
                key = (split, dialogue_id, utterance_id)
                if key in seen:
                    raise ValueError(f"Duplicate official MELD turn: {key}")
                seen.add(key)
                item = {
                    "split": split,
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "source_index": source_index,
                    "speaker_name": raw["Speaker"].strip(),
                    "speaker_id": raw["Speaker"].strip(),
                    "transcript": raw["Utterance"].strip(),
                    "emotion": raw["Emotion"].strip().lower(),
                }
                if item["emotion"] not in EMOTION_LABELS:
                    raise ValueError(f"Unknown MELD emotion in {key}: {item['emotion']}")
                dialogues[(split, dialogue_id)].append(item)
    for key, turns in dialogues.items():
        utterance_ids = [turn["utterance_id"] for turn in turns]
        if any(right <= left for left, right in zip(utterance_ids, utterance_ids[1:])):
            raise ValueError(f"MELD turn order is not increasing in official CSV order for {key}")
    return dialogues


def reconstruct_context(
    selected_dir: Path, annotations_dir: Path, asr_cache_dir: Path, output: Path,
    *, max_previous_turns: int = 3,
) -> dict:
    if max_previous_turns < 0:
        raise ValueError("max_previous_turns cannot be negative")
    dialogues = _read_dialogues(annotations_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {"max_previous_turns": max_previous_turns, "label_order": list(EMOTION_LABELS), "splits": {}}
    for split in ("train", "val", "test"):
        targets = read_manifest(selected_dir / f"{split}.jsonl")
        if len(targets) != {"train": 1000, "val": 200, "test": 200}[split]:
            raise ValueError(f"Expected fixed {split} subset size, got {len(targets)}")
        if any(row.get("task") != "emotion" or row.get("split") != split for row in targets):
            raise ValueError(f"Expected seven-class emotion rows for {split}")
        built = []
        for target in targets:
            dialogue_key = (split, target["dialogue_id"])
            turns = dialogues.get(dialogue_key)
            if turns is None:
                raise ValueError(f"Official dialogue missing for target {target['id']}")
            target_index = target["utterance_id"]
            current = next((turn for turn in turns if turn["utterance_id"] == target_index), None)
            if current is None:
                raise ValueError(f"Official MELD target turn missing for {target['id']}")
            target_speaker = target.get("speaker_id") or "Unknown"
            if target_speaker != current["speaker_id"]:
                raise ValueError(f"Speaker mismatch for {target['id']}")
            current_emotion = EMOTION_LABELS[target["label"]]
            if current_emotion != current["emotion"]:
                raise ValueError(f"Emotion mapping mismatch for {target['id']}")
            prior = [
                turn for turn in turns
                if turn["dialogue_id"] == target["dialogue_id"]
                and turn["source_index"] < current["source_index"]
                and turn["utterance_id"] < current["utterance_id"]
            ][-max_previous_turns:] if max_previous_turns else []
            for turn in prior:
                if turn["source_index"] >= current["source_index"] or turn["utterance_id"] >= current["utterance_id"]:
                    raise AssertionError(f"Future context leak for {target['id']}")
            asr_path = asr_cache_dir / split / f"{target['id']}.json"
            if not asr_path.is_file():
                raise FileNotFoundError(f"Existing Audio + Text ASR cache missing: {asr_path}")
            current_asr = json.loads(asr_path.read_text(encoding="utf-8"))["text"]
            target_line = f"[CURRENT] {target_speaker}: {current_asr}"
            context_lines = [f"{turn['speaker_name']}: {turn['transcript']}" for turn in prior]
            context_lines.append(target_line)
            item = {
                **target,
                "context_turns": [
                    {
                        "dialogue_id": turn["dialogue_id"],
                        "utterance_id": turn["utterance_id"],
                        "source_index": turn["source_index"],
                        "speaker_name": turn["speaker_name"],
                        "speaker_id": turn["speaker_id"],
                        "transcript": turn["transcript"],
                    }
                    for turn in prior
                ],
                "current_asr_transcript": current_asr,
                "context_text": "\n".join(context_lines),
                "current_source": "existing_local_asr_cache",
            }
            built.append(item)
        if [r["id"] for r in built] != [r["id"] for r in targets]:
            raise AssertionError(f"IDs/order changed in {split}")
        summary["splits"][split] = {
            "count": len(built),
            "id_label_hash": hashlib.sha256(
                json.dumps([(r["id"], r["label"]) for r in built], separators=(",", ":")).encode()
            ).hexdigest(),
            "contexts_with_previous_turns": sum(bool(r["context_turns"]) for r in built),
        }
        all_rows.extend(built)
    with output.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    split_dir = output.parent / "manifests"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        with (split_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in (item for item in all_rows if item["split"] == split):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["total"] = len(all_rows)
    summary["output"] = str(output)
    summary["context_turns_verified_same_dialogue_and_strictly_prior"] = True
    return summary


def print_context_examples(context_path: Path, *, count: int = 5, seed: int = 42) -> list[dict]:
    rows = [json.loads(line) for line in context_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Keep the held-out test split unseen before training begins.
    candidates = [row for row in rows if row["split"] in ("train", "val")]
    selected = random.Random(seed).sample(candidates, min(count, len(candidates)))
    for index, row in enumerate(selected, start=1):
        print(f"Example {index} | {row['split']} | {row['id']} | gold={EMOTION_LABELS[row['label']]}")
        for turn in row["context_turns"]:
            print(f"  {turn['speaker_name']} (speaker_id={turn['speaker_id']}): {turn['transcript']}")
        print(f"  [CURRENT] {row.get('speaker_id') or 'Unknown'}: {row['current_asr_transcript']}")
    return selected


def extract_context_text_features(
    context_path: Path, cache_dir: Path, *, device: str = "cpu", batch_size: int = 64,
) -> dict:
    rows = [json.loads(line) for line in context_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    encoder = TextEncoder(device)
    missing = [row for row in rows if not feature_path(cache_dir, "text", row).is_file()]
    for start in range(0, len(missing), batch_size):
        for row in missing[start:start + batch_size]:
            vector = encoder.encode(row["context_text"])
            write_vector(feature_path(cache_dir, "text", row), vector)
    return {
        "model": TextEncoder.model_id,
        "device": device,
        "rows": len(rows),
        "reused_context_embeddings": len(rows) - len(missing),
        "new_context_embeddings": len(missing),
        "existing_audio_embeddings_reused": True,
    }
