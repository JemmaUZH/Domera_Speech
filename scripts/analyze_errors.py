"""Export observed failures for manual review without inventing error categories."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="*.predictions.jsonl from evaluate.py")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("runs/errors.csv"))
    args = parser.parse_args()
    rows = []
    for line in args.predictions.read_text().splitlines():
        item = json.loads(line)
        if item["gold"] == item["predicted"]:
            continue
        asr_file = args.cache_dir / "asr" / args.split / f"{item['id']}.json"
        asr = json.loads(asr_file.read_text())["text"] if asr_file.exists() else ""
        rows.append({
            "id": item["id"], "gold": item["gold"], "predicted": item["predicted"],
            "prediction_confidence": max(item["probabilities"]),
            "reference_transcript": item.get("reference_transcript") or "",
            "asr_transcript": asr,
            "observed_error_category": "", "notes": "",
        })
    rows.sort(key=lambda row: row["prediction_confidence"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id", "gold", "predicted", "prediction_confidence", "reference_transcript", "asr_transcript", "observed_error_category", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} failures to {args.output}")


if __name__ == "__main__":
    main()
