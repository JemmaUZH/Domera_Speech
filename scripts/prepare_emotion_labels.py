import argparse
import json
from pathlib import Path

from speech_sentiment.emotion import prepare_emotion_manifests


def main():
    parser = argparse.ArgumentParser(description="Relabel the existing MELD subset with its seven official emotions")
    parser.add_argument("--selected-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/emotion"))
    args = parser.parse_args()
    print(json.dumps(prepare_emotion_manifests(args.selected_dir, args.annotations_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
