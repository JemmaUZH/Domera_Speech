import argparse
import json
from pathlib import Path

from speech_sentiment.features import extract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--system", choices=["audio", "text", "fusion", "mimi"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifests = [args.data_dir / f"{split}.jsonl" for split in ("train", "val", "test")]
    print(json.dumps(extract(manifests, args.cache_dir, args.system, args.device, args.force), indent=2))


if __name__ == "__main__":
    main()
