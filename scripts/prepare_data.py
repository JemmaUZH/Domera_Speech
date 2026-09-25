import argparse
import json
from pathlib import Path

from speech_sentiment.subset import DEFAULT_SIZES, prepare_subset


def main():
    parser = argparse.ArgumentParser(description="Create one stratified MELD subset from raw clips or MELD.Raw.tar.gz")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--meld-root", type=Path, help="Directory with extracted official raw clips")
    source.add_argument("--meld-archive", type=Path, help="Official MELD.Raw.tar.gz; selected clips only are converted")
    parser.add_argument("--annotations-root", type=Path, default=Path("data/annotations"), help="Directory containing the three official *_sent_emo.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-size", type=int, default=DEFAULT_SIZES["train"])
    parser.add_argument("--val-size", type=int, default=DEFAULT_SIZES["val"])
    parser.add_argument("--test-size", type=int, default=DEFAULT_SIZES["test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reserve-per-class", type=int, default=20, help="Backup clips per class for missing/corrupt media")
    args = parser.parse_args()
    source_path = args.meld_root or args.meld_archive
    sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    print(json.dumps(prepare_subset(source_path, args.annotations_root, args.output_dir, sizes, seed=args.seed, reserve_per_class=args.reserve_per_class), indent=2))


if __name__ == "__main__":
    main()
