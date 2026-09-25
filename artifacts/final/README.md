# Final MELD Release Artifacts

This folder contains the files needed by the two core project paths described in the repository README.

- `checkpoints/sentiment/` and `checkpoints/emotion/`: the three validated local classifier checkpoints per task. These are byte-for-byte copies of the existing trained checkpoints.
- `split_ids.json`: the fixed MELD sample IDs only; it contains no audio paths or utterance text.
- `results.json`: final metrics, per-class scores, confusion matrices, split distributions, and sample hashes.
- `sentiment_diagnostic.json`: text-only and fusion diagnostics, including the text sanity examples.
- `checkpoint_sha256.json`: SHA-256 checksums for all included checkpoint files.

The raw MELD media, feature caches, intermediate runs, external API outputs, and credentials are not included. See the root README for installation, demo, data setup, and evaluation commands.
