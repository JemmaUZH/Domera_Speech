# MELD subset results

**Run:** 24 September 2026, Apple M1 Pro, 16 GB RAM, macOS arm64, Python 3.12. Live inference used CPU for all systems, batch size one, with one untimed warmup pass before measuring all 200 test clips. The three checkpoints were selected on validation macro-F1, then evaluated once on the same test IDs. No test labels were used for model selection.

## Data

The official MELD train/dev/test boundaries were retained. A seed-42 stratified sample selected 1,000 train, 200 validation, and 200 test utterances. Counts by negative / neutral / positive were 295 / 471 / 234 for train, 73 / 85 / 42 for validation, and 64 / 96 / 40 for test. One corrupt training clip (`dia125_utt3`) was replaced by a same-class reserve. The three evaluation reports have identical train, validation, and test sample hashes. The fixed sample IDs are included in [`artifacts/final/split_ids.json`](../artifacts/final/split_ids.json); raw media is not redistributed.

## Quality and inference cost

| System | Validation macro-F1 | Test macro-F1 | Test accuracy | Mean latency / utterance | RTF | Peak process RSS |
|---|---:|---:|---:|---:|---:|---:|
| emotion2vec Base audio-only | 0.450 | 0.430 | 0.445 | 0.144 s | 0.045 | 2.929 GiB |
| emotion2vec + ASR/MiniLM | 0.516 | 0.528 | 0.520 | 0.847 s | 0.266 | 2.930 GiB |
| Continuous Mimi | 0.451 | 0.448 | 0.460 | 0.097 s | 0.031 | 1.822 GiB |

The fusion system's mean stage latency was 0.123 s for emotion2vec, 0.715 s for ASR, 0.010 s for MiniLM, and less than 0.001 s for the classifier. These stages ran sequentially. End-to-end latency includes all of them. RTF is total inference seconds divided by total audio seconds.

| System | Negative F1 | Neutral F1 | Positive F1 | Known parameters | Known model-state size |
|---|---:|---:|---:|---:|---:|
| audio-only | 0.419 | 0.506 | 0.366 | 93.9M | 358 MiB |
| fusion | 0.476 | 0.510 | 0.600 | 116.7M* | 445 MiB* |
| Mimi | 0.464 | 0.497 | 0.382 | 79.4M | 367 MiB |

\* Fusion parameter and model-state totals exclude faster-whisper's CTranslate2 ASR model, which is included in latency and peak RSS. Peak RSS is a process high-water mark, measured in a fresh process for each system.

Confusion matrices use rows=true and columns=predicted in negative / neutral / positive order:

- Audio-only: `[[31, 21, 12], [38, 43, 15], [15, 10, 15]]`
- Fusion: `[[34, 18, 12], [38, 40, 18], [7, 3, 30]]`
- Mimi: `[[32, 23, 9], [30, 43, 23], [12, 11, 17]]`

## Failure review and decision

On the 200 test clips, 38 cases were wrong for audio-only and correct for fusion; 23 went the other way; 73 were wrong for both; Mimi disagreed with fusion on 98. These are overlapping groups. The per-utterance failure-review CSV is local and is not included in the release bundle; no subjective error causes were assigned automatically.

Fusion gained about 0.098 test macro-F1 over audio-only and 0.081 over Mimi, with its clearest per-class gain on positive sentiment. Its average latency was about 5.9 times audio-only and 8.7 times Mimi, largely due to ASR. For a quality-first assistant on this MELD subset, fusion is the strongest candidate; for a strict latency or memory budget, Mimi is a viable alternative to investigate. These conclusions are specific to 200 scripted TV-dialogue test clips. The official split reuses actors, confidence scores are uncalibrated, and no chunked streaming inference was measured.

## Reproduction details

Frozen encoders: `iic/emotion2vec_base`, faster-whisper `base.en`, `sentence-transformers/all-MiniLM-L6-v2`, and Kyutai's unquantized Mimi latent from `tokenizer-e351c8d8-checkpoint125.safetensors`. Classifiers used a 128-unit GELU MLP with dropout 0.2, class-weighted cross entropy, seed 42, and validation macro-F1 selection. The installed versions for this run were torch 2.5.1, FunASR 1.4.16, ModelScope 1.31.0, faster-whisper 1.2.1, sentence-transformers 5.7.0, and moshi 0.2.13. Full machine-readable outputs and per-utterance timings are in local `runs/` files. Mimi feature extraction used CPU because longer clips failed under Apple MPS; the other cached embeddings were extracted using MPS. Live benchmarks all used CPU.
