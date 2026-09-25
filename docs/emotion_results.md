# Seven-emotion MELD results

**Run:** 24 September 2026 on Apple M1 Pro, 16 GB RAM, macOS arm64. The three systems used the same frozen encoders, feature cache, 1,000 training clips, 200 validation clips, and 200 test clips as the earlier three-class sentiment comparison. The official MELD `Emotion` labels were applied to the existing selected IDs. No test labels were used for model selection. These results are stored separately in `data/emotion/` and `runs/emotion/`; the three-class sentiment artifacts were preserved unchanged.

## Class counts

| Split | anger | disgust | fear | joy | neutral | sadness | surprise | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 120 | 28 | 20 | 169 | 471 | 66 | 126 | 1,000 |
| Validation | 23 | 4 | 10 | 29 | 85 | 20 | 29 | 200 |
| Test | 28 | 7 | 3 | 31 | 96 | 18 | 17 | 200 |

The selected subset was originally stratified by **sentiment**, not emotion. Fear and disgust have very small test supports, so their per-class F1 and the seven-class macro-F1 are sensitive to individual examples. The three emotion systems still use exactly the same examples.

## Quality and inference cost

The checkpoint for each system was chosen by validation macro-F1. Live inference used CPU for all systems, batch size one, one untimed warmup clip, and the same 200 test WAVs. RTF is total inference time divided by total audio duration.

| System | Validation macro-F1 | Test macro-F1 | Test weighted-F1 | Test accuracy | Mean latency / utterance | RTF | Peak process RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| emotion2vec Base audio-only | 0.257 | 0.179 | 0.269 | 0.240 | 0.146 s | 0.046 | 2.928 GiB |
| emotion2vec + ASR/MiniLM | **0.315** | **0.262** | **0.403** | **0.390** | 0.887 s | 0.278 | 2.931 GiB |
| Continuous Mimi | 0.242 | 0.250 | 0.370 | 0.360 | **0.079 s** | **0.025** | **1.853 GiB** |

| System | anger | disgust | fear | joy | neutral | sadness | surprise |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio-only F1 | 0.294 | 0.065 | 0.000 | 0.179 | 0.343 | 0.192 | 0.182 |
| Fusion F1 | 0.262 | 0.143 | 0.000 | 0.423 | 0.529 | 0.258 | 0.222 |
| Mimi F1 | 0.300 | 0.308 | 0.000 | 0.289 | 0.503 | 0.118 | 0.235 |

All three models missed the three fear clips. The fusion model's largest per-class advantages over audio-only were on joy and neutral. The class-weighted F1 exceeds macro-F1 because neutral accounts for 96 of 200 test clips.

Fusion's mean component time was 0.117 s for emotion2vec, 0.761 s for ASR, 0.009 s for MiniLM, and less than 0.001 s for the classifier. The branches were run sequentially. Known model parameters are 93.9M for audio-only, 116.7M for fusion, and 79.4M for Mimi; corresponding known model-state sizes are 358, 445, and 367 MiB. Fusion's parameter and state totals exclude the CTranslate2 ASR model, while latency and peak RSS include it.

## Errors and reproducibility

On the 200 test clips, 44 examples were wrong for audio-only but correct for fusion; 14 went the other way; 108 were wrong for both; Mimi disagreed with fusion on 116. These groups overlap. Aggregate metrics and all three 7×7 confusion matrices are included in [`artifacts/final/results.json`](../artifacts/final/results.json). Per-utterance failure-review files and latency traces remain local and are not included in the release bundle.

The selected IDs and audio paths were verified equal to the earlier sentiment manifests for train, validation, and test. All cached audio, text, and Mimi features were present; the emotion task required no new feature extraction. The three emotion reports have matching sample hashes and benchmark IDs. The sample IDs and hashes are included in the release artifacts. The final local test suite contains 20 passing tests.
