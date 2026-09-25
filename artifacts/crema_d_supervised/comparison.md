# CREMA-D supervised emotion results

The three classifier heads were trained using 1,000 CREMA-D train utterances and selected on 200 speaker-disjoint validation utterances by Macro-F1. Pretrained acoustic encoders were frozen. All systems were evaluated on the same fixed, speaker-disjoint 200-example test set.

| System | Macro-F1 | Weighted-F1 | Accuracy |
|---|---:|---:|---:|
| Emotion2Vec audio-only | 0.649 | 0.647 | 65.0% |
| Emotion2Vec + Text | 0.643 | 0.643 | 64.5% |
| Continuous Mimi | 0.644 | 0.643 | 64.5% |

The target labels are anger, disgust, fear, joy, neutral, and sadness. These are CREMA-D-trained heads and must be treated as a separate experiment from the MELD-trained models and MELD benchmark results. Per-class metrics, confusion matrices, predictions, validation histories, full-precision metrics, and checkpoint hashes are included beside this file. The run configuration records the shared test-manifest hash and package versions.

To reproduce, download the official CREMA-D subset and create the speaker-disjoint 1,000/200 training manifests using the steps in the repository README, then run `train_crema_d_supervised.py --system all`. The audio and generated caches are not included in this artifact.
