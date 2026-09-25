# CREMA-D Supervised Follow-up Artifacts

This is a supplementary six-class emotion experiment. The frozen pretrained encoders are unchanged; separate MLP classifier heads were trained on CREMA-D. Training used 1,000 train and 200 speaker-disjoint validation utterances; validation Macro-F1 selected the checkpoint. Evaluation used the fixed, speaker-disjoint 200-utterance test subset. No audio, transcript, or CREMA-D manifest is redistributed here.

| System | Macro-F1 | Weighted-F1 | Accuracy |
|---|---:|---:|---:|
| Emotion2Vec audio-only | 0.649 | 0.647 | 65.0% |
| Emotion2Vec + Text | 0.643 | 0.643 | 64.5% |
| Continuous Mimi | 0.644 | 0.643 | 64.5% |

Each system folder contains the complete per-class metrics, test predictions and probabilities, confusion matrix, validation training history, and feature extraction summary. `run_config.json` records the run setup and `comparison.json` contains full-precision aggregate metrics. The three classifier checkpoints are included under `checkpoints/`; their SHA-256 digests are in `checkpoint_sha256.json`.

To regenerate the audio and manifests, follow the CREMA-D instructions in the repository README and run `prepare_crema_d.py` followed by `prepare_crema_d_supervised.py`. Run `train_crema_d_supervised.py --system all` to recreate these metrics and checkpoints. Model and audio caches are written outside this artifact directory.
