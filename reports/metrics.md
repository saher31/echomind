# EchoMind Metrics

The model was trained and tested in `notebooks/classification.ipynb`.

## Internal Test Set

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9726 |
| Loss | 0.1358 |
| Precision | 0.9935 |
| Recall | 0.9390 |

## External ACDC Evaluation

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9000 |
| Loss | 0.6535 |
| Precision | 0.9667 |
| Recall | 0.8286 |

## Notes

- Task: Binary cardiac MRI classification.
- Classes: `Normal`, `Sick`.
- Architecture: MobileNetV2 transfer learning with fine-tuning.
- Input shape: `224 x 224 x 3`.
- Output: one sigmoid score where values closer to `1` indicate `Sick`.
