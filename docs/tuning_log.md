# Tuning log

Validation split decides later trials. Test metrics are recorded after a run is accepted and are not used to choose the next setting. At most 10 trials follow a clean baseline. Code-fix retries of the baseline do not count.

## Baseline submission

- Commit: `498109b` on `standardize-outputs`
- Recipe: fine-tuned ResNet18, five seeds, two patient folds, 30 epochs, early stopping on `val_loss` (patience 8), AdamW `1e-4`, ReduceLROnPlateau, class-weight power 1, baseline geometric augmentation
- First attempt: image-only `366958` and multimodal `366959` on commit `498109b`. Fold 0 failed in under a minute because `PYTHON` was the system interpreter. Fold 1 was cancelled.
- Retry: commit `bdc8d2c`, image-only `366964`, multimodal `366965`, still `--array=0-1%1` on `master-queuesave`.
- Status: resubmitted, not yet accepted.
- Output groups: `baseline_image_only`, `baseline_multimodal`
