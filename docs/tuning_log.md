# Tuning log

Validation split decides later trials. Test metrics are recorded after a run is accepted and are not used to choose the next setting. At most 10 trials follow a clean baseline. Code-fix retries of the baseline do not count.

## Baseline submission

- Commit: `498109b` on `standardize-outputs`
- Recipe: fine-tuned ResNet18, five seeds, two patient folds, 30 epochs, early stopping on `val_loss` (patience 8), AdamW `1e-4`, ReduceLROnPlateau, class-weight power 1, baseline geometric augmentation
- First attempt: image-only `366958` and multimodal `366959` on commit `498109b`. Fold 0 failed in under a minute because `PYTHON` was the system interpreter. Fold 1 was cancelled.
- Retry on `bdc8d2c`: image-only `366964` and multimodal `366965`. Fold 0 still failed on `ceres` because `.venv/bin/python` points at `/usr/bin/python3`, which is Python 3.10 on that node. Fold 1 started training but early stopping treated the initial infinite best loss as final, so a member stopped at epoch 8. Both jobs were cancelled.
- The virtualenv now points at Python 3.12.9 under `/vol/miltank/users/kalem/python`, and early stopping accepts the first finite validation loss. Partial run directories were removed.
- Current retry: commit `126d38d`, image-only `366994`, multimodal `366995`, `--array=0-1%1` on `master-queuesave`.
- Status: resubmitted, not yet accepted.
- Output groups: `baseline_image_only`, `baseline_multimodal`
