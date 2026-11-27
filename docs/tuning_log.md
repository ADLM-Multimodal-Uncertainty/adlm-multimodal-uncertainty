# Tuning log

Validation split decides later trials. Test metrics are recorded after a run is accepted and are not used to choose the next setting. At most 10 trials follow a clean baseline. Code-fix retries of the baseline do not count.

## Baseline submission

- Commit: `498109b` on `standardize-outputs`
- Recipe: fine-tuned ResNet18, five seeds, two patient folds, 30 epochs, early stopping on `val_loss` (patience 8), AdamW `1e-4`, ReduceLROnPlateau, class-weight power 1, baseline geometric augmentation
- First attempt: image-only `366958` and multimodal `366959` on commit `498109b`. Fold 0 failed in under a minute because `PYTHON` was the system interpreter. Fold 1 was cancelled.
- Retry on `bdc8d2c`: image-only `366964` and multimodal `366965`. Fold 0 still failed on `ceres` because `.venv/bin/python` points at `/usr/bin/python3`, which is Python 3.10 on that node. Fold 1 started training but early stopping treated the initial infinite best loss as final, so a member stopped at epoch 8. Both jobs were cancelled.
- The virtualenv now points at Python 3.12.9 under `/vol/miltank/users/kalem/python`, and early stopping accepts the first finite validation loss. Partial run directories were removed.
- Accepted run: commit `126d38d`, image-only `366994`, multimodal `366995`. Both folds completed in 31–33 minutes with empty error logs.
- Output groups: `baseline_image_only`, `baseline_multimodal`
- Report: `outputs/result-analysis/default_baseline` on the validation split (463 images, 276 patients).

## Baseline result

Training was consistent. All 20 members had finite validation loss and accuracy. Every member's best validation loss beat epoch 1. Early stopping kept that best epoch, after 12–17 epochs. Accuracy was not constant.

Validation, pooled:

- Image-only: accuracy 68.7%, balanced accuracy 67.4%, macro-F1 65.4%, NLL 0.858, Brier 0.446, epistemic uncertainty 0.106 nats.
- Image + metadata: accuracy 70.6%, balanced accuracy 67.7%, macro-F1 67.1%, NLL 0.849, Brier 0.438, epistemic uncertainty 0.087 nats.
- Metadata changed balanced accuracy by +0.35 percentage points and macro-F1 by +1.67. Mean epistemic uncertainty fell by 0.018 nats (17% relative, 62% of images). Total and aleatoric uncertainty rose.

Members still overfit after the selected epoch: training accuracy climbs above 90% while validation accuracy stays near 65% and validation loss turns upward. The next trial softens the class weights rather than raising the learning rate.

## Trial 1: class-weight power 0.5

- Change from the accepted baseline: `train.class_weight_power=0.5` only.
- Image-only job `367016`, run group `power_0.5_image_only`.
- Multimodal job `367017`, run group `power_0.5_multimodal`.
- Same commit `126d38d`, `--array=0-1%1`, `master-queuesave`.
- Status: submitted. Trials used: 1 of 10.
