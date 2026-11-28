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
- Both folds completed. Image-only `367016` took 34 and 31 minutes on `prometheus`. Multimodal `367017` took 27 and 29 minutes on `ceres`. Error logs were empty.
- All 20 members had finite validation loss and accuracy. Every best validation loss beat epoch 1. Early stopping kept that best epoch, after 12–19 epochs. Accuracy was not constant.
- Report: `outputs/result-analysis/power_0.5_vs_baseline` on the validation split (463 images, 276 patients).

## Trial 1 result

Rejected. The current best stays the accepted baseline (`class_weight_power=1.0`).

Validation, pooled:

- Image-only: accuracy 73.2%, balanced accuracy 62.7%, macro-F1 64.8%, NLL 0.808, Brier 0.406, epistemic uncertainty 0.094 nats.
- Image + metadata: accuracy 71.5%, balanced accuracy 59.1%, macro-F1 61.9%, NLL 0.777, Brier 0.395, epistemic uncertainty 0.087 nats.

Against the baseline, balanced accuracy fell by 4.7 percentage points for image-only and 8.6 for image + metadata. Macro-F1 fell by 0.6 and 5.2. Accuracy, NLL, and Brier improved. That is a shift toward the common classes: SCC F1 fell from 28.9% to 17.0% (image-only) and from 22.5% to 9.1% (image + metadata); melanoma recall fell from 80% to 60% and 50%.

Metadata no longer helps classification. Balanced accuracy changed by −3.6 percentage points and macro-F1 by −3.0. Mean epistemic uncertainty fell by only 0.007 nats (7% relative, 54% of images), against 0.018 nats (17%, 62%) on the baseline.

Power 0 would weaken the rare-class weights further, so that axis stops here. Training accuracy still ends near 98% while the selected validation accuracy stays near 65–73%. The next trial lowers the learning rate on the accepted baseline.

## Trial 2: learning rate 3e-5

- Change from the accepted baseline: `train.optimizer.lr=3e-5` only. Class-weight power stays 1.0.
- Image-only job `367028`, run group `lr_3e-5_image_only`.
- Multimodal job `367029`, run group `lr_3e-5_multimodal`.
- Commit `14e993e` on the cluster, `--array=0-1%1`, `master-queuesave`. Training code is unchanged from `126d38d`.
- Status: submitted. Trials used: 2 of 10.
