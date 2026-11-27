# Tuning log

Validation split decides later trials. Test metrics are recorded after a run is accepted and are not used to choose the next setting. At most 10 trials follow a clean baseline. Code-fix retries of the baseline do not count.

## Baseline submission

- Commit: `498109b` on `standardize-outputs`
- Recipe: fine-tuned ResNet18, five seeds, two patient folds, 30 epochs, early stopping on `val_loss` (patience 8), AdamW `1e-4`, ReduceLROnPlateau, class-weight power 1, baseline geometric augmentation
- Image-only array: SLURM job `366958`, `--array=0-1%1`
- Multimodal array: SLURM job `366959`, `--array=0-1%1`
- QoS: `master-queuesave` (two jobs, one task from each array)
- Status: submitted, not yet accepted
- Output groups: `baseline_image_only`, `baseline_multimodal`
