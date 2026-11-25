# Yannic: final paired ensemble setup

Prepared 2026-11-25 on branch `experiments-yannic`.

## Objective and run matrix

Compare image-only with image + clinical metadata on PAD-UFES-20 and test whether
metadata reduces ensemble-based epistemic uncertainty while maintaining or improving
classification, calibration and error identification.

| Modality | Patient folds | Members per fold | Total models |
|---|---|---|---|
| Image only | 0, 1 | Seeds 0, 1, 2, 3, 4 | 10 |
| Image + metadata | 0, 1 | Seeds 0, 1, 2, 3, 4 | 10 |

Each fold has its own ensemble. Never average predictions from models trained on
different test folds. Each patient is tested exactly once across the two folds.
Use image IDs to align paired outputs and patient IDs for bootstrap resampling.

## Fixed protocol

- Split: patient-grouped `GroupKFold(n_splits=2)`, generated deterministically from
  the same metadata ordering for both modalities; no class stratification.
- For each held-out fold, reserve 20% of the remaining patients for validation
  (`split.seed=42`), leaving approximately 40/10/50 train/validation/test overall.
  Image proportions can differ from patient proportions. Record actual counts.
- Backbone: ImageNet-pretrained ResNet18, fully fine-tuned, including BatchNorm.
  Image encoder projection: 128 units, ReLU, dropout 0.2.
- Separate model and optimizer for every seed, modality and fold. No shared
  trainable parameters or checkpoints, bootstrap resampling, or warm starts from
  another member. Matching seeds give matching starting image-encoder weights across
  modalities; pretrained backbone weights are identical across all models. Member
  diversity comes from random heads/projections, dropout, augmentation and batch order.
- Multimodal branch: training-only median imputation and standardization for age and
  lesion diameters; one-hot encoding for configured categorical clinical variables;
  64-unit metadata MLP. Image-only removes the metadata branch. No biopsy/ID features.
- Fusion: concatenation, 128-unit hidden head, dropout 0.3, six output classes.
- Images: EXIF orientation correction, RGB, resize 224×224, ImageNet normalization;
  training-only random horizontal flip. Validation/test transforms deterministic.
- Optimizer: AdamW, learning rate 1e-4, default weight decay 0.01, no scheduler.
- Loss: inverse-training-class-frequency weighted cross-entropy. Weights are fit
  separately per training fold and identical across modalities/members in that fold.
- Budget: 30 epochs, batch size 32. No early stopping; retain each member's first
  best validation macro-F1 checkpoint. Training and validation history saved each epoch.
- Ensemble: arithmetic mean of five members' probabilities, never mean logits.
- Evaluate the selected checkpoint once at run completion. Root aggregation currently
  re-evaluates members to produce aligned archives; this does not train or select again.
- Calibration and threshold fitting are deferred and use validation predictions only.
  Test labels must not inform hyperparameter or threshold selection.

## Run commands

The shared preset is `configs/experiment/yannic_2fold_ensemble.yaml`; image-only
inherits it and disables the metadata encoder. Print both composed configs first:

```bash
uv run python -m src.train experiment=yannic_2fold_ensemble --cfg job
uv run python -m src.train experiment=yannic_2fold_ensemble_image_only --cfg job
```

Run all four ensembles, sequentially through Hydra's standard launcher:

```bash
uv run python -m src.train -m \
  experiment=yannic_2fold_ensemble_image_only,yannic_2fold_ensemble \
  fold=0,1 \
  data.root=/ABSOLUTE/PATH/TO/PAD_UFES_20 \
  output_root=/ABSOLUTE/PATH/TO/OUTPUTS \
  device=cuda
```

On a SLURM cluster launch through an allocated GPU job, never on its login node.
Commit the code before cluster training. Existing checkpoints must not be overwritten;
choose a distinct `name` for any changed protocol or restart. Paths above are placeholders;
no real experiment has been launched without verified data and compute access.

Outputs are `<output_root>/runs/yannic_2fold_<image_only|image_metadata>-patient_kfold-f<0|1>-s42/`.
The top-level seed 42 is a run label; member training seeds are 0–4 and split seed is 42.

## Outputs required before analysis

Each member saves:

- `best.pt`, resolved `config.yaml`, `history.csv`, basic test metrics/predictions and
  training/confusion plots.
- `val_predictions.npz` and `test_predictions.npz`: uncalibrated raw logits and class
  probabilities, integer labels, class order, image IDs and patient IDs, at best checkpoint.
- `split_metadata.csv`: exact ordered partition rows with original clinical values,
  lesion IDs where present, and missing entries; `metadata_missingness.csv` explicitly
  records missingness for every configured feature.
- `metadata_preprocessor.joblib`: fitted training-only preprocessing. Re-evaluation
  restores this and saved split rows, so mutable source metadata cannot change membership
  or feature fitting. Legacy runs fall back to rebuilding from source metadata.
- `image_hashes.json` and the source metadata SHA256 identify the original dataset;
  preserve the original images separately for re-inference and qualitative figures.
- `provenance.json`: code commit/status, package versions, Python, data configuration,
  member seed, runtime device/backend settings and starting image-encoder weight hash.
- `source_snapshot/`: actual source/configs/docs/scripts and dependency manifests;
  `source_changes.patch`: tracked changes against HEAD.

Each ensemble root saves validation and test `*_member_predictions.npz` with shape
`(5, image, 6)` for both logits and probabilities, plus member seeds and sample/class
identities. Member label/ID/class alignment is validated before aggregation. Root test
CSV includes mean probabilities, predictive entropy, expected member entropy and
mutual information, all in nats. No probabilities need to be inferred from rounded figures.

These artifacts support the complete [analysis plan](uncertainty_analysis.md) without
retraining: probability quality/calibration, paired uncertainty changes, correctness
transitions, error detection, selective prediction, subgroups and ensemble-size analyses.
Write the analysis code after training. Quantitative analysis needs saved predictions;
image galleries additionally require original images. Fit temperature/thresholds on
validation; use paired patient-level bootstrap intervals on held-out predictions.
Repeated ensemble training would be needed to estimate variability from training randomness.

## Validation and launch status

Synthetic tests cover probability/logit equivalence, both modalities with two-fold
patient splits, preprocessing/provenance artifacts, checkpoint reconstruction after
removal of original metadata, and matching initial image encoder weights across modalities.
Record test results and actual data/compute paths here before real launch.

Validation on 2026-11-25: all 73 tests passed using the existing CPU environment
`/tmp/adlm-fold-tests/bin/python` with `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
and `MPLCONFIGDIR=/tmp/adlm-mpl`. Changed Python files pass Ruff and `git diff --check`.
Hydra composition succeeded for the shared preset; config tests cover both variants.
Real launch is pending the dataset path and usable training host: the current machine
has no working CUDA device and the dataset is absent from searched workspace/data paths.

## Verified VM dataset and Windows handoff

The dataset is now verified at
`/home/yannic-neu/code/uni/adlm/adlm-multimodal-uncertainty/data/pad_ufes_20`.
It contains 2,298 metadata rows, 1,373 patients, and all referenced image files.
All six classes are present in each partition of both folds.

| Fold | Train images / patients | Validation images / patients | Test images / patients |
|---|---|---|---|
| 0 | 925 / 549 | 224 / 138 | 1,149 / 686 |
| 1 | 910 / 548 | 239 / 138 | 1,149 / 687 |

Validation melanoma counts are 7 and 3 respectively; checkpoint selection may be noisy.
Both modalities retain these exact same splits; do not adjust splits using test outcomes.
The earlier missing-data launch blocker is resolved. GPU training remains deferred:
the VM currently exposes no working CUDA device.

The dataset need not move. The active worktree can access it through the absolute
`data.root` override. `data/pad_ufes_20` is the repository default relative to the
checkout; the /vol/miltank storage rule applies to chair-cluster jobs. Data and outputs
are ignored by Git and must be transferred separately to a Windows/WSL training host.
The prepared source changes must be committed and transferred/pushed before cloning
can reproduce this setup; a normal remote clone cannot see local uncommitted files.

Preferred handoff, assuming a supported NVIDIA GPU: install Ubuntu under WSL2 on the
physical Windows GPU host, clone/check out branch experiments-yannic inside WSL's Linux
filesystem, copy data there, install the project environment, and verify
`torch.cuda.is_available()` before training. WSL uses the Windows NVIDIA driver;
do not install a Linux NVIDIA display driver inside WSL.
WSL is a Windows feature, not an add-on installed inside this existing Ubuntu VM.
Accessing the existing VM from Windows alone does not grant it GPU access.

References:
- https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl
- https://docs.nvidia.com/cuda/wsl-user-guide/
