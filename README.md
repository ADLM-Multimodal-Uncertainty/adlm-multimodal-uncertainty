# The Value of a Second Modality: Epistemic Uncertainty in Multimodal Skin Lesion Classification

ADLM practical, WiSe 2026/27, Chair for AI in Medicine and Healthcare (TUM). Supervisor: Raphael.
Team: Larissa Manalil, Yannic Neu, Emre Kalkan.

Research question: does adding clinical metadata to an image classifier reduce its epistemic
uncertainty? Per test image, dEU = EU(image) - EU(image + metadata). Dataset: PAD-UFES-20.

## Layout

```
configs/            experiment configs, one group per concern
  data/               dataset paths, image size, metadata variant
  model/              image-only, fusion, modality-dropout
  uq/                 ensemble, MC dropout, Laplace
  train/              optimiser, schedule, epochs
data/               not committed (except splits/)
  raw/                PAD-UFES-20 images + metadata as downloaded
  processed/          derived data
  splits/             patient-grouped fold assignment (committed)
docs/               code-related documentation
  diagrams/           architecture and data-flow diagrams
  cluster_guide.md    VPN, ssh, storage, SLURM
  eda_plan.md         EDA plan and findings
src/adlm_uq/        Python package
  data/               dataset, splits, metadata encoding, transforms
  models/             image encoder, tabular encoder, fusion
  uq/                 uncertainty methods and TU/AU/EU decomposition
  evaluation/         metrics and statistics
scripts/slurm/      sbatch job scripts
notebooks/          exploration only, never training or preprocessing
tests/              unit tests
results/            final figures and tables for slides, poster, report
```

## Conventions

- Commit before every training run; record the commit hash with the run.
- No training or preprocessing in notebooks; scripts take config overrides, no hard-coded paths.
- Splits are grouped by `patient_id`; `lesion_id` is only unique within a patient.
- Data, checkpoints and outputs live on `/vol/miltank/...` on the cluster, never in the repo.
- One branch per feature or experiment; PR with one review into `main`.
