# The Value of a Second Modality: Epistemic Uncertainty in Multimodal Skin Lesion Classification

ADLM practical, WiSe 2026/27, TUM AIM. Team: Larissa Manalil, Yannic Neu, Emre Kalkan.
Supervisor: Raphael. Does clinical metadata reduce an image classifier's epistemic uncertainty?
Per test image, dEU = EU(image) - EU(image + metadata), on PAD-UFES-20.

This repo only runs the GPU experiments. Data analysis lives in
[eda](https://github.com/ADLM-Multimodal-Uncertainty/eda); slides, report and figures in
[reports-and-presentations](https://github.com/ADLM-Multimodal-Uncertainty/reports-and-presentations).

## Setup

```bash
uv sync --extra dev
export DATA_ROOT=/path/to/data     # $DATA_ROOT/pad-ufes-20/{metadata.csv,images/*.png}
export OUTPUT_ROOT=/path/to/outputs
```

Download PAD-UFES-20 from [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2). Only
the patient-grouped fold assignment in `data/splits/` is committed. Cluster accounts, storage
and SLURM: `docs/cluster_guide.md`.

## Running experiments

```bash
sbatch scripts/slurm/smoke_test.sbatch                    # first job: GPU + environment check
sbatch scripts/slurm/train.sbatch <hydra overrides>       # one run
sbatch scripts/slurm/ensemble_array.sbatch <overrides>    # 5 folds x 5 seeds
```

Every variant (encoder, fusion, split, ensemble, UQ method) is a YAML file in its `configs/`
folder and is selected through Hydra overrides, so experiments never require code changes.

## Layout

```
configs/          one folder per swappable part: data, split, model/{image_encoder,
                  tabular_encoder,fusion}, train, uq, ensemble, experiment
src/              data/, models/, uq/, train.py, evaluate.py, metrics.py
scripts/slurm/    smoke_test, train, ensemble_array
data/splits/      committed fold assignment
tests/
docs/             cluster guide
```

## Conventions

- Splits are grouped by `patient_id`; `lesion_id` is only unique within a patient.
- The 804 images with the clinical block missing are all non-biopsied ACK/SEK/NEV; report
  results with and without missingness indicators and on complete rows only.
- Commit before every run (hash into W&B and Notion). Data, outputs and checkpoints live on
  `/vol/miltank`, never in the repo.
- One branch per feature, PR + one review into `main`, CI green.
