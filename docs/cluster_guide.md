# Cluster guide (chair cluster, SLURM)

Source: kickoff deck slides 7-18. Ask in the `#server-cluster` Slack channel when something breaks.

## 1. Accounts and VPN

1. Activate the CIT account: https://ucentral.in.tum.de/cgi-bin/activate.cgi and change the password.
2. VPN: https://vpn.rbg.tum.de with CIT credentials. Download the config for the chair (`aim`) and add
   these two lines to the `.ovpn` file:

       route 131.159.110.0 255.255.255.0   # fmi (servers)
       route 131.159.128.0 255.255.255.0   # galileo (workstations)

   Linux: `sudo openvpn vpn-il31-2.4-linux.ovpn`. macOS: Tunnelblick or OpenVPN Connect with the same file.
   Windows: use the standard `.ovpn`, not the "wintun" variant, otherwise the routes do not work.

## 2. SSH

    ssh <cit-username>@131.159.110.9      # selene
    ssh <cit-username>@selene             # if DNS resolves through the VPN

Recommended `~/.ssh/config` on the laptop:

    Host selene
        HostName 131.159.110.9
        User <cit-username>
        ServerAliveInterval 60

`selene` is a login node: submit jobs, create environments, move files. Never train on it.

## 3. Storage

| Path | Use |
|------|-----|
| `~` / `/u/home/<user>` | must stay empty |
| `/vol/miltank/users/<user>/` | your code, outputs, checkpoints (group limit 500 GB, check with `du -sh`) |
| `/vol/miltank/projects/practical_wise2627/<group>/` | shared data and shared checkpoints |
| `/meta/users/<user>/` | fast metadata storage: conda, uv cache, vscode-server |
| `/vol/cuttlefish/` | fast scratch, small, slower than miltank under full load |

First login:

    mkdir -p /meta/users/$USER/{conda,cache,vscode-server}
    ln -s /meta/users/$USER/conda ~/.conda
    ln -s /meta/users/$USER/cache ~/.cache
    ln -s /meta/users/$USER/vscode-server ~/.vscode-server
    mkdir -p /vol/miltank/users/$USER/{code,outputs,checkpoints}

Data layout the code expects (set `DATA_ROOT`):

    $DATA_ROOT/
      pad-ufes-20/
        metadata.csv
        images/          # all .png from the Mendeley download, flat
      isic2019/          # later

## 4. Environment

    ml python/uv
    cd /vol/miltank/users/$USER/code
    git clone <repo-url> adlm-multimodal-uncertainty && cd adlm-multimodal-uncertainty
    export UV_CACHE_DIR=/meta/users/$USER/.cache/uv
    uv sync                       # creates .venv from uv.lock; same env for all three of us

Put in `~/.bashrc` (that is allowed; it is a dotfile, not data):

```bash
export REPO_DIR=/vol/miltank/users/$USER/code/adlm-multimodal-uncertainty
export DATA_ROOT=/vol/miltank/projects/practical_wise2627/<group>/data
# Or override DATA_DIR directly if using your own copy under /vol/miltank/users/$USER/data/pad_ufes_20:
# export DATA_DIR=/vol/miltank/users/$USER/data/pad_ufes_20
export OUTPUT_ROOT=/vol/miltank/users/$USER/outputs
export UV_CACHE_DIR=/meta/users/$USER/.cache/uv
export XDG_CACHE_HOME=/meta/users/$USER/cache
export TORCH_HOME=/meta/users/$USER/cache/torch
export MPLCONFIGDIR=/meta/users/$USER/cache/matplotlib
export WANDB_DIR=/vol/miltank/users/$USER/outputs/wandb
export WANDB_API_KEY=<your key>      # or `wandb login` once
```

Also create the SLURM log directory once before submitting any job:

```bash
mkdir -p /vol/miltank/users/$USER/outputs/slurm
ls -la ~   # verify home directory contains only dotfiles and symlinks to /meta
```

If PAD-UFES-20 is not yet in `/vol/miltank/projects/practical_wise2627/<group>/data`, upload it
from your laptop directly into `/vol/miltank` (never `~`):

```bash
# Run on your laptop (with VPN connected):
ssh selene "mkdir -p /vol/miltank/users/<cit-username>/data/pad_ufes_20"
rsync -avh --progress data/pad_ufes_20/ \
  selene:/vol/miltank/users/<cit-username>/data/pad_ufes_20/
```

## 5. SLURM

Submit: `sbatch scripts/slurm/train.sbatch`. Monitor: `squeue -u $USER`, `scontrol show job <id>`,
`sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`. Cancel: `scancel <id>`.

Two QoS:
- `master-queuesave`: at most 2 jobs, never pre-empted. Use for the runs that matter.
- `master`: unlimited jobs, pre-empted by higher priority, not scheduled under full load. Use for sweeps.

Rules
- Request a GPU only if the job uses one (never request `--gres=gpu` for analysis/plotting).
- Fewer than 32 CPUs (6 is enough for ResNet18/50), well below 128 GB RAM (24 GB is enough).
- Long-running anything (training, preprocessing, downloads, analysis) goes through `sbatch`, not `selene`.
- No training in notebooks.
- All data, scripts, outputs, checkpoints and SLURM logs on `/vol/miltank/...`; caches on `/meta/users/$USER/...`.
- Check storage regularly: `du -sh /vol/miltank/users/$USER` (500 GB group limit).

Templates in `scripts/slurm/`:
- `smoke_test.sbatch`: 15-minute GPU job (`master-queuesave`) verifying `nvidia-smi`, CUDA PyTorch,
  `/meta` caches, `pytest -q`, and a 1-epoch CUDA smoke run on `/vol/miltank` data. Run this first.
- `train.sbatch`: single training run on 1 GPU, config via Hydra overrides.
- `fold_ensemble_array.sbatch`: fold-level job array (`--array=0-4%2`) for `DeepEnsemble` presets
  (`eu_*`, `deep_ensemble*`). Each task runs one complete 5-member ensemble fold; `%2` respects
  `master-queuesave`'s 2-job limit. Override with `--array=0-1%2` for 2-fold screening or
  `--array=2-4%2` to expand remaining folds.
- `ensemble_array.sbatch`: 25-task fold x seed array for non-ensemble (`ensemble=null`) presets only.
- `analyze.sbatch`: **CPU-only** job (no GPU requested) running `python -m src.analyze` on completed
  runs under `$OUTPUT_ROOT/runs` to write all tables, EU metrics, and plots to `$OUTPUT_ROOT/analysis`.

## 5b. Running the Paired EU Experiments on the Cluster

All four `eu_*` presets keep the backbone fully trainable (`freeze=false`), use ImageNet weights,
AdamW `lr=1e-4`, five ensemble seeds `[0,1,2,3,4]`, and the committed 5-fold patient split.
Every completed fold saves `best.pt`, `history.csv`, `test_member_predictions.npz` (probabilities
and raw logits), and `val_member_predictions.npz` (validation logits and probabilities), so all
downstream calibration, EU measures, and plots run on CPU without retraining.

### Step 0: Commit, push, pull on `selene`, and run the smoke test

```bash
# On laptop:
git push -u origin feature/paired-eu-experiments

# On selene:
cd /vol/miltank/users/$USER/code/adlm-multimodal-uncertainty
git fetch origin && git checkout feature/paired-eu-experiments && git pull
uv sync --extra dev
mkdir -p /vol/miltank/users/$USER/outputs/slurm
sbatch scripts/slurm/smoke_test.sbatch
squeue -u $USER
```

Wait until `smoke_test` finishes with `COMPLETED` (`sacct -j <jobid>`) and check its log in
`/vol/miltank/users/$USER/outputs/slurm/`.

### Stage 1: Baseline (ResNet18, all 5 folds)

```bash
sbatch scripts/slurm/fold_ensemble_array.sbatch experiment=eu_baseline_r18_image_only
sbatch scripts/slurm/fold_ensemble_array.sbatch experiment=eu_baseline_r18_multimodal
```

### Stage 2: Experiment 6 (ResNet50, screen on folds 0-1 first)

```bash
sbatch --array=0-1%2 scripts/slurm/fold_ensemble_array.sbatch experiment=eu_resnet50_image_only
sbatch --array=0-1%2 scripts/slurm/fold_ensemble_array.sbatch experiment=eu_resnet50_multimodal
```

Run the CPU-only analysis to check the Experiment 6 decision rule in `summary_tables.md`:

```bash
sbatch scripts/slurm/analyze.sbatch
```

Expand ResNet50 to folds 2-4 **only if** `summary_tables.md` flags a dEU sign/CI shift or a
balanced-accuracy change > 2 percentage points:

```bash
sbatch --array=2-4%2 scripts/slurm/fold_ensemble_array.sbatch experiment=eu_resnet50_image_only
sbatch --array=2-4%2 scripts/slurm/fold_ensemble_array.sbatch experiment=eu_resnet50_multimodal
```

### Stage 3: Experiment 4 (Positive control: 25% and 50% training patients, reusing 100%)

Use the chosen encoder from Stage 2 (`eu_baseline_r18_*` if keeping ResNet18, or `eu_resnet50_*`
if switching to ResNet50). Always pass a distinct `name=...` so fractions do not overwrite the
100% baseline runs:

```bash
# Example with ResNet18 (swap experiment=... and r18 -> r50 if ResNet50 won Stage 2):
sbatch --array=0-1%2 scripts/slurm/fold_ensemble_array.sbatch \
  experiment=eu_baseline_r18_image_only data.train_fraction=0.25 name=eu_size_r18_f025_image_only
sbatch --array=0-1%2 scripts/slurm/fold_ensemble_array.sbatch \
  experiment=eu_baseline_r18_multimodal data.train_fraction=0.25 name=eu_size_r18_f025_multimodal
sbatch --array=0-1%2 scripts/slurm/fold_ensemble_array.sbatch \
  experiment=eu_baseline_r18_image_only data.train_fraction=0.5 name=eu_size_r18_f050_image_only
sbatch --array=0-1%2 scripts/slurm/fold_ensemble_array.sbatch \
  experiment=eu_baseline_r18_multimodal data.train_fraction=0.5 name=eu_size_r18_f050_multimodal
```

Run the CPU-only analysis again (`sbatch scripts/slurm/analyze.sbatch`) and check
`positive_control_summary.csv` and `summary_tables.md`. If EU does not decrease monotonically
or the 25%-vs-100% patient bootstrap CI includes 0, expand folds 2-4 with `--array=2-4%2` using
the exact same `name=...` overrides, then run `sbatch scripts/slurm/analyze.sbatch` one final time.

### Pull lightweight analysis tables and plots back to your laptop

Do **not** transfer large `.pt` checkpoints back to your laptop; sync only `outputs/analysis/`:

```bash
# Run on your laptop (with VPN connected):
mkdir -p outputs/cluster_analysis
rsync -avh --progress \
  selene:/vol/miltank/users/<cit-username>/outputs/analysis/ \
  outputs/cluster_analysis/
```

## 6. Interactive debugging (shs: VS Code / Jupyter inside a SLURM job)

Tool: https://gitlab.lrz.de/christian.beischl/shs

1. On selene: clone shs into `/vol/miltank/users/$USER/shs`, run its install script, follow the printed
   instructions.
2. Edit `shs/user.config`: set `VSC_KEY_PATH` to a public key (`ssh-keygen` if none) and add
   `VSC_OUTPUT=/vol/miltank/users/$USER/outputs/shs` to capture the slurm log.
3. Laptop `~/.ssh/config`:

       Host aim-tunnel
           ProxyCommand ssh selene "bash /vol/miltank/users/<user>/shs/svscode.sh"
           StrictHostKeyChecking no
           ConnectTimeout 60
           User <user>

4. VS Code: Remote-SSH -> `aim-tunnel`. If it fails, run the ProxyCommand manually and read the file in
   `VSC_OUTPUT`.

Use this for stepping through a batch, not for training.

## 7. Local development loop

- Laptop: `uv sync`, `DATA_ROOT` pointing at a folder with `metadata.csv` and a 100-200 image subset.
- A CPU run of `python -m src.train` on a tiny subset (small images, one epoch) must finish in
  under a minute; this is the correctness check before every cluster run.
- Tests: `uv run pytest -q`.
- Commit, push, then `git pull` on selene and submit.

## 8. Checklist before any real run

- [ ] Code committed and pushed; commit hash in the W&B config.
- [ ] Config file under `configs/` (no values edited in the script).
- [ ] Output and checkpoint paths under `/vol/miltank`.
- [ ] Correct QoS, GPU only if needed, time limit realistic.
- [ ] Experiments row created in Notion with status "running".
