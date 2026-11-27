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

`selene` is a login node. Use it to `git pull`, `sbatch`, `squeue`, `sacct`, `scancel`, and read logs. Do not train on it, and do not download or install there. `uv python install` and `uv sync` download large files, so they belong in `scripts/slurm/sync_env.sbatch`, not in an SSH session.

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

Data layout the training script accepts. `DATA_ROOT` may point at a directory that contains
either `pad_ufes_20/` or `pad-ufes-20/`; both names are checked, and `metadata.csv` must be
directly inside the chosen directory.

    $DATA_ROOT/pad_ufes_20/
      metadata.csv
      images/          # all images from the Mendeley download, flat or imgs_part_*/

## 4. Environment

On a laptop, clone and commit as usual. On selene, only update the checkout and submit the install job:

    cd /vol/miltank/users/$USER/code/adlm-multimodal-uncertainty
    git pull
    mkdir -p /vol/miltank/users/$USER/outputs/slurm
    sbatch scripts/slurm/sync_env.sbatch

That CPU job installs CPython 3.12 under `/vol/miltank/users/$USER/python`, builds `.venv` against that exact file, and runs `uv sync --extra dev`. The download happens on a compute node. Submit it again after a lockfile change. Do not run `uv python install`, `uv venv`, or `uv sync` in the SSH session.

`/usr/bin/python3` is not one program. It is Python 3.12 on selene and Python 3.10 on some compute nodes, including ceres. A virtualenv linked to that path imports Hydra on the login node and then fails on those nodes with `ModuleNotFoundError: No module named 'hydra'`. `sync_env.sbatch` refuses an interpreter outside `/vol/miltank`.

Put in `~/.bashrc` (that is allowed; it is a dotfile, not data):

    export DATA_ROOT=/vol/miltank/projects/practical_wise2627/<group>/data
    export OUTPUT_ROOT=/vol/miltank/users/$USER/outputs
    export UV_CACHE_DIR=/meta/users/$USER/.cache/uv
    export TORCH_HOME=/vol/miltank/users/$USER/checkpoints/torch

Before the first job, confirm home contains only dotfiles and symlinks into `/meta`:

    ls -la ~
    mkdir -p /vol/miltank/users/$USER/outputs/slurm
    du -sh /vol/miltank/users/$USER

## 5. SLURM

Submit: `sbatch scripts/slurm/train.sbatch`. Monitor: `squeue -u $USER`, `scontrol show job <id>`,
`sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`. Cancel: `scancel <id>`.

Two QoS:
- `master-queuesave`: at most two jobs at a time, including array tasks. They are not pre-empted.
  Use this for training. Two arrays submitted with `%2` can start four tasks and break the cap;
  submit each paired setup with `--array=0-1%1` so the two setups together use exactly two jobs.
- `master`: unlimited jobs, pre-empted by higher priority, and not scheduled when the cluster is full.
  Do not use it for these runs.

Rules
- Request a GPU only for training. Analysis and plotting stay on the laptop, or on a CPU job if they
  are ever submitted. Do not request `--gres=gpu` for `src.analyze`.
- Fewer than 32 CPUs (6 is enough), well below 128 GB RAM (24 GB is enough).
- Long-running anything (training, preprocessing, downloads) goes through sbatch, not selene.
- No training in notebooks.
- All data, scripts, outputs, checkpoints, and SLURM logs live on `/vol/miltank/...`.
  Caches live on `/meta/users/$USER/...`. The home directory stays empty.
- A single `sbatch` command is one Hydra run, not a multirun. Changing a hyperparameter
  without `name=...` overwrites `outputs/runs/<experiment>/`. Give every trial its own name.
- Submit only a commit that has been pushed and pulled. The job prints that commit hash.

Templates in `scripts/slurm/`:
- `sync_env.sbatch`: one CPU job, no GPU. Installs the shared Python 3.12 and the project packages. Run this instead of `uv sync` on selene.
- `smoke_test.sbatch`: one short GPU job that prints `nvidia-smi`, imports torch, and runs pytest. Submit it only after `sync_env.sbatch` has finished. The job uses `.venv/bin/python`.
- `train.sbatch`: one training command. Pass Hydra overrides, including `experiment=` and `fold=`.
- `fold_ensemble_array.sbatch`: one array task per fold (`--array=0-1%1`). Each task trains all five
  ensemble members of that fold and writes `members/seed_<s>/` plus the aggregated predictions.
  Logs go to `/vol/miltank/users/$USER/outputs/slurm/`.

Paired baseline, which stays inside the two-job cap:

    sbatch --array=0-1%1 scripts/slurm/fold_ensemble_array.sbatch experiment=baseline_image_only
    sbatch --array=0-1%1 scripts/slurm/fold_ensemble_array.sbatch experiment=baseline_multimodal

The dataset is staged to node-local `/tmp` for the job. The saved `config.yaml` still records
the `/vol/miltank` data path. Set `STAGE_TO_LOCAL=0` to read images from the network filesystem.

Copy a finished run group back to a laptop, without checkpoint files:

    scripts/local/sync_cluster_runs.sh baseline_image_only baseline_multimodal

`CLUSTER_USER` and `CLUSTER_HOST` (default `selene`) select the remote account. Set
`INCLUDE_CHECKPOINTS=1` to copy `*.pt` as well.

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
- Commit, push, then on selene `git pull`. If `uv.lock` changed, submit `scripts/slurm/sync_env.sbatch` and wait for it. Do not install on the login node.
- Do not start another `master-queuesave` job while two are already pending or running. `sync_env.sbatch` counts as one of those two.

## 8. Checklist before any real run

- [ ] Code committed and pushed; `git pull` done on selene. Packages were installed by `sbatch scripts/slurm/sync_env.sbatch`, not by a download in the SSH session.
- [ ] `.venv/bin/python` points at `/vol/miltank/users/$USER/python/...`, not at `/usr/bin/python3`.
- [ ] The job log will print that commit hash.
- [ ] `ls -la ~` shows only dotfiles and symlinks, no data.
- [ ] `du -sh /vol/miltank/users/$USER` is comfortably under the 500 GB group limit.
- [ ] Config file under `configs/` (no values edited in the script). A changed setting has its own `name=`.
- [ ] Output and checkpoint paths under `/vol/miltank`.
- [ ] `master-queuesave`, at most two jobs, GPU only for training, time limit realistic.
- [ ] `mkdir -p /vol/miltank/users/$USER/outputs/slurm` has been run once.
