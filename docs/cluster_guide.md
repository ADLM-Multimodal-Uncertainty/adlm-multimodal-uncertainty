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
    export UV_CACHE_DIR=/meta/users/$USER/cache/uv
    # create the environment from the repo's dependency file once it exists (same env for all three)

Put in `~/.bashrc` (that is allowed; it is a dotfile, not data):

    export DATA_ROOT=/vol/miltank/projects/practical_wise2627/<group>/data
    export OUTPUT_ROOT=/vol/miltank/users/$USER/outputs
    export WANDB_DIR=/vol/miltank/users/$USER/outputs/wandb
    export WANDB_API_KEY=<your key>      # or `wandb login` once

## 5. SLURM

Submit: `sbatch scripts/slurm/<job>.sbatch`. Monitor: `squeue -u $USER`, `scontrol show job <id>`,
`sacct -j <id> --format=JobID,State,Elapsed,MaxRSS`. Cancel: `scancel <id>`.

Two QoS:
- `master-queuesave`: at most 2 jobs, never pre-empted. Use for the runs that matter.
- `master`: unlimited jobs, pre-empted by higher priority, not scheduled under full load. Use for sweeps.

Rules
- Request a GPU only if the job uses one. EDA and preprocessing: no `--gres`.
- Fewer than 32 CPUs, well below 128 GB RAM.
- Long-running anything (training, preprocessing, downloads) goes through sbatch, not selene.
- No training in notebooks.
- All data, scripts, outputs, checkpoints on `/vol/miltank/...`.

Job scripts go in `scripts/slurm/`. The first one everyone should run is a short GPU job that
prints `nvidia-smi` and checks that torch sees the GPU.

Job arrays: `#SBATCH --array=0-24%2` runs 25 tasks, 2 at a time (fits master-queuesave).
`fold = SLURM_ARRAY_TASK_ID // 5`, `seed = SLURM_ARRAY_TASK_ID % 5`.

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

- Laptop: same environment, `DATA_ROOT` pointing at a folder with `metadata.csv` and a 100-200
  image subset.
- A one-epoch CPU run on that subset with small images should finish in under a minute; this is
  the correctness check before every cluster run. Run the tests too.
- Commit, push, then `git pull` on selene and submit.

## 8. Checklist before any real run

- [ ] Code committed and pushed; commit hash in the W&B config.
- [ ] Config file under `configs/` (no values edited in the script).
- [ ] Output and checkpoint paths under `/vol/miltank`.
- [ ] Correct QoS, GPU only if needed, time limit realistic.
- [ ] Experiments row created in Notion with status "running".
