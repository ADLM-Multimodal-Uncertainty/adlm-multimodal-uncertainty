"""Persist split membership, preprocessing and provenance before training."""
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import torch
from omegaconf import OmegaConf


def save_run_artifacts(run_dir, cfg, data, model):
    source_root = Path(__file__).resolve().parents[1]

    def git(*args):
        result = subprocess.run(['git', *args], cwd=source_root, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else None

    snapshot = run_dir / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for directory in ('src', 'configs', 'docs', 'scripts'):
        shutil.copytree(source_root / directory, snapshot / directory,
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    for filename in ('pyproject.toml', 'uv.lock', 'README.md', 'AGENTS.md'):
        if (source_root / filename).exists():
            shutil.copy2(source_root / filename, snapshot / filename)
    parts = []
    for name in ('train', 'val', 'test'):
        frame = getattr(data, name).df.copy()
        frame['partition'] = name
        parts.append(frame)
    # Full source columns preserve lesion IDs and raw missingness for subgroup analyses.
    rows = pd.concat(parts, ignore_index=True)
    rows.to_csv(run_dir / 'split_metadata.csv', index=False)
    missingness = rows[['patient_id', 'img_id', 'partition']].copy()
    for column in [*cfg.data.numeric_features, *cfg.data.categorical_features]:
        values = rows[column]
        if column in cfg.data.numeric_features:
            values = pd.to_numeric(values, errors='coerce')
        missingness[f'missing_{column}'] = values.isna()
    missingness.to_csv(run_dir / 'metadata_missingness.csv', index=False)
    joblib.dump(data.preprocessor, run_dir / 'metadata_preprocessor.joblib')
    metadata_path = Path(cfg.data.metadata_csv)
    image_hashes = {}
    for name in ('train', 'val', 'test'):
        dataset = getattr(data, name)
        for img_id, path in zip(dataset.df.img_id, dataset.paths, strict=True):
            with Path(path).open('rb') as image_file:
                image_hashes[str(img_id)] = hashlib.file_digest(image_file, 'sha256').hexdigest()
    (run_dir / 'image_hashes.json').write_text(json.dumps(image_hashes, indent=2))
    provenance = {
        'created_utc': datetime.now(UTC).isoformat(),
        'git_commit': git('rev-parse', 'HEAD'),
        'git_status': git('status', '--porcelain'),
        'python': platform.python_version(),
        'packages': sorted(f'{d.metadata["Name"]}=={d.version}'
                           for d in importlib.metadata.distributions()),
        'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'data_config': OmegaConf.to_container(cfg.data, resolve=True),
        'seed': int(cfg.seed),
        'device': str(next(model.parameters()).device),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'image_encoder_initial_sha256': hashlib.sha256(b''.join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for value in model.image_encoder.state_dict().values()
        )).hexdigest(),
    }
    (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    (run_dir / 'source_changes.patch').write_text(git('diff', 'HEAD', '--binary') or '')
