"""A small synthetic PAD-UFES-20 lookalike, so the pipeline is testable without the dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from src import CLASSES

REPO = Path(__file__).resolve().parents[1]
DATA_CFG = OmegaConf.load(REPO / "configs" / "data" / "pad_ufes20.yaml")
PATIENTS_PER_CLASS = 20


def synthetic_metadata(seed: int = 0) -> pd.DataFrame:
    """One or two images per patient, every patient with a single diagnosis, some missing values."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, diagnostic in enumerate(CLASSES):
        for p in range(PATIENTS_PER_CLASS):
            patient = f"PAT_{k}{p:02d}"
            for image in range(1 + p % 2):
                row = {
                    "patient_id": patient,
                    "lesion_id": p,
                    "img_id": f"{patient}_{p}_{image}.png",
                    "diagnostic": diagnostic,
                    "biopsed": True,
                }
                for col in DATA_CFG.numeric_features:
                    row[col] = np.nan if rng.random() < 0.2 else float(rng.integers(1, 90))
                for col in DATA_CFG.categorical_features:
                    row[col] = rng.choice(["True", "False", None])
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """`<root>/metadata.csv` and `<root>/images/imgs_part_1/*.png` in the draft's layout."""
    root = tmp_path / "pad_ufes_20"
    image_dir = root / "images" / "imgs_part_1"
    image_dir.mkdir(parents=True)
    df = synthetic_metadata()
    df.to_csv(root / "metadata.csv", index=False)
    rng = np.random.default_rng(1)
    for img_id in df["img_id"]:
        pixels = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(image_dir / img_id)
    return root
