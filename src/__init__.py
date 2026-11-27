"""Epistemic uncertainty in multimodal skin lesion classification (ADLM WiSe 26/27)."""

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]

# Keys that identify where a run executes, not which model was trained. A multirun
# that only changes these must keep the readable run-group name.
SWEEP_TAG_EXCLUDED = {
    "experiment",
    "name",
    "tag",
    "fold",
    "seed",
    "output_root",
    "device",
    "data.root",
    "data.num_workers",
    "data.subset",
    "data.image_size",
    "model.image_encoder.pretrained",
    "model.image_encoder.freeze",
    "split.csv",
}


def meaningful_overrides(overrides: list[str]) -> list[str]:
    """Task overrides that change the trained model and therefore the run directory."""
    kept = []
    for item in overrides:
        key = item.lstrip("+~").split("=", 1)[0]
        if key in SWEEP_TAG_EXCLUDED:
            continue
        kept.append(item.replace("/", "-"))
    return kept


def sweep_tag() -> str:
    """Suffix that keeps Hydra multirun jobs from writing over the same directory.

    Returns an empty string for a normal run, and for a multirun whose overrides are
    only infrastructure (fold, paths, smoke-test image size, and so on).
    """
    try:
        hydra_cfg = HydraConfig.get()
        mode = hydra_cfg.mode
    except (ValueError, AttributeError):
        return ""
    if mode != RunMode.MULTIRUN:
        return ""
    kept = meaningful_overrides(list(hydra_cfg.overrides.task))
    if not kept:
        return ""
    return "_" + "__".join(kept)


if not OmegaConf.has_resolver("sweep_tag"):
    OmegaConf.register_new_resolver("sweep_tag", sweep_tag, use_cache=False)
