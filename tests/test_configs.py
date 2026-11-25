"""Every config option composes and matches its `_target_`; experiments build the draft model."""

import inspect

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_object
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.models import build_model

from .conftest import REPO

CONFIGS = REPO / "configs"
GROUPS = [
    "analysis",
    "split",
    "model/image_encoder",
    "model/tabular_encoder",
    "model/fusion",
    "train",
    "ensemble",
    "experiment",
]
OVERRIDES = [
    f"{'+' if g == 'analysis' else ''}{g}={p.stem}"
    for g in GROUPS
    for p in sorted((CONFIGS / g).glob("*.yaml"))
]
OFFLINE = ["model.image_encoder.pretrained=false", "model.image_encoder.freeze=false"]


def _compose(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        cfg = compose("config", overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])


def _targets(node):
    if isinstance(node, dict):
        if "_target_" in node:
            yield node
        for value in node.values():
            yield from _targets(value)


@pytest.mark.parametrize("override", OVERRIDES)
def test_option_composes_and_matches_target(override):
    cfg = OmegaConf.to_container(_compose([override]), resolve=True)
    for node in _targets(cfg):
        kwargs = {k: v for k, v in node.items() if not k.startswith("_")}
        inspect.signature(get_object(node["_target_"])).bind_partial(**kwargs)


def _linear_shapes(module: nn.Module) -> list[tuple[int, int]]:
    return [(m.in_features, m.out_features) for m in module.modules() if isinstance(m, nn.Linear)]


@pytest.mark.parametrize(
    ("experiment", "head"),
    [
        ("draft_baseline", [(512, 128), (20, 64), (192, 128), (128, 6)]),
        ("draft_image_only", [(512, 128), (128, 128), (128, 6)]),
    ],
)
def test_experiment_builds_the_draft_architecture(experiment, head):
    cfg = _compose([f"experiment={experiment}", *OFFLINE])
    model = build_model(cfg.model, n_classes=6, n_meta_features=20)
    assert _linear_shapes(model) == head
    logits = model(torch.randn(2, 3, 64, 64), torch.randn(2, 20))
    assert logits.shape == (2, 6)


def test_frozen_backbone_has_no_trainable_parameters_and_stays_in_eval_mode():
    cfg = _compose(["experiment=draft_baseline", "model.image_encoder.freeze=true"])
    model = build_model(cfg.model, n_classes=6, n_meta_features=20).train()
    assert not any(p.requires_grad for p in model.image_encoder.backbone.parameters())
    assert not model.image_encoder.backbone.training
    assert model.image_encoder.projection.training


def test_unknown_experiment_fails_instead_of_running_the_defaults():
    with pytest.raises(Exception, match="experiment/does_not_exist"):
        _compose(["experiment=does_not_exist"])


def test_unfreezing_backbone_enables_training_and_updates_weights():
    cfg = _compose(["experiment=draft_baseline", *OFFLINE])
    model = build_model(cfg.model, n_classes=6, n_meta_features=20).train()
    encoder = model.image_encoder
    assert all(p.requires_grad for p in encoder.backbone.parameters())
    assert encoder.backbone.training
    assert encoder.backbone.bn1.training
    before = encoder.backbone.conv1.weight.detach().clone()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    loss = model(torch.randn(2, 3, 32, 32), torch.randn(2, 20)).square().mean()
    loss.backward()
    assert encoder.backbone.conv1.weight.grad is not None
    optimizer.step()
    assert not torch.equal(before, encoder.backbone.conv1.weight)


def test_default_patient_folds_and_backbone_training():
    cfg = _compose(["experiment=patient_5fold_multimodal"])
    assert cfg.split.n_folds == 2
    assert cfg.model.image_encoder.freeze is False
    assert cfg.model.image_encoder.pretrained is True


@pytest.mark.parametrize(
    ("experiment", "architecture", "uses_metadata"),
    [
        ("eu_baseline_r18_multimodal", "resnet18", True),
        ("eu_baseline_r18_image_only", "resnet18", False),
        ("eu_resnet50_multimodal", "resnet50", True),
        ("eu_resnet50_image_only", "resnet50", False),
    ],
)
def test_eu_experiment_presets_fully_fine_tune_five_member_ensembles(
    experiment, architecture, uses_metadata
):
    cfg = _compose([f"experiment={experiment}"])
    assert cfg.split.n_folds == 5
    assert cfg.model.image_encoder.arch == architecture
    assert cfg.model.image_encoder.freeze is False
    assert cfg.train.optimizer.lr == pytest.approx(1e-4)
    assert list(cfg.ensemble.seeds) == [0, 1, 2, 3, 4]
    assert (cfg.model.get("tabular_encoder") is not None) is uses_metadata


def test_analyze_config_composes_and_matches_target():
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        cfg = compose("analyze")
    node = OmegaConf.to_container(cfg.analysis, resolve=True)
    kwargs = {k: v for k, v in node.items() if not k.startswith("_")}
    inspect.signature(get_object(node["_target_"])).bind_partial(**kwargs)
