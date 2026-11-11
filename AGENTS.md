# Instructions for coding agents

## Feature branches are required

- Develop every new feature on a dedicated branch. Never implement or commit new features
  directly on `main` or `master`.
- Check the current branch before editing. If it is `main` or `master`, create and switch to
  a descriptive feature branch first (for example, `git switch -c feature/ensemble-uq`).
- Merge through a pull request with at least one review and passing CI, as required by the
  project conventions.

## Hydra is required

Hydra configuration is a design requirement. Read `README.md` and
`docs/design_decisions.md` before changing experiment behavior.

- Keep experiment choices and tunable parameters in `configs/`. Use existing config groups,
  `_target_` instantiation, and component contracts in the module docstrings.
- Add experiment presets in `configs/experiment/`; reuse the common training and evaluation
  pipeline. Do not add parallel argument parsers or experiment-specific training scripts.
- Expose new options in YAML instead of hard-coding experiment settings in Python. Use Hydra
  overrides (`key=value`) and multiruns for variations and sweeps.
- Preserve resolved per-run `config.yaml` output and evaluation from that saved configuration.
  Set a distinct `name=...` when changing settings that otherwise share a run directory.
- Check composition with `uv run python -m src.train experiment=<name> --cfg job`.
  For configurable component changes, run `uv run pytest tests/test_configs.py` and relevant
  pipeline tests. Update documentation with the new options and commands.

## Experiment scope

The required comparison is image-only versus image plus metadata. Metadata-only mode is
not required.
