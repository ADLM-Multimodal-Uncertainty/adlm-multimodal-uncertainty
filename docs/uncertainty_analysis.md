# Image-only versus image + metadata: analysis plan

Use five independently trained ensemble members per modality with identical patient
splits, seed lists, pretrained initialization, training budget, preprocessing and
checkpoint selection. Fine-tune separate image encoders for the primary comparison.
Do not combine models from different held-out folds as ensemble members.

Let p_m be member probabilities and p_bar their mean. Entropies use natural logs.
EU is an ensemble disagreement proxy, not a proven measurement of all epistemic
uncertainty. Expected member entropy is not a clean measurement of irreducible noise.
Lower uncertainty must be assessed alongside correctness and probability quality.

| Analysis | Calculation | Figure | Required saved data |
|---|---|---|---|
| Classification | Balanced accuracy, macro-F1, per-class recall/precision, accuracy | Normalized confusion matrices | Mean probabilities, labels, class order |
| Probability quality | NLL = mean(-log p_bar[y]); Brier = mean(sum_c (p_bar[c]-onehot(y)[c])²) | Paired estimates and confidence intervals | Mean probabilities and labels |
| Calibration | Confidence versus accuracy; ECE with specified bins, NLL and Brier | Reliability diagrams with bin counts | Mean probabilities and labels; validation logits for fitting temperature |
| Total uncertainty | H(p_bar) | Distributions by mode and class | Mean probabilities |
| Epistemic proxy | EU = H(p_bar)-mean_m H(p_m) | Paired scatter with identity line | All member probabilities |
| Expected member entropy | mean_m H(p_m) | Total/expected/epistemic panels | All member probabilities |
| Primary paired effect | Delta EU = EU(image)-EU(multimodal); mean, median, fraction positive | Delta EU distribution and per-class estimates | Both modes' member probabilities, image and patient IDs |
| Error detection | AUROC and AUPRC for incorrect predictions using EU, entropy and 1-max(p_bar); report error prevalence | Correct/error distributions and PR curves | Probabilities, labels; member probabilities for EU |
| Selective prediction | Retained-set error at 80/90/95% coverage, AURC; specify ties | Risk–coverage curves for each uncertainty score | Same as error detection |
| Correctness transitions | Wrong→correct, correct→wrong, correct→correct, wrong→wrong and their Delta EU | Counts and uncertainty-change distributions | Paired predictions and member probabilities |
| Subgroups | Repeat metrics and Delta EU by clinical missingness, age, sex, site; report counts | Subgroup estimates and intervals | Raw metadata, IDs and outputs |
| Qualitative examples | Corrected/new errors, confident mistakes, largest Delta EU | Image gallery with truth, probabilities and EU | Images, metadata and prediction IDs |
| Ensemble size | Average metrics/EU over subsets of 2–5 members | Metrics versus member count | Member probabilities |

Use paired patient-bootstrap intervals: resample patients, retain all their images,
and use the same resample for both modes. These intervals describe test sampling,
not training randomness. Additional independent ensemble repetitions assess the latter.
Never select calibration parameters or operational rejection thresholds on test data.
Rare-class results require counts and cautious interpretation. Inspect metadata missingness
as a potential shortcut; missingness-associated performance is not causal evidence.

## Artifact contract

Each member saves best.pt, resolved config.yaml and history.csv, plus:

- val_predictions.npz and test_predictions.npz: raw logits, probabilities, labels,
  class order, image IDs and patient IDs.
- split_metadata.csv: exact train/validation/test rows including original metadata,
  lesion identifiers when available and raw missing values.
- metadata_preprocessor.joblib: fitted training-only sklearn preprocessing. Load only
  trusted local artifacts in a compatible environment.
- provenance.json: Git commit/status, Python/packages, metadata SHA256 and data config.
- source_snapshot/: training/evaluation source, configs and dependency manifests,
  including currently untracked implementation files.
- source_changes.patch: tracked changes against HEAD (untracked source files must
  also be preserved or committed before running).
- image_hashes.json: image content SHA256 per ID. Hashes identify data; they do not
  replace preservation of the original images.

Ensemble roots save val_member_predictions.npz and test_member_predictions.npz,
including stacked member logits/probabilities, seeds and shared labels/IDs/classes.
Existing test_predictions.csv and test_metrics.json remain available.
Config plus source preserves evaluation transforms; preserve the source commit and
any uncommitted/untracked source as well as the environment.

All tabular analyses above can be calculated from these outputs without retraining.
Averaged probabilities alone cannot recover member disagreement. Model weights alone
cannot regenerate predictions: retain all member checkpoints, configs, code, preprocessing
and original images/metadata. New-run re-evaluation restores saved split rows and fitted preprocessing; legacy
runs rebuild them from the original dataset. Original images remain required.
Training history cannot be recovered from the final checkpoint. Qualitative image
figures require original images. No analysis figures are generated until runs exist.

## Encoder protocol

Primary experiment: identical pretrained starting weights, independently fine-tuned
copies for image-only and multimodal training. Gradients depend on metadata/fusion,
so learned image encoders may differ. This estimates the overall benefit of metadata
including representation adaptation.

Optional controlled ablation: freeze exactly the same encoder weights for both modes,
including BatchNorm statistics, and train separate heads. This estimates metadata's
benefit conditional on fixed image features. A task-trained shared encoder must be
trained on training patients only. If image-only supervised training supplies the
encoder, explicitly report that asymmetry. A trainable encoder shared between both
heads optimizes a different joint objective and is a third experiment.

Sources:
- https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf
- https://proceedings.mlr.press/v70/guo17a.html
- https://proceedings.mlr.press/v162/huang22e.html
- https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf
