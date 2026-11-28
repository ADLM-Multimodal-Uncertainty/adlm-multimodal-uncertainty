# Result analysis specification

Status: implemented by `python -m src.analyze`. The primary comparison is image-only versus image plus clinical metadata. Hyperparameter comparisons use the same formulas with the declared reference and candidate.

## 1. Comparisons and inputs

| Comparison ID | Cohort and class order | Reference | Candidate | Run IDs and test folds | Priority |
| --- | --- | --- | --- | --- | --- |
| primary | PAD-UFES-20, six classes | Image-only | Image + full metadata | `baseline_image_only` and `baseline_multimodal`, folds 0 and 1 | Primary |
| recipe | PAD-UFES-20, six classes | Baseline recipe | Candidate recipe, both as image-only and image + metadata | The four run groups passed to `src.analyze` | When tuning |

Optional comparisons to confirm include reduced metadata, three-class cohorts, and wrong metadata at inference. For each accepted comparison, record the feature list, member seeds, source revision, input artifact paths, and exclusion rules.

- Read member probabilities, image IDs, patient IDs, labels, and class order from each fold's prediction artifacts. Record any additional inputs required by optional analyses.
- Join reference and candidate on fold and image ID. Require unique matches, matching patient IDs and labels, and compatible class order and ensemble sizes. Record paired image and patient counts.
- Average member probabilities within each fold. Pool the held-out images to compute the headline image-weighted metrics. Recompute nonlinear metrics such as macro-F1 on the pooled predictions. Show fold-specific results alongside the pooled estimates.

### Default experiment setup

Apply the same settings to the image-only and multimodal setups.

| Setting | Default |
| --- | --- |
| Patient split | Two patient-disjoint folds. Each fold serves once as test, with validation drawn at the patient level from the other fold. |
| Ensemble | Five members per setup per fold, with the same member seed list for both setups |
| Image backbone | Trainable throughout fine-tuning |
| Training loss | Class-weighted cross-entropy (`train.weighted_loss=true`, `train.class_weight_power=1.0`). Weights are `(N / (K n_k))^power`, then rescaled to mean 1. Power 0 is unweighted. |
| Early stopping | Minimize `val_loss`, patience 8, minimum improvement 0. Keep the best checkpoint, including epoch 1. |
| Maximum training epochs | 30 per member |
| Learning-rate scheduling | AdamW at `1e-4`, weight decay `0.01`. `ReduceLROnPlateau` on the same monitor, factor 0.5, patience 3, minimum learning rate `1e-6`, stepped once per epoch after validation. |
| Augmentation | Training only, after the patient split: resize to 224, horizontal and vertical flips at probability 0.5, rotation of 15 degrees with reflect padding. No crop and no color jitter. |

Record the resolved settings for each run, including any deviations from these defaults.

## 2. Metric definitions and calculation

For image $i$, let $y_i$ be the ground truth, $p_{imc}$ the probability assigned to class $c$ by member $m$, and $q_{ic}$ the mean of $p_{imc}$ across members. The predicted class is the argmax of $q_{ic}$, using the first class in the declared order for ties. Let $N$ be the number of evaluated images and $K$ the number of task classes.

| Metric | Calculation | Reporting |
| --- | --- | --- |
| Accuracy | Correct predictions divided by $N$ | %, higher is better |
| Per-class precision, recall, F1 | $TP/(TP+FP)$, $TP/(TP+FN)$, and $2TP/(2TP+FP+FN)$, using one-vs-rest counts | %, with true class support |
| Balanced accuracy | Mean per-class recall | %, higher is better |
| Macro-F1 | Mean per-class F1 | %, higher is better |
| Negative log-likelihood (NLL) | Mean of $-\log q_{i,y_i}$, flooring the true-class probability at `1e-12` before the natural log. A probability of 1 is not clipped and contributes 0. | Nats, lower is better |
| Multiclass Brier score | Mean across images of $\sum_c (q_{ic}-\mathbf{1}[y_i=c])^2$ | Sum over classes, lower is better |
| Total uncertainty (predictive entropy) | $H(q_i)$, where $H(p)=-\sum_c p_c\log p_c$ and $0\log 0=0$ | Per image and image-weighted mean, nats |
| Aleatoric uncertainty (mean member entropy) | Mean of $H(p_{im})$ across members for each image | Per image and image-weighted mean, nats |
| Epistemic uncertainty proxy (EU) | Total uncertainty minus aleatoric uncertainty for each image | Per image and image-weighted mean, nats |
| Paired total uncertainty reduction | Image-only total uncertainty minus multimodal total uncertainty for each image | Mean and median in nats, relative reduction (%), and share of images with a reduction (%) |
| Paired aleatoric uncertainty reduction | Image-only aleatoric uncertainty minus multimodal aleatoric uncertainty for each image | Mean and median in nats, relative reduction (%), and share of images with a reduction (%) |
| Paired EU reduction (dEU) | Image-only EU minus multimodal EU for each image | Mean and median in nats, relative reduction (%), and share of images with a reduction (%) |
| Setting change dAU, dEU, dTU | For one setup, baseline uncertainty minus candidate-setting uncertainty on each image | Mean and median in nats, relative to the baseline mean (%), and share of images with a positive change (%). Separate rows for image-only and image + metadata |
| Change in the metadata EU reduction, d(dEU) | Candidate-setting dEU minus baseline-setting dEU on each image | Same summaries. Positive means metadata reduces EU by more under the candidate setting |

Per-class precision, recall, and F1 use zero when a denominator is zero. Balanced accuracy and macro-F1 average only over classes that have at least one true image in the evaluated rows; a class absent from a fold is omitted rather than scored as zero. The confusion matrix still uses the full declared class order. Report paired classification changes as candidate minus reference in percentage points.

For each uncertainty quantity, relative reduction is $100$ times the mean paired reduction divided by the mean reference uncertainty over the same paired images. Report it as unavailable when the denominator is zero. The share of images with a reduction is $100$ times the number of strictly positive paired differences divided by the number of pairs. Positive differences indicate lower candidate uncertainty. For a recipe comparison, dAU, dEU, and dTU use the baseline setting as the reference and the candidate setting as the candidate, separately for image-only and image + metadata. d(dEU) is the candidate metadata EU reduction minus the baseline metadata EU reduction.

## 3. Tables

| ID | Priority | Rows | Required columns |
| --- | --- | --- | --- |
| T1 Cohort and pairing | Core | Comparison × fold, plus pooled | Reference and candidate run IDs, ensemble size, paired image and patient counts, class counts |
| T2 Performance and uncertainty | Core | Condition × fold, plus pooled | Accuracy, balanced accuracy, macro-F1, NLL, Brier, mean total uncertainty, mean aleatoric uncertainty, mean EU |
| T3 Paired effects | Core | Comparison | Paired counts, change in balanced accuracy and macro-F1, and for total, aleatoric, and epistemic uncertainty: mean and median reduction in nats, relative reduction (%), and share of images with a reduction (%). `paired_effects.csv` keeps these as wide columns. `REPORT.md` prints one narrow block per comparison: a four-column classification row, then a five-column uncertainty table with rows Total, Aleatoric, and Epistemic. |
| T3b Setting uncertainty change | Core for a recipe comparison | One row per between-setting quantity | `setting_uncertainty_change.csv`: dAU, dEU, and dTU for image-only and for image + metadata, then d(dEU). Mean, median, relative change, share positive, and the patient-cluster interval. A single image-versus-metadata report does not write this table. |
| T4 Class breakdown | Supporting | Condition × true class | Support, precision, recall, F1. Include per-class dEU and paired counts in comparison rows if selected. |

## 4. Figures

| ID | Priority | Figure | Axes, units, and reference marks |
| --- | --- | --- | --- |
| F1 | Core | Paired EU scatter | Image-only EU on x, multimodal EU on y, both in nats. Use common limits and a diagonal identity line. Each matched image contributes one point. |
| F2 | Core | Distribution of paired dEU | Four columns in order: both correct, reference wrong and candidate correct (fixed), reference correct and candidate wrong (broken), both wrong. Plot dEU in nats on x and image count on y. Share bin edges and axis limits, mark zero, and label each panel with its image count. |
| F2 candles | Core | dEU candles for the same four groups | Same group order as F2, one box per group. dEU in nats on y, a zero line, and the image count in the category label. The box is the quartile range, whiskers are 1.5 IQR, fliers are images outside the whiskers, and a diamond marks the mean. |
| F3 | Core | Confusion matrices | True class on rows, predicted class on columns, with shared class order and color scale. Normalize by true-class support and annotate raw counts. |
| F4 | Supporting | Paired effect estimates | Comparison or class on y, mean dEU in nats on x, with a zero line. |
| F7 | Core for a recipe comparison | Uncertainty change between settings | dAU, dEU, and dTU for each setup, then d(dEU), in nats on x. A zero line. Bars are 95% patient-cluster intervals. |
| F5 | Core | Total versus aleatoric uncertainty across experiments | Mean total uncertainty on x and mean aleatoric uncertainty on y, both in nats. Add one point per setup per completed experiment using pooled held-out predictions. Identify the experiment and setup for each point and regenerate the scatter after every completed experiment. |
| F6 | Core | Validation curves | Side-by-side panels for validation loss versus epoch and validation accuracy (%) versus epoch. Plot both setups in each panel. Use member histories matched by fold and seed. Show individual traces and a setup mean at each observed epoch, with contributing member counts where early stopping changes coverage. |

## 5. Deliverables and acceptance checks

### Output layout

```text
outputs/result-analysis/
  <experiment_id>/
    manifest.yaml
    REPORT.md
    tables/
      cohort_and_pairing.csv
      performance_and_uncertainty.csv
      paired_effects.csv
      class_breakdown.csv
      setting_uncertainty_change.csv
    data/
      paired_images.csv
      validation_history.csv
    figures/
      paired_eu_scatter.png
      paired_deu_distribution.png
      paired_deu_candles.png
      confusion_matrices.png
      paired_effect_estimates.png
      validation_curves.png
      setting_uncertainty_change.png
  summary/
    experiment_uncertainty.csv
    total_vs_aleatoric_uncertainty.png
```

- Use a distinct experiment ID when inputs or settings change. A rerun with the same inputs and settings regenerates the same paths. For per-comparison figures within an experiment, append the comparison ID to the filename before the extension (`primary` for a single image-only versus metadata report; `baseline_modality`, `candidate_modality`, `image_only_change`, and `multimodal_change` for a recipe comparison).
- `eval_split=val` compares validation predictions while tuning. `eval_split=test` is the final report and is the only mode that updates `summary/`.
- A single-setup report sets `reference` and `candidate`. A recipe report sets `baseline_image`, `baseline_multimodal`, `candidate_image`, `candidate_multimodal`, and `candidate_label`. Do not set both forms at once. `setting_uncertainty_change.csv` and its figure are written only for a recipe report.
- Record the analysis source revision, input artifact paths, resolved experiment settings, class order, and metric conventions in `manifest.yaml`. Link the tables and figures from `REPORT.md`.
- Write UTF-8 CSVs with explicit units in column names where applicable. Key paired rows by comparison, fold, and image ID. Key validation rows by setup, fold, member seed, and epoch.
- Update the summary CSV by experiment ID and setup after every configured test fold is present and `eval_split=test`. Regenerate F5 from that CSV so reruns replace the previous points for the same experiment.

### Acceptance checks

- Verify that paired rows, class counts, and metric values agree across tables and figures. Check that F2 panel counts and F2 candle labels sum to the paired image count and that F6 uses only epochs present in the member histories.
- Check formulas on a small hand-calculated example. Identical member distributions yield zero disagreement, and identical conditions yield zero paired effects.

The generator is `src.analyze` with `configs/analyze.yaml`. It reads `{val,test}_member_predictions.npz` and member `history.csv` files produced by `src.train`.
