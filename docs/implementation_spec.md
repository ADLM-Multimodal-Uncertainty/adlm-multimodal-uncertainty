# Result analysis specification

Status: draft template. The primary comparison is image-only versus image plus clinical metadata. Items marked **TBD** require a decision before implementation.

## 1. Comparisons and inputs

| Comparison ID | Cohort and class order | Reference | Candidate | Run IDs and test folds | Priority |
| --- | --- | --- | --- | --- | --- |
| primary | PAD-UFES-20, six classes | Image-only | Image + full metadata | TBD | Primary |
| TBD | TBD | TBD | TBD | TBD | Optional |

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
| Training loss | Class-weighted cross-entropy (`train.weighted_loss=true`), with inverse-frequency weights calculated from the training partition |
| Early stopping | Enabled. Validation monitor, patience, and minimum improvement are TBD. Keep the best validation checkpoint. |
| Maximum training epochs | 30 per member |
| Learning-rate scheduling | Enabled. Scheduler, parameters, and step timing are TBD. |

Record the resolved settings for each run, including any deviations from these defaults.

## 2. Metric definitions and calculation

For image $i$, let $y_i$ be the ground truth, $p_{imc}$ the probability assigned to class $c$ by member $m$, and $q_{ic}$ the mean of $p_{imc}$ across members. The predicted class is the argmax of $q_{ic}$, using the first class in the declared order for ties. Let $N$ be the number of evaluated images and $K$ the number of task classes.

| Metric | Calculation | Reporting |
| --- | --- | --- |
| Accuracy | Correct predictions divided by $N$ | %, higher is better |
| Per-class precision, recall, F1 | $TP/(TP+FP)$, $TP/(TP+FN)$, and $2TP/(2TP+FP+FN)$, using one-vs-rest counts | %, with true class support |
| Balanced accuracy | Mean per-class recall | %, higher is better |
| Macro-F1 | Mean per-class F1 | %, higher is better |
| Negative log-likelihood (NLL) | Mean of $-\log q_{i,y_i}$, flooring the true-class probability at **TBD** before taking the natural log | Nats, lower is better |
| Multiclass Brier score | Mean across images of $\sum_c (q_{ic}-\mathbf{1}[y_i=c])^2$ | Sum over classes, lower is better |
| Total uncertainty (predictive entropy) | $H(q_i)$, where $H(p)=-\sum_c p_c\log p_c$ and $0\log 0=0$ | Per image and image-weighted mean, nats |
| Aleatoric uncertainty (mean member entropy) | Mean of $H(p_{im})$ across members for each image | Per image and image-weighted mean, nats |
| Epistemic uncertainty proxy (EU) | Total uncertainty minus aleatoric uncertainty for each image | Per image and image-weighted mean, nats |
| Paired total uncertainty reduction | Image-only total uncertainty minus multimodal total uncertainty for each image | Mean and median in nats, relative reduction (%), and share of images with a reduction (%) |
| Paired aleatoric uncertainty reduction | Image-only aleatoric uncertainty minus multimodal aleatoric uncertainty for each image | Mean and median in nats, relative reduction (%), and share of images with a reduction (%) |
| Paired EU reduction (dEU) | Image-only EU minus multimodal EU for each image | Mean and median in nats, relative reduction (%), and share of images with a reduction (%) |

Define zero-denominator handling and the class inclusion rule for macro averages when an evaluated fold lacks a class (**TBD**). Report paired classification changes as candidate minus reference in percentage points.

For each uncertainty quantity, relative reduction is $100$ times the mean paired reduction divided by the mean image-only uncertainty over the same paired images. Report it as unavailable when the denominator is zero. The share of images with a reduction is $100$ times the number of strictly positive paired differences divided by the number of pairs. Positive differences indicate lower multimodal uncertainty. For other comparisons, use the declared reference and candidate in the same formulas.

## 3. Tables

| ID | Priority | Rows | Required columns |
| --- | --- | --- | --- |
| T1 Cohort and pairing | Core | Comparison × fold, plus pooled | Reference and candidate run IDs, ensemble size, paired image and patient counts, class counts |
| T2 Performance and uncertainty | Core | Condition × fold, plus pooled | Accuracy, balanced accuracy, macro-F1, NLL, Brier, mean total uncertainty, mean aleatoric uncertainty, mean EU |
| T3 Paired effects | Core | Comparison | Paired counts, change in balanced accuracy and macro-F1, and for total, aleatoric, and epistemic uncertainty: mean and median reduction in nats, relative reduction (%), and share of images with a reduction (%) |
| T4 Class breakdown | Supporting | Condition × true class | Support, precision, recall, F1. Include per-class dEU and paired counts in comparison rows if selected. |

## 4. Figures

| ID | Priority | Figure | Axes, units, and reference marks |
| --- | --- | --- | --- |
| F1 | Core | Paired EU scatter | Image-only EU on x, multimodal EU on y, both in nats. Use common limits and a diagonal identity line. Each matched image contributes one point. |
| F2 | Core | Distribution of paired dEU | Four columns in order: both correct, image-only wrong and multimodal correct, image-only correct and multimodal wrong, both wrong. Plot dEU in nats on x and image count on y. Share bin edges and axis limits, mark zero, and label each panel with its image count. |
| F3 | Core | Confusion matrices | True class on rows, predicted class on columns, with shared class order and color scale. Normalize by true-class support and annotate raw counts. |
| F4 | Supporting | Paired effect estimates | Comparison or class on y, mean dEU in nats on x, with a zero line. |
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
    data/
      paired_images.csv
      validation_history.csv
    figures/
      paired_eu_scatter.png
      paired_deu_distribution.png
      confusion_matrices.png
      paired_effect_estimates.png
      validation_curves.png
  summary/
    experiment_uncertainty.csv
    total_vs_aleatoric_uncertainty.png
```

- Use a distinct experiment ID when inputs or settings change. A rerun with the same inputs and settings regenerates the same paths. For per-comparison figures within an experiment, append the comparison ID to the filename before the extension.
- Record the analysis source revision, input artifact paths, resolved experiment settings, class order, and metric conventions in `manifest.yaml`. Link the tables and figures from `REPORT.md`.
- Write UTF-8 CSVs with explicit units in column names where applicable. Key paired rows by comparison, fold, and image ID. Key validation rows by setup, fold, member seed, and epoch.
- Update the summary CSV by experiment ID and setup after both test folds complete. Regenerate F5 from that CSV so reruns replace the previous points for the same experiment.

### Acceptance checks

- Verify that paired rows, class counts, and metric values agree across tables and figures. Check that F2 panel counts sum to the paired image count and that F6 uses only epochs present in the member histories.
- Check formulas on a small hand-calculated example. Identical member distributions yield zero disagreement, and identical conditions yield zero paired effects.

Existing references: [uncertainty analysis plan](uncertainty_analysis.md) and [metadata experiment report](../outputs/metadata-experiment-analysis/REPORT.md).
