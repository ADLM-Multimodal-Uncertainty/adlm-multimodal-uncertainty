# Exploratory data analysis: plan and first findings

The findings below come from a first pass over the metadata CSV only (figures kept in
`project-docs/eda-figures/` outside the repo). Image statistics are added once the images are on
the cluster. The numbered prefixes are the planned output file names.

## What the EDA covers

1. Structure (`01_structure.*`): images per lesion, lesions per patient, images per patient;
   number of lesions with more than one image (these must never be split across folds).
2. Labels (`02_*`): class counts (images and patients), ground-truth source (biopsy vs clinical
   consensus) per class.
3. Missingness (`03_*`): missing fraction per attribute, missingness pattern heatmap, clinical
   block missingness by class and biopsy status.
4. Informativeness (`04_*`): Cramer's V between each categorical/boolean attribute and the
   diagnosis (epsilon^2 from Kruskal-Wallis for numeric ones), on all rows and on complete rows;
   age by class; body region given class.
5. Metadata-only baselines (`05_*`): logistic regression and random forest on the encoded
   attributes under a 5-fold patient-grouped split, for the full vector, with and
   without numeric missingness indicators, complete rows only, and the three attribute subsets.
6. Images (`06_*`, needs the images): resolution scatter, example grid per class.
7. Fold table (`07_fold_table.csv`): class and patient counts per fold of the split.

## Findings so far (metadata only)

Structure
- 2298 images, 1373 patients. `lesion_id` is only unique within a patient: 1641 distinct ids
  but 1891 distinct (patient, lesion) pairs. Grouping by lesion alone would leak; we group by patient.
- 361 lesions have more than one image (768 images in total); 355 patients have several lesions;
  179 patients carry more than one diagnosis.

Labels
- BCC 845, ACK 730, NEV 244, SEK 235, SCC 192, MEL 52. MEL has 52 images: expect wide CIs,
  never report MEL uncertainty estimates as precise.
- All BCC, SCC and MEL images are biopsy-confirmed; ACK 24%, NEV 25%, SEK 6%. Label noise is
  therefore class-dependent and concentrated in the benign/premalignant classes.

Missingness: the main confound of this project
- 804 images (35%) have the entire clinical block missing (smoke, drink, pesticide, gender,
  histories, water/sewage, fitzpatrick, diameters, family background). `grew` and `changed` are
  missing in ~400 more. Age and region are never missing.
- Every one of those 804 images is non-biopsied and belongs to ACK, NEV or SEK. No BCC, SCC or
  MEL image has the block missing. "Metadata is missing" is thus a perfect rule for "not
  malignant" on this dataset.
- Consequence for the research question: a fusion model may lower its epistemic uncertainty
  simply by reading the missingness pattern, not the clinical content. The attributes of the
  missing block have Cramer's V ~0.50 with the diagnosis on all rows but only 0.04-0.15 on
  complete rows.
- Decision: the main experiments are run twice, on all rows (clinically realistic) and on the
  1494 complete rows, and every dEU claim is checked on both. Dropping explicit missingness
  indicator columns is not a sufficient control, because "unknown" one-hot levels still encode
  the pattern.

Informativeness (complete rows, i.e. without the shortcut)
- Symptom attributes carry most of the signal: grew 0.35, changed 0.33, elevation 0.32,
  itch 0.31, bleed 0.28, region 0.28, hurt 0.24. Demographics and exposure history are weak
  (0.04-0.15). Age is strongly associated (NEV patients are much younger).
- This fixes the subsets for the "informative vs weak metadata" question:
  `symptoms` (informative) vs `history` (weak) vs `demographic`.

Metadata-only baselines (5-fold patient-grouped CV, chance BACC = 0.167)
- Logistic regression, full vector, all rows: BACC 0.65, macro AUC 0.91.
- Complete rows only: BACC 0.57, macro AUC 0.84 (the drop is the shortcut).
- Subsets: symptoms 0.57, demographic 0.51, history 0.34.
- The metadata alone is a strong second modality; the question is whether it changes what the
  image model does not know (EU), not only what it predicts.

## Still to do when images are available

- Resolution distribution and aspect ratios (smartphone photos, expect high variance).
- Example grid per class; look for recurring artefacts (rulers, pen marks, ink) that a model
  could use as shortcuts.
- Colour statistics per class to decide on colour augmentation strength.
- Confirm every `img_id` in the CSV has a file (the dataset paper reports 2298 images).
