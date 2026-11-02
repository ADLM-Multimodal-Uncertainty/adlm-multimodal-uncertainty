# Design decisions

Open design questions for the experiments in this repo, the options for each, and where the
code stands. Last updated: 2 Nov 2026.

Status labels:

- **Current**: what `experiment=draft_baseline` does today (the default).
- **Available**: implemented; select it through config or a CLI override, no code change.
- **Not implemented**: needs code first (a new class plus a YAML, or a change to the pipeline).

Dataset context: 2,298 images from 1,373 patients. Class counts: ACK 730, BCC 845, MEL 52,
NEV 244, SCC 192, SEK 235. The default holdout split has 1,602 / 354 / 342 images
(train / val / test), with 41 / 6 / 5 MEL.

---

## A. Data and splits

### A1. Split protocol
Current: one patient-grouped holdout, ~70/15/15, seed 42, retried until every class is in every
partition. Not stratified: test holds only 5 MEL images.
- Patient holdout 70/15/15 (`split=patient_holdout`): **Current**
- Patient-grouped 5-fold from the committed CSV, test = `fold`, val = `fold + 1`
  (`split=patient_kfold fold=0..4`): **Available**
- Repeated holdout over several split seeds (`split.seed=...`): **Available**
- Class-stratified patient-grouped split (holdout or new folds): **Not implemented**
- Nested cross-validation (inner folds for model selection): **Not implemented**

### A2. Unit of analysis
Current: every image is one sample; metrics are per image.
- Image level: **Current**
- Lesion level (aggregate images of one lesion): **Not implemented**
- Patient level (aggregate per patient): **Not implemented**

### A3. Role of the validation set
Current: checkpoint selection only (best epoch by `train.monitor`).
- Checkpoint selection: **Current**
- Also fit calibration (temperature scaling) on validation: **Not implemented**
- Also tune decision thresholds on validation: **Not implemented**

### A4. Test-set discipline
Current: the test set is scored automatically at the end of every run.
- Evaluate test after every run: **Current**
- Separate, deliberate test step after model selection (as in the draft repo): **Not implemented**
  (`python -m src.evaluate <run_dir>` exists, but `train.py` still calls it automatically)

### A5. Rows with the clinical block missing (804 images)
Current: numeric values imputed with the training median, categorical values get a "Missing"
category; no indicator columns.
- Median imputation + "Missing" category: **Current**
- Add missingness indicator columns: **Not implemented**
- Train and evaluate on complete rows only: **Not implemented**
- Report all variants side by side: **Not implemented**

### A6. Leakage columns
Current: `biopsed` (how the label was verified) and the ID columns are never used as features.
- Exclude `biopsed`, `patient_id`, `lesion_id`, `img_id`: **Current**

---

## B. Metadata preprocessing

### B1. Feature set
Current: all 21 clinical attributes (3 numeric, 18 categorical), listed in
`configs/data/pad_ufes20.yaml`.
- All 21 attributes: **Current**
- Subsets (demographic, symptoms, history) by editing `data.numeric_features` /
  `data.categorical_features`: **Available**
- Drop ancestry and socioeconomic covariates (`background_*`, `has_piped_water`,
  `has_sewage_system`): **Available** (same lists)
- No metadata (image only, `experiment=draft_image_only`): **Available**
- Permuted-metadata control (shuffle metadata across images): **Not implemented**

### B2. Numeric imputation
- Training median: **Current**
- Training mean: **Not implemented**
- Median plus missingness indicator: **Not implemented**
- Model-based imputation: **Not implemented**

### B3. Numeric scaling
- Z-score fitted on training rows: **Current**
- Robust scaling (median / IQR): **Not implemented**
- Quantile transform: **Not implemented**
- None: **Not implemented**

### B4. Categorical encoding
- One-hot, "Missing" as its own category, unseen categories encode as all zeros: **Current**
- `fitspatrick` as an ordinal number: **Not implemented**
- Learned per-feature embeddings: **Not implemented**
- Group rare categories: **Not implemented**

### B5. Boolean attributes
- Treated as categories (one-hot over True / False / Missing): **Current**
- Single 0/1 column plus missing indicator: **Not implemented**

---

## C. Images

### C1. Resolution and aspect ratio
Current: resized to 224 x 224 without keeping the aspect ratio.
- Square resize to `data.image_size` (224): **Current**
- Other square size, e.g. 320 or 384 (`data.image_size=...`): **Available**
- Pad to square, then resize: **Not implemented**
- Resize shorter side, then center crop: **Not implemented**

### C2. Pixel normalization
- ImageNet mean and std: **Current**
- PAD-UFES-20 dataset statistics: **Not implemented**
- Per-image normalization: **Not implemented**

### C3. Training augmentation
Current: random horizontal flip only, hard-coded in `src/data/transforms.py` (not configurable).
- Horizontal flip: **Current**
- Vertical flip, rotation: **Not implemented**
- Random resized crop: **Not implemented**
- Color jitter (careful: color can be diagnostic): **Not implemented**
- Blur, cutout / random erasing: **Not implemented**
- RandAugment / TrivialAugment: **Not implemented**
- No augmentation: **Not implemented**

### C4. Test-time augmentation
- None: **Current**
- Average predictions over flips: **Not implemented**

---

## D. Class imbalance

### D1. Loss function
- Unweighted cross-entropy: **Current**
- Inverse-frequency class weights, n / (K x count_k) (`train.weighted_loss=true`):
  **Available**
- Label smoothing (`+train.loss.label_smoothing=0.1`; the `+` adds a key the YAML does not
  list): **Available**
- Softened (square-root) class weights: **Not implemented**
- Focal loss: **Not implemented**
- Class-balanced loss (effective number of samples), LDAM: **Not implemented**

### D2. Sampling
- Plain shuffling: **Current**
- Oversampling with a weighted random sampler: **Not implemented**
- Undersampling of majority classes: **Not implemented**
- Class-balanced batches: **Not implemented**

### D3. Post-hoc correction
- None: **Current**
- Logit / prior adjustment: **Not implemented**
- Per-class decision thresholds tuned on validation: **Not implemented**

### D4. Interaction with uncertainty estimates
Reweighting and resampling change the predicted probabilities, so they change calibration and
every entropy-based uncertainty. Open question: use an unweighted model for the UQ experiments,
or a weighted one followed by calibration (A3).
- Not decided: **Not started**

---

## E. Image encoder

### E1. Architecture family
- ResNet18 (torchvision): **Current**
- Other torchvision models ending in `fc`: ResNet34/50/101, ResNeXt, Wide-ResNet
  (`model.image_encoder.arch=resnet50`): **Available**
- EfficientNet, ConvNeXt: **Not implemented** (need their own encoder class)
- ViT, DeiT, Swin: **Not implemented**
- Foundation / self-supervised backbones (DINOv2, CLIP): **Not implemented**
- Dermatology-pretrained backbones: **Not implemented**

### E2. Pretraining
- Supervised ImageNet weights: **Current**
- Random initialisation (`pretrained=false freeze=false`): **Available**
- Self-supervised or foundation weights: **Not implemented**

### E3. Fine-tuning
- Fully frozen backbone (`model.image_encoder.freeze=true`): **Current**
- Full fine-tuning (`model.image_encoder.freeze=false`, usually with a lower lr): **Available**
- Fine-tune only the last block(s): **Not implemented**
- Gradual unfreezing: **Not implemented**
- Layer-wise / discriminative learning rates: **Not implemented**
- LoRA or adapters: **Not implemented**

### E4. BatchNorm in the backbone
- Frozen running statistics (backbone kept in eval mode while frozen): **Current**
- Train-mode BatchNorm when fine-tuning (statistics updated per batch): **Available** (what
  `freeze=false` does)
- Frozen statistics with trainable affine weights while fine-tuning: **Not implemented**
- Replace with GroupNorm: **Not implemented**

### E5. Features taken from the backbone
- Final pooled vector (512 for ResNet18): **Current**
- Multi-scale features from several stages: **Not implemented**
- Spatial feature maps / patch tokens (needed for attention-based fusion): **Not implemented**

### E6. Image projection layer
- Linear 512 -> 128, ReLU, dropout 0.2: **Current**
- Other width or dropout (`model.image_encoder.proj_dim`, `.dropout`): **Available**
- No projection (`model.image_encoder.proj_dim=null`): **Available**
- Projection with normalization layer: **Not implemented**

### E7. Feature caching for a frozen backbone
- Recompute backbone features every epoch (needed for augmentation): **Current**
- Precompute features once and train heads on them (much faster, fixes augmentation):
  **Not implemented**

---

## F. Metadata encoder and fusion

### F1. Metadata (tabular) encoder
- MLP, one hidden layer of 64, dropout 0.2: **Current**
- Other depth / widths / dropout (`model.tabular_encoder.hidden_dims=[128,64]`): **Available**
- No encoder, raw features concatenated (`model.tabular_encoder.hidden_dims=[]`): **Available**
- Per-feature embeddings: **Not implemented**
- FT-Transformer / TabTransformer: **Not implemented**
- Normalization layers in the MLP: **Not implemented**

### F2. Fusion strategy
- Concatenate pooled features, then MLP (`model/fusion=concat`): **Current**
- Late fusion: average or learned weighting of per-modality logits: **Not implemented**
- Gated fusion: **Not implemented**
- FiLM (metadata modulates image features): **Not implemented**
- Cross-attention between metadata and image tokens: **Not implemented** (needs E5)
- Bilinear / tensor fusion: **Not implemented**
- Product of experts: **Not implemented**

### F3. Classifier head
- One hidden layer: 192 -> 128, ReLU, dropout 0.3, -> 6: **Current**
- Other hidden width or dropout (`model.fusion.hidden`, `.dropout`): **Available**
- More hidden layers: **Not implemented**
- Normalization in the head: **Not implemented**

### F4. Missing metadata at inference
- A model with a metadata branch requires metadata (error otherwise): **Current**
- Zero-fill the metadata vector: **Not implemented**
- Learned "missing" embedding: **Not implemented**
- Modality dropout during training (see G1): **Not implemented**

---

## G. Training

### G1. Image-only vs multimodal: joint or separate training
Current: two separate runs (`draft_image_only`, `draft_baseline`). They share only the frozen
backbone; projection and heads are trained independently.
- Separate runs: **Current**
- One model with modality dropout, evaluated with and without metadata (same weights):
  **Not implemented**
- One run with a shared image projection, an image-only head and a fusion head, summed loss:
  **Not implemented**
- Weighting of the summed losses: **Not implemented**

### G2. Epochs
- 10 epochs: **Current**
- More epochs, e.g. 30 (`train.epochs=30`): **Available**

### G3. Early stopping and checkpoint choice
- No early stopping; keep the epoch with the best `train.monitor`: **Current**
- Early stopping with patience on validation loss or macro-F1: **Not implemented**
- Select on validation loss (lower is better): **Not implemented** (`monitor` only maximises)
- Keep the last epoch: **Not implemented**
- Checkpoint averaging / SWA: **Not implemented**

### G4. Optimizer and learning rate
- AdamW, lr 1e-3, PyTorch default weight decay 0.01: **Current**
- Other learning rate (`train.optimizer.lr=3e-4`): **Available**
- Other weight decay (`+train.optimizer.weight_decay=...`): **Available**
- Other optimizer, e.g. SGD with momentum
  (`train.optimizer._target_=torch.optim.SGD +train.optimizer.momentum=0.9`): **Available**
- Separate learning rates for backbone and heads: **Not implemented**

### G5. Learning-rate schedule
- Constant learning rate: **Current**
- Cosine annealing: **Not implemented**
- Linear warmup: **Not implemented**
- Reduce on plateau: **Not implemented**

### G6. Batch size and precision
- Batch size 32, full precision: **Current**
- Batch size 16 or 64 (`train.batch_size=...`): **Available**
- Gradient accumulation: **Not implemented**
- Mixed precision (AMP) for cluster GPUs: **Not implemented**

### G7. Regularization
- Dropout in projection, metadata encoder and head; frozen backbone: **Current**
- Other dropout rates / weight decay: **Available**
- Mixup / CutMix: **Not implemented**

### G8. Seeds and repetitions
- One training seed (42) and one split seed (42): **Current**
- Several seeds per configuration, e.g. `-m seed=0,1,2,3,4`: **Available**
- Report mean and spread over seeds and folds: **Not implemented** (no aggregation script)

---

## H. Metrics and evaluation

### H1. Checkpoint-selection metric
- Validation macro-F1: **Current**
- Validation accuracy or balanced accuracy (`train.monitor=...`): **Available**
- Validation loss: **Not implemented** (see G3)
- MEL recall: **Not implemented**

### H2. Reported metrics
- Accuracy, balanced accuracy, macro-F1, per-class precision / recall / F1, confusion matrix:
  **Current**
- Macro one-vs-rest AUROC: **Not implemented**
- Per-class sensitivity and specificity (MEL in particular): **Not implemented**
- Calibration: ECE, NLL, Brier score: **Not implemented**
- Selective prediction (risk-coverage curve, AURC): **Not implemented**

### H3. Aggregation and statistical confidence
- Single run, image level: **Current**
- Patient-level aggregation: **Not implemented**
- Patient-level bootstrap confidence intervals: **Not implemented**
- Paired tests between configurations: **Not implemented**
- Mean and spread over seeds and folds: **Not implemented**

### H4. Overfitting diagnostics
- Training and validation loss and metrics per epoch in `history.csv` and `curves.png`
  (`train.track_train_metrics=true`): **Current**

### H5. Visualizations
- Loss and metric curves (train dashed, validation solid), test confusion matrix: **Current**
- Reliability diagrams: **Not implemented**
- Per-class bar charts: **Not implemented**
- Uncertainty histograms / dEU plots: **Not implemented**

---

## I. Uncertainty quantification (project core)

Nothing in this section is started: `configs/ensemble/` and `configs/uq/` are empty.

### I1. Epistemic uncertainty estimator
- Deep ensemble over seeds: **Not implemented**
- MC dropout (which dropout layers, number of passes): **Not implemented**
- Last-layer Laplace approximation (laplace-torch is already a dependency): **Not implemented**
- Ensemble of heads only on the shared frozen backbone (cheap, but less diverse, likely
  underestimates epistemic uncertainty): **Not implemented**
- SWAG: **Not implemented**

### I2. Ensemble diversity and size
- Seeds only: **Not implemented**
- Seeds plus bootstrap resampling of the training data: **Not implemented**
- Different architectures per member: **Not implemented**
- Size: 5 or 10 members: **Not decided**

### I3. Uncertainty decomposition
- Entropy-based: total = aleatoric + epistemic (mutual information): **Not implemented**
- Variance-based decomposition: **Not implemented**

### I4. With vs without metadata (how dEU is measured)
- Design 1, paired separate models (image-only vs multimodal): **Not implemented**
  (the paired models exist as `draft_image_only` / `draft_baseline`, the dEU computation does not)
- Design 2, same weights with metadata masked (needs G1 modality dropout): **Not implemented**

### I5. Controls for dEU
- Permuted metadata: **Not implemented**
- Metadata subsets: **Available** for training (B1); dEU comparison **Not implemented**
- Complete rows only: **Not implemented**
- Without missingness indicators: **Not implemented** (indicators do not exist yet, A5)

### I6. Calibration before uncertainty
- None: **Current**
- Temperature scaling on the validation set: **Not implemented**

---

## Suggested order

1. A1 and G8: split protocol and seeds (decide how noisy every comparison is).
2. G1 and I4: joint or separate training (decides how dEU is defined).
3. D1 and D4: imbalance handling and its effect on probabilities.
4. E3, G2, G3: fine-tuning, epoch budget, early stopping.
5. I1 to I3: uncertainty method, once the base model is fixed.
