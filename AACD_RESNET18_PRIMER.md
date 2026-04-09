# AACD ResNet-18 Primer

This document explains the AACD path used when the student is `resnet18` and `use_mobilevit=false`.

It is based on the implementation in:

- `src/models/aacd_module.py`
- `src/models/components/aacd_campus.py`
- `src/models/components/agreement.py`
- `src/models/components/concept_basis.py`
- `src/models/components/aacd_criterion.py`
- `src/models/components/campus.py`
- `src/models/components/ae_svc.py`
- `src/models/components/cca_module.py`
- `src/models/components/dino_teacher.py`

## 1. Scope

This primer covers the path configured by:

- `model.net.student.arch=resnet18`
- `model.net.use_mobilevit=false`

Example config: `configs/experiment/aacd_cub.yaml`

In this path:

- the CLIP image encoder is frozen
- the DINOv2 image encoder is frozen
- the text encoder is frozen through CLIP text features
- the ResNet-18 student is trainable
- the AACD concept-space modules are trainable, except for cached AE-SVC encoders and fixed CCA statistics

## 2. High-Level Idea

AACD trains a student classifier while using two frozen teachers:

- CLIP image features
- DINOv2 image features

The two teacher spaces do not match directly, so the code first builds a shared aligned space:

1. extract teacher features on the training set
2. regularize those features with AE-SVC
3. fit CCA between CLIP and DINO features
4. define a concept basis on top of the CCA space
5. distill the student into that concept space

The key design choice is:

- do not force the student to copy raw CLIP or raw DINO features directly
- instead, distill toward a concept-space target built from agreement between the two teachers

## 3. Main Objects

### 3.1 Teacher-Student Wrapper

`AACDTeacherStudent` in `src/models/components/aacd_campus.py` builds:

- `clip_teacher`: frozen CLIP image encoder
- `dino_teacher`: frozen DINOv2 image encoder
- `student`: ResNet-18 classifier path
- `condensation_shared`: student projection head into shared space
- `agreement`: CCA projection and teacher agreement module
- `concept_basis`: concept projection and disagreement gating
- `clip_ae_encoder`, `dino_ae_encoder`: frozen AE-SVC encoders used after offline initialization

### 3.2 Lightning Module

`AACDModule` in `src/models/aacd_module.py` does two jobs:

- one-time AACD initialization before normal training
- normal epoch-by-epoch training of the student path and concept basis

This separation matters:

- AE-SVC and CCA are prepared in `setup("fit")`
- the main `training_step()` does not retrain AE-SVC or refit CCA

## 4. What Is Frozen and What Is Trainable

### Frozen

- CLIP image encoder
- DINOv2 image encoder
- CLIP text features
- trained AE-SVC encoders after initialization
- CCA means and projection matrices
- concept anchor matrix `Phi`

### Trainable During Main AACD Training

- ResNet-18 student backbone and classifier
- `condensation_shared`
- `ConceptBasis.A`

### Not Used for Loss

- `align_nlp` exists but is frozen and marked diagnostic-only

## 5. End-to-End Pipeline

AACD for ResNet-18 runs in two phases:

1. offline-style initialization inside `setup("fit")`
2. normal per-batch training inside `training_step()`

The sections below follow that order.

## 6. Phase A: Initialization Before Main Training

## 6.1 Trigger Point

When `trainer.fit(model, datamodule, ...)` is called, Lightning invokes `AACDModule.setup("fit")`.

For AACD, `setup("fit")` calls `_setup_aacd_state()`, which either:

- loads a cached initialization from disk, or
- runs the full initialization pipeline once and caches it

This initialization contains the AE-SVC training, CCA fitting, and concept basis setup.

## 6.2 Step A1: Extract Frozen Teacher Features on the Training Set

The code iterates over `dm.data_train` and computes:

- CLIP image features
- DINO image features
- labels

The preprocessing is branch-specific:

- CLIP branch uses CLIP mean/std normalization
- DINO branch uses ImageNet mean/std normalization

Mathematically, for each image `x_i`:

- `c_i = f_clip(x_i^clip) in R^(d_c)`
- `d_i = f_dino(x_i^dino) in R^(d_d)`

where:

- `f_clip` is frozen CLIP image encoding
- `f_dino` is frozen DINOv2 encoding
- both outputs are L2-normalized in their wrappers

The full training matrices are:

- `C = [c_1; ...; c_N] in R^(N x d_c)`
- `D = [d_1; ...; d_N] in R^(N x d_d)`

These are cached to disk for reuse.

## 6.3 Step A2: Train AE-SVC on CLIP Features

The code trains an autoencoder on the cached CLIP features:

- input: `C`
- target: reconstruct `C`
- latent: same dimension as input in this implementation

The model is `AE_SVC(input_dim=d_c, latent_dim=d_c)`.

For a batch `x` of teacher features:

- `z = Encoder(x)`
- `x_rec = Decoder(z)`

The loss is:

`L_AE = 25 L_rec + L_cov + 15 L_var + L_mean`

with:

- `L_rec = mean(||x - x_rec||_2^2)`
- `L_cov = ||Cov(z) - I||_F^2`
- `L_var = mean_k (Var(z_k) - 1)^2`
- `L_mean = mean_k (mean(z_k))^2`

Interpretation:

- `L_rec` preserves information
- `L_cov` discourages correlated latent dimensions
- `L_var` pushes each latent dimension toward unit variance
- `L_mean` centers the latent space near zero

After training, the code keeps only the encoder and transforms all CLIP features:

- `C_svc = Encoder_clip(C)`

## 6.4 Step A3: Train AE-SVC on DINO Features

Exactly the same procedure is repeated for DINO:

- model: `AE_SVC(input_dim=d_d, latent_dim=d_d)`
- transformed features: `D_svc = Encoder_dino(D)`

After this point:

- `clip_ae_encoder` is frozen and stored in the model
- `dino_ae_encoder` is frozen and stored in the model

These encoders are later reused at training time for batch teacher features.

## 6.5 Step A4: Fit CCA Between CLIP and DINO

Now AACD fits CCA on:

- `C_svc`
- `D_svc`

Let the centered matrices be:

- `C_0 = C_svc - mu_C`
- `D_0 = D_svc - mu_D`

Compute covariance matrices:

- `Sigma_CC = C_0^T C_0 / N + lambda I`
- `Sigma_DD = D_0^T D_0 / N + lambda I`
- `Sigma_CD = C_0^T D_0 / N`

CCA whitens each side and solves the SVD:

- `M = Sigma_CC^(-1/2) Sigma_CD Sigma_DD^(-1/2)`
- `M = U S V^T`

The canonical correlations are the singular values:

- `rho_1 >= rho_2 >= ...`

The projection matrices are:

- `A = U^T Sigma_CC^(-1/2)`
- `B = V^T Sigma_DD^(-1/2)`

The code keeps the top `s` rows:

- `A_s in R^(s x d_c)`
- `B_s in R^(s x d_d)`

where `s = 128` in the ResNet-18 config.

Per-sample CCA projections are then:

- `c_tilde_i = (c_i - mu_C) A_s^T in R^s`
- `d_tilde_i = (d_i - mu_D) B_s^T in R^s`

This is the shared teacher space used by AACD.

## 6.6 Step A5: Initialize Agreement Diagnostics

`AgreementModule.initialize(...)` stores:

- `mu_C`, `mu_D`
- `A_s`, `B_s`
- class prototypes in shared space

The class prototype for class `k` is:

- `p_k = mean_{i: y_i=k} 0.5 (c_tilde_i + d_tilde_i)`

Then each `p_k` is L2-normalized.

These prototypes are used only for diagnostic top-1 teacher agreement logging, not as the main distillation target.

## 6.7 Step A6: Initialize Concept Basis from Text

The model has frozen CLIP text features for all class prompts.

For class text features:

- `t_j in R^(d_c)`

the code applies the CLIP AE-SVC encoder:

- `t_j^svc = Encoder_clip(t_j)`

Then it projects text into the same CLIP-side CCA space:

- `t_j^cca = (t_j^svc - mu_C) A_s^T in R^s`

These projected text directions are stacked into a matrix and normalized column-wise to initialize:

- `Phi in R^(s x K)`

where `K = num_concepts = 128`.

Then the learnable concept matrix is initialized as:

- `A_concept := Phi`

This means the learnable concept basis starts from text-derived concept directions in the shared teacher space.

## 6.8 Step A7: Calibrate Per-Concept Gating Strength

The code computes CCA-projected training features:

- `c_tilde_train`
- `d_tilde_train`

Then it projects them into concept space:

- `z_i^c = c_tilde_i A_norm`
- `z_i^d = d_tilde_i A_norm`

where `A_norm` is the column-normalized concept matrix.

Per-concept disagreement is:

- `delta_{i,k} = |z_{i,k}^c - z_{i,k}^d|`

The code estimates disagreement variance per concept:

- `sigma_k^2 = Var(delta_:,k) / 2`

and sets the per-concept gate sharpness:

- `alpha_k = 1 / (2 sigma_k^2)`

Interpretation:

- small disagreement variance means concept `k` is stable across teachers
- stable concepts get larger `alpha_k`
- larger `alpha_k` makes the gate decay faster when disagreement appears

The code also copies top CCA correlations into `correlation_weights`.

## 6.9 Step A8: Cache Initialization

The following are cached:

- agreement state
- concept basis state
- trained CLIP AE-SVC encoder
- trained DINO AE-SVC encoder
- readiness flags

Future runs can skip the expensive initialization and load this state directly.

## 7. Phase B: Per-Batch Forward Pass During Training

After initialization, the main training loop starts.

For one mini-batch of images `x` with labels `y`, the forward path is:

## 7.1 Step B1: Branch-Specific Input Preparation

For each batch image:

- `x_clip = normalize_clip(x)`
- `x_dino = normalize_imagenet(x)`
- `x_student = normalize_imagenet(x)`

In the ResNet-18 path, the student uses the ImageNet-style input path.

## 7.2 Step B2: Frozen Teacher Features

The model computes:

- `c = f_clip(x_clip)`
- `d = f_dino(x_dino)`

Both are detached frozen features.

## 7.3 Step B3: Student Forward

The ResNet-18 student is implemented as:

- torchvision ResNet-18 backbone with the original classifier removed
- dropout
- linear classifier

For each sample:

- `h = f_student(x_student) in R^(d_s)`
- `logits = W_cls Dropout(h) + b`

The code L2-normalizes `h` before returning it from the student wrapper.

For ResNet-18, `d_s` is the backbone feature dimension of the modified network.

## 7.4 Step B4: Apply the Frozen AE-SVC Encoders

If AE-SVC initialization is ready, the code transforms teacher features:

- `c' = Encoder_clip(c)`
- `d' = Encoder_dino(d)`

This keeps train-time teacher features consistent with the offline features used to fit CCA.

Without this step, the CCA projections would be mismatched relative to initialization.

## 7.5 Step B5: Project Both Teachers Into the Shared CCA Space

Using the stored CCA statistics:

- `c_tilde = (c' - mu_C) A_s^T`
- `d_tilde = (d' - mu_D) B_s^T`

Both have shape:

- `[B, s]`, with `s = 128`

This is the teacher alignment stage.

## 7.6 Step B6: Diagnostic Teacher Agreement

The code computes normalized similarities to class prototypes:

- `S_clip = normalize(c_tilde) P^T`
- `S_dino = normalize(d_tilde) P^T`

Then:

- `clip_top1 = argmax S_clip`
- `dino_top1 = argmax S_dino`
- `agree_top1 = (clip_top1 == dino_top1)`

This is logged as an agreement diagnostic only.

It does not directly create the distillation target in the upgraded AACD path.

## 7.7 Step B7: Project Teachers Into Concept Space

Let the normalized concept basis be:

- `A_bar = normalize_columns(A_concept)`

Then concept activations are:

- `z^c = c_tilde A_bar in R^(B x K)`
- `z^d = d_tilde A_bar in R^(B x K)`

These are concept-wise teacher responses.

## 7.8 Step B8: Compute Per-Concept Disagreement Gates

For each sample `i` and concept `k`:

- `delta_{i,k} = |z_{i,k}^c - z_{i,k}^d|`
- `w_{i,k} = exp(-alpha_k delta_{i,k})`

Properties:

- if the teachers agree on a concept, `delta` is small and `w` is near `1`
- if they disagree, `delta` is larger and `w` shrinks toward `0`

This is the core AACD gating mechanism.

It is concept-wise, not sample-wise.

## 7.9 Step B9: Build the Shared Teacher Distillation Target

AACD fuses the two teacher concept activations using correlation weights:

- `u_k = correlation_weight_k`
- `z_shared_{i,k} = u_k z^c_{i,k} + (1 - u_k) z^d_{i,k}`

So:

- concepts with stronger CLIP/DINO correlation trust the CLIP concept more
- weaker ones lean relatively more toward DINO

This produces:

- `shared_target in R^(B x K)`

## 7.10 Step B10: Project the Student Into Shared and Concept Spaces

The student hidden feature `h` is pushed through a small projection head:

- `s = g(h) in R^s`

where `g = condensation_shared`.

Then the student concept activations are:

- `z_hat = s A_bar in R^(B x K)`

This is the student representation that will be matched against both:

- `shared_target`
- text-derived concept targets

## 7.11 Step B11: Build Text Concept Targets

For each ground-truth label `y_i`, the code fetches the corresponding frozen CLIP text feature:

- `t_{y_i}`

Then it uses the CLIP-side AE-SVC encoder and CLIP-side CCA projection:

- `t_{y_i}^svc = Encoder_clip(t_{y_i})`
- `t_{y_i}^cca = (t_{y_i}^svc - mu_C) A_s^T`

Finally:

- `z_i^text = t_{y_i}^cca A_bar`

This gives:

- `text_concept_targets in R^(B x K)`

Important detail:

- there is no separate text-only AE-SVC
- text reuses the trained CLIP AE-SVC encoder because text lives on the CLIP side of the shared space in this implementation

## 7.12 Step B12: Compute Concept Basis Regularizers

Two regularizers are evaluated:

### Anchoring

- `L_anchor = ||A_concept - Phi||_F^2 / (sK)` in effect, since the code uses `.mean()`

Meaning:

- keep the learnable concept basis close to the text-initialized concept basis

### Orthogonality

Let `A_bar = normalize_columns(A_concept)`.

- `L_orth = ||A_bar^T A_bar - I||_F^2`

Meaning:

- encourage different concepts to stay distinct rather than collapse into similar directions

## 8. Training Loss

The total loss in `AACDCriterion` is:

`L_total = cls_w L_cls + kd_scale lambda_shared L_shared + kd_scale lambda_txt L_txt + lambda_anchor L_anchor + lambda_orth L_orth + kd_scale lambda_feat L_feat`

For the ResNet-18 path:

- `lambda_feat = 0`
- so `L_feat = 0`

### 8.1 Classification Loss

- `L_cls = CrossEntropy(logits, y)`

with label smoothing in the code.

### 8.2 Shared Concept KD

- `L_shared = mean_{i,k} w_{i,k} (z_hat_{i,k} - z_shared_{i,k})^2`

Interpretation:

- if the teachers disagree on concept `k` for sample `i`, down-weight that concept
- if they agree, push the student toward the fused teacher target strongly

### 8.3 Text KD

- `L_txt = mean_{i,k} w_{i,k} (z_hat_{i,k} - z_text_{i,k})^2`

Interpretation:

- use the same concept-wise agreement gate for text supervision
- text supervision is strongest on concepts where the image teachers also agree

### 8.4 Concept Regularization

- `L_concept_reg = lambda_anchor L_anchor + lambda_orth L_orth`

This replaces the older geometry loss mentioned in comments.

### 8.5 Epoch Scheduling

The criterion changes weights over training:

- `cls_w = lambda_cls + progress * (1 - lambda_cls)`
- `kd_scale = 1 - 0.5 * progress`

where:

- `progress = epoch / max_epochs`

Interpretation:

- classification weight grows during training
- KD influence is reduced slightly over time

## 9. What Actually Gets Optimized Each Batch

During `training_step()`:

1. run the forward pass described above
2. build `L_total`
3. backpropagate `L_total`
4. update only parameters with `requires_grad=True`

In practice for ResNet-18 AACD, this means optimization primarily updates:

- ResNet-18 student parameters
- student classifier
- `condensation_shared`
- `ConceptBasis.A`

It does not update:

- CLIP teacher
- DINO teacher
- CCA matrices
- AE-SVC encoders

## 10. Why AE-SVC Exists Here

CCA is sensitive to covariance structure and feature scaling.

The role of AE-SVC is to regularize each teacher space before CCA so that:

- features remain reconstructive
- latent dimensions are less entangled
- latent statistics are closer to standardized

This makes the subsequent CCA alignment more stable and more meaningful.

In short:

- AE-SVC improves each individual teacher space
- CCA aligns the two teacher spaces
- ConceptBasis turns the aligned space into concept directions
- AACD distills the student only on concepts where the teachers agree

## 11. Why the Concept Anchoring Loss Matters

The concept matrix starts from text-derived directions:

- `A_concept(0) = Phi`

But during training, `A_concept` is allowed to move.

Without anchoring:

- the concept basis could drift too far from the semantic text initialization
- concept dimensions might become arbitrary hidden axes that no longer correspond well to text semantics

So the anchoring loss keeps the learned concept basis semantically tethered.

## 12. ResNet-18-Specific Notes

This primer is only for `use_mobilevit=false`.

That means:

- no patch tokens
- no semantic patch aggregation
- no feature-wise distillation term
- no intermediate-stage projection list

The ResNet-18 path is simpler:

- one global hidden feature from the student
- one classifier head
- one shared projection head into the AACD concept space

## 13. Compact Step-by-Step Summary

For a training run:

1. instantiate frozen CLIP, frozen DINOv2, trainable ResNet-18 student, agreement module, concept basis
2. extract CLIP and DINO features on the training set
3. train AE-SVC on CLIP features and keep the encoder
4. train AE-SVC on DINO features and keep the encoder
5. fit CCA between AE-SVC-transformed CLIP and DINO features
6. initialize teacher shared-space prototypes
7. project CLIP text features through CLIP AE-SVC and CLIP CCA
8. initialize the concept basis from projected text features
9. calibrate per-concept gate strengths from teacher disagreement statistics
10. cache the whole AACD initialization state
11. for each batch, compute frozen CLIP features, frozen DINO features, and ResNet-18 student features
12. transform teacher features with the frozen AE-SVC encoders
13. project teacher features into the shared CCA space
14. project those shared teacher features into concept space
15. compute per-concept agreement gates
16. fuse teacher concept activations into a shared teacher target
17. project the student into shared space and then concept space
18. project the ground-truth text feature into concept space
19. compute classification, shared KD, text KD, anchoring, and orthogonality losses
20. update the trainable student and concept parameters

## 14. Minimal Mental Model

If you want the shortest correct mental model, it is this:

- CLIP and DINO are frozen teachers
- AE-SVC cleans each teacher feature space
- CCA aligns the cleaned teacher spaces
- text initializes a semantic concept basis on top of that aligned space
- teacher disagreement is converted into soft concept-wise confidence weights
- the ResNet-18 student is trained to match reliable concepts, not every teacher signal equally

## 15. Useful File Map

- `src/models/aacd_module.py`: initialization pipeline, caching, AE-SVC training, CCA fitting, Lightning loop
- `src/models/components/aacd_campus.py`: batch forward graph for AACD
- `src/models/components/ae_svc.py`: AE-SVC architecture and loss
- `src/models/components/cca_module.py`: offline CCA solver
- `src/models/components/agreement.py`: shared-space projection and diagnostic agreement
- `src/models/components/concept_basis.py`: concept projection, gating, anchoring, orthogonality
- `src/models/components/aacd_criterion.py`: final loss
- `src/models/components/campus.py`: ResNet-18 student wrapper and CLIP teacher wrapper
- `src/models/components/dino_teacher.py`: DINOv2 wrapper
