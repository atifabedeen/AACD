# Agreement-Aware Correlation-Guided Distillation With Code-Grounded Teacher Conflict Handling

## Summary
This document rewrites the AACD idea to match the current `VL2Lite` implementation rather than the earlier conceptual proposal. The implemented method uses two frozen teachers, CLIP and DINOv2, fits a shared CCA subspace from cached training-set teacher features, initializes a prototype-based agreement module in that shared space, and trains a lightweight student with a combination of classification loss, shared-space distillation, CLIP text distillation, geometry regularization, and optional feature-wise distillation. The implementation is narrower and cleaner than the original draft: disagreement is handled through calibrated hard/soft gating rather than only a smooth weight, MobileViT is optional rather than default, and the geometry term is covariance-based rather than a Gram-to-identity constraint.

## Problem Setup And Why CLIP/DINO Conflict Matters
AACD assumes two strong but heterogeneous frozen teachers:

- CLIP contributes semantic structure through aligned image-text representations.
- DINOv2 contributes discriminative visual structure learned from self-supervised image modeling.

These teachers are useful for fine-grained recognition, but they do not naturally live in the same feature space and they do not always make compatible decisions. Directly distilling both without reconciliation would mix heterogeneous representations and allow conflicting supervision to reach the student. The implemented AACD pipeline addresses this in two stages:

1. It aligns CLIP and DINO features into a shared correlated space using offline CCA.
2. It measures agreement in prototype-similarity space and gates distillation when the two teachers disagree strongly.

This turns teacher conflict into an explicit training signal instead of treating all teacher outputs as equally reliable.

## Implemented AACD Pipeline
The current AACD path is built around `src/models/components/aacd_campus.py`, `src/models/components/agreement.py`, `src/models/components/cca_module.py`, and `src/models/components/aacd_criterion.py`.

### 1. Frozen teacher setup
- CLIP image features come from the existing `TeacherNet` wrapper.
- DINOv2 image features come from `DINOv2Teacher`.
- CLIP text features are precomputed from dataset class prompts and kept frozen.

### 2. Offline shared-space fitting
Before normal training, `AACDModule.setup("fit")` loads or extracts full training-set CLIP and DINO features, then fits a CCA model:

- cached features are stored under `cache_dir`
- `cca_s` determines the retained shared dimension
- `shared_dim` in the model must match `cca_s`

The fitted CCA object provides the projection matrices and centering statistics used later by the agreement module.

### 3. Agreement-module initialization
After CCA fitting, the agreement module is initialized once using the training-set teacher features and labels:

- CLIP and DINO features are projected into the shared space
- the projected features are averaged to form a shared teacher representation
- per-class prototypes are built from the averaged shared features
- prototype similarities are used to calibrate teacher margins and the disagreement threshold `delta_hi`

This makes the disagreement logic data-dependent rather than heuristic-only.

### 4. Student forward path
At training time the model computes:

- CLIP image features
- DINO image features
- frozen CLIP text features
- aligned CLIP text features in the student space
- student hidden features and logits
- student shared representation through a condensation MLP

Two student paths exist:

- Default path: ResNet-based student from the original VL2Lite setup
- Optional path: MobileViT student with semantic patch aggregation and optional multi-scale feature distillation

The shipped CUB experiment uses the ResNet-18 path by default.

### 5. Agreement-aware shared KD
When the agreement module is initialized, it:

- projects CLIP and DINO features into the shared CCA space
- computes prototype similarities for each teacher
- measures disagreement by the distance between the two similarity vectors
- assigns each sample to full KD, soft KD, or label-only supervision

The shared teacher target for distillation is the mean of the projected CLIP and projected DINO features.

### 6. Additional training signals
The total objective combines:

- supervised classification on student logits
- agreement-gated shared-space KD
- CLIP text KD gated by CLIP correctness and margin
- geometry preservation on the student shared representation
- optional feature-wise KD from intermediate student stages

## Code-Faithful Mathematical Formulation

### Teacher features and CCA alignment
For an image `x_i`, let:

```math
c_i = f_C(x_i), \quad d_i = f_D(x_i)
```

CCA is fit offline on the full training-set teacher features and returns centered projections:

```math
\tilde{c}_i = A(c_i - \mu_C), \quad \tilde{d}_i = B(d_i - \mu_D)
```

where `A` and `B` are the retained CCA projection matrices and the retained dimension is:

```math
s = \texttt{cca\_s} = \texttt{shared\_dim}
```

### Shared target
The code uses the mean projected teacher feature as the shared distillation target:

```math
z_i^{\text{shared}} = \frac{1}{2}(\tilde{c}_i + \tilde{d}_i)
```

The student maps its hidden feature `h_i` to a shared representation through a condensation head:

```math
\hat{z}_i = g_{\text{shared}}(h_i)
```

### Prototype-similarity agreement space
After initialization, each class `k` has a normalized prototype `p_k` built from training-set shared teacher features. For each sample, the module computes:

```math
S_C(i,k) = \cos(\tilde{c}_i, p_k), \quad S_D(i,k) = \cos(\tilde{d}_i, p_k)
```

and the disagreement score:

```math
\delta_i = \left\| S_C(i,:) - S_D(i,:) \right\|_2
```

The module also computes top-1 classes and margins from both similarity vectors.

### Hard/soft/zero KD gating
The code defines a continuous agreement score:

```math
w_i^{\text{agree}} = \exp(-\alpha \delta_i)
```

but the active shared KD path is governed by discrete gates:

```math
\text{full\_shared\_kd}_i
=
[\text{agree\_top1}_i]
[\text{clip\_margin}_i \ge \tau_C^{hi}]
[\text{dino\_margin}_i \ge \tau_D^{hi}]
[\delta_i < \tau_\delta]
```

```math
\text{soft\_shared\_kd}_i
=
[\text{agree\_top1}_i]
[\delta_i < \tau_\delta]
[ \neg \text{full\_shared\_kd}_i ]
```

```math
\text{strong\_disagree}_i
=
[ \neg \text{agree\_top1}_i ] \lor [\delta_i \ge \tau_\delta]
```

with the sample-wise shared KD weight:

```math
w_i^{\text{shared}} =
\begin{cases}
1.0 & \text{if full\_shared\_kd}_i \\
0.3 & \text{if soft\_shared\_kd}_i \\
0.0 & \text{otherwise}
\end{cases}
```

The implemented shared KD loss is:

```math
\mathcal{L}_{shared}
=
\frac{1}{B}\sum_i
w_i^{\text{shared}}
\cdot
\text{MSE}(\hat{z}_i, z_i^{\text{shared}})
```

### CLIP text KD
CLIP text features are aligned into the student feature space. The student and CLIP teacher then induce logits over class text features:

```math
\ell_i^{student}
=
\gamma \, h_i \, \bar{t}^{\top} / T
,\quad
\ell_i^{clip}
=
\gamma \, c_i \, t^{\top} / T
```

where `t` are frozen CLIP text features, `\bar{t}` are aligned text features, `T` is the KD temperature, and `\gamma` is the CLIP logit scale.

This text KD is not agreement-driven. Its sample weight depends only on whether CLIP predicts the ground-truth label and how large the CLIP margin is:

```math
w_i^{text} \in \{0, 0.5, 1.0\}
```

The loss is KL divergence between the student text logits and CLIP text logits, weighted by `w_i^{text}`.

### Geometry preservation
The implemented geometry loss regularizes the batch statistics of the student shared representation, not its pairwise Gram matrix:

```math
z_i = \hat{z}_i, \quad z_i^c = z_i - \mu_z
```

```math
\Sigma_z = \frac{1}{B}(Z^c)^\top Z^c
```

```math
\mathcal{L}_{geom}
=
\mathcal{L}_{cov}
+
15 \mathcal{L}_{var}
+
\mathcal{L}_{mean}
```

where:

- `\mathcal{L}_{cov}` penalizes off-diagonal covariance
- `\mathcal{L}_{var}` pushes diagonal covariance toward one
- `\mathcal{L}_{mean}` keeps feature means near zero

### Optional feature-wise KD
If the MobileViT branch is enabled, intermediate student features are projected into the shared dimension and matched to the same shared target:

```math
\mathcal{L}_{feat}
=
\frac{1}{L}\sum_{l=1}^{L}
\frac{1}{B}\sum_i
w_i^{shared}
\cdot
\| q_l(f_i^{(l)}) - z_i^{shared} \|_1
```

This term exists in code but is disabled by default in the main AACD config.

## Training Objective And Scheduling
The implemented criterion uses a dynamic schedule rather than fixed weights throughout training.

Let:

```math
p = \frac{\text{epoch}}{\max(\text{max\_epochs}, 1)}
```

Then:

```math
w_{cls}(p) = \lambda_{cls} + p(1 - \lambda_{cls})
```

```math
w_{kd}(p) = 1 - 0.5p
```

The final implemented loss is:

```math
\mathcal{L}
=
w_{cls}(p)\mathcal{L}_{cls}
+
w_{kd}(p)\lambda_{shared}\mathcal{L}_{shared}
+
w_{kd}(p)\lambda_{txt}\mathcal{L}_{txt}
+
\lambda_{geom}\mathcal{L}_{geom}
+
w_{kd}(p)\lambda_{feat}\mathcal{L}_{feat}
```

Interpretation:

- classification gradually becomes dominant as training progresses
- KD terms remain active but are mildly decayed over time
- geometry regularization stays fixed throughout training

In the default AACD config:

- `lambda_shared = 0.3`
- `lambda_txt = 0.2`
- `lambda_geom = 0.1`
- `lambda_feat = 0.0`
- `lambda_cls = 0.01`

## What Changed From The Earlier Proposal
The current implementation preserves the main intuition of agreement-aware distillation, but it is more concrete and narrower than the earlier proposal.

### 1. No per-dimension shared-signal masking after projection
The earlier draft described selecting only specific correlated dimensions after projection. The implemented code does not do post-CCA per-dimension masking during training. Instead, it keeps the retained shared subspace of dimension `s` and uses the averaged projected teacher feature as the target.

### 2. Shared KD is driven by calibrated gates, not only a smooth weight
The earlier draft emphasized a continuous agreement weighting term. The implementation computes `agreement_w = exp(-alpha * delta)`, but the active shared KD loss is primarily controlled by the discrete `kd_shared_weight` values `1.0`, `0.3`, or `0.0`, derived from agreement and calibrated margin thresholds.

### 3. Text KD is CLIP-gated, not dual-teacher-gated
The conceptual proposal treated teacher agreement as a general reliability mechanism. In code, CLIP text KD is gated only by CLIP top-1 correctness and CLIP margin thresholds. DINO does not directly gate the text KD term.

### 4. Geometry preservation is covariance regularization
The earlier idea described a normalized Gram-matrix or orthogonality-style constraint. The implemented geometry loss instead regularizes the mean, variance, and covariance of the student shared features.

### 5. MobileViT is optional, not the default AACD path
The earlier pipeline was written as if patch-token aggregation were central. In the shipped CUB experiment config, the default student is `resnet18` with `use_mobilevit: false`. MobileViT, semantic patch aggregation, and feature-wise distillation are implemented as an optional branch.

### 6. Feature-wise KD exists but is disabled by default
The earlier draft presented feature-wise KD as part of the main method. In the current config it is implemented but turned off by default through `lambda_feat: 0.0`.

### 7. Initialization workflow is operationally important
The current system requires offline cache loading or feature extraction, followed by CCA fitting and agreement-module initialization. This initialization is not just an implementation detail; it is part of the training contract for AACD.

## How Prior-Paper Ideas Appear In This Implementation
The code reflects a combination of prior-paper ideas, but only partially and only where they are concretely implemented.

### CSA-style shared-space alignment
The closest code-grounded influence is the use of CCA to reconcile heterogeneous CLIP and DINO teacher spaces before combining them. The implementation uses offline centering, covariance estimation, whitening, and SVD to obtain the correlated shared subspace.

### Similarity-space agreement reasoning
Teacher conflict is not resolved by arbitrarily picking one teacher or averaging logits directly. Instead, the code compares each teacher's similarity pattern against class prototypes in the shared space and uses that comparison to decide whether shared KD should be trusted.

### Geometry-preserving compression
The student shared representation is regularized to maintain stable distributional structure under compression. This is not a direct reproduction of an external method, but it clearly follows the idea that compressed embeddings should preserve useful geometry rather than collapse or entangle dimensions.

### NanoSD-like feature-level distillation spirit
The optional intermediate projector heads follow the general idea that compact students can benefit from layer-wise latent supervision. In this repo that idea appears as an optional multi-scale projection module rather than a full always-on feature distillation regime.

Overall, the implementation should be read as a principled hybrid: CCA-based teacher reconciliation, prototype-based agreement gating, CLIP-driven linguistic transfer, and lightweight geometry regularization under a frozen multi-teacher setup.

## Training And Configuration Notes
The main AACD configuration is defined in `configs/model/aacd.yaml` and `configs/experiment/aacd_cub.yaml`.

### Default experiment choices
- dataset: CUB-200-2011 via `experiment=aacd_cub`
- student: `resnet18`
- DINO teacher: `dinov2_vits14`
- shared dimension: `128`
- `agreement_alpha = 2.0`
- `use_mobilevit = false`
- `lambda_feat = 0.0`

### Important configuration contracts
- `model.net.shared_dim` must match `model.cca_s`
- the agreement module must be initialized before standard AACD evaluation or checkpoint-based test use
- checkpoint reload relies on saved agreement buffers, including the CCA projections, means, prototypes, and threshold buffers

### Operational workflow
The intended flow is:

1. prepare the dataset
2. create or load the cached teacher features
3. fit CCA and initialize the agreement module
4. train the student
5. optionally reload the trained checkpoint for test

## Validation Status And Implementation Caveats
This writeup is grounded in the current AACD code path and in the focused component tests in `tests/test_aacd_components.py`, which confirm:

- CCA projection uses centered teacher features
- agreement-module state reload preserves projection and threshold buffers
- semantic patch aggregation produces normalized attention weights
- text KD is gated by CLIP correctness and CLIP margin
- shared KD respects hard, soft, and label-only regimes
- the MobileViT branch classifies aggregated hidden features rather than the raw GAP baseline

### Caveats
- The runbook and the training module do not use exactly the same cache filename convention: the runbook references `aacd_features_<data>.pth`, while `AACDModule.setup()` looks for `aacd_features_v2_<data>.pth`. The module behavior should be treated as authoritative.
- The feature-cache creation path is operationally important for stable AACD startup, especially in DDP settings.
- `agreement_w` is exposed in model outputs and useful for analysis, but it should not be described as the main active KD weighting term in the current loss.
- Any description of the implemented method should avoid claiming post-CCA dimension masking, always-on MobileViT, or Gram-identity geometry preservation, because those are not what the current code does.
