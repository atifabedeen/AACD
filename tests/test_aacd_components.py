import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from src.models.components.ae_svc import AE_SVC
from src.models.components.aacd_campus import AACDTeacherStudent
from src.models.components.aacd_criterion import AACDCriterion
from src.models.components.agreement import AgreementModule
from src.models.components.cca_module import CCAProjection
from src.models.components.concept_basis import ConceptBasis
from src.models.components.patch_aggregation import SemanticAwareAggregation


class DummyTeacher(nn.Module):
    def __init__(self, feature: torch.Tensor):
        super().__init__()
        self.feature = feature
        self.last_input = None
        self.model = type(
            'ClipStub',
            (),
            {'logit_scale': nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32), requires_grad=False)},
        )()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return self.feature.to(x.device).expand(x.size(0), -1)


class DummyAgreement(nn.Module):
    """Minimal stub for tests that bypass full initialization."""
    def __init__(self, shared_dim: int):
        super().__init__()
        self.shared_dim = shared_dim
        self._initialized = False
        self.register_buffer('_mean_c', torch.zeros(4))
        self.register_buffer('_mean_d', torch.zeros(4))
        self.register_buffer('_A', torch.eye(shared_dim, 4))
        self.register_buffer('_B', torch.eye(shared_dim, 4))

    @property
    def mu_C(self):
        return self._mean_c

    @property
    def mu_D(self):
        return self._mean_d

    @property
    def cca_A(self):
        return self._A

    @property
    def cca_B(self):
        return self._B


class DummyFeatDistill(nn.Module):
    def __init__(self, shared_dim: int):
        super().__init__()
        self.shared_dim = shared_dim

    def project(self, intermediates: list[torch.Tensor]) -> list[torch.Tensor]:
        batch_size = intermediates[0].size(0)
        device = intermediates[0].device
        return [torch.zeros(batch_size, self.shared_dim, device=device)]


class DummyMobileStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 4
        self.classifier = nn.Linear(4, 3, bias=False)
        self.last_input = None
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]))

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor):
        self.last_input = x.detach().clone()
        batch_size = x.size(0)
        patch_tokens = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]],
            device=x.device,
        ).repeat(batch_size, 1, 1)
        gap_features = torch.zeros(batch_size, 4, device=x.device)
        gap_logits = -torch.ones(batch_size, 3, device=x.device)
        intermediates = [torch.ones(batch_size, 3, device=x.device)]
        return patch_tokens, gap_features, gap_logits, intermediates


def _base_outputs(num_concepts: int = 4) -> dict:
    """Build a mock outputs dict matching the upgraded AACD forward signature."""
    B = 2
    K = num_concepts
    return {
        'logits': torch.randn(B, 3),
        'student_shared': torch.randn(B, 4),
        'student_concept_acts': torch.randn(B, K),
        'shared_target': torch.randn(B, K),
        'per_concept_gate': torch.ones(B, K),
        'text_concept_targets': torch.randn(B, K),
        'hidden_features': F.normalize(torch.randn(B, 5), dim=1),
        'aligned_nlp': F.normalize(torch.randn(3, 5), dim=1),
        'clip_img_feats': F.normalize(torch.randn(B, 6), dim=1),
        'dino_img_feats': F.normalize(torch.randn(B, 6), dim=1),
        'frozen_nlp_feats': F.normalize(torch.randn(3, 6), dim=1),
        'agree_top1': torch.tensor([True, True]),
        'mean_delta': 0.1,
        'concept_anchoring_loss': torch.tensor(0.01),
        'concept_orth_loss': torch.tensor(0.001),
        'projected_intermediates': None,
        'clip_logit_scale': torch.tensor(3.0),
        'patch_entropy': torch.tensor(0.0),
        'gap_features': None,
        'gap_logits': None,
    }


def test_cca_projection_and_agreement_use_centered_features() -> None:
    rng = np.random.default_rng(0)
    clip = rng.normal(size=(20, 4))
    dino = clip @ rng.normal(size=(4, 3)) + 0.05 * rng.normal(size=(20, 3))

    cca = CCAProjection(dim_c=4, dim_d=3, s=2, reg=1e-3)
    cca.fit(clip, dino)

    clip_proj = cca.project_clip(clip)
    dino_proj = cca.project_dino(dino)
    expected_clip = (clip - cca.mean_c[None, :]) @ cca.A_s.T
    expected_dino = (dino - cca.mean_d[None, :]) @ cca.B_s.T
    assert_close(torch.tensor(clip_proj), torch.tensor(expected_clip), atol=1e-6, rtol=1e-6)
    assert_close(torch.tensor(dino_proj), torch.tensor(expected_dino), atol=1e-6, rtol=1e-6)

    labels = torch.tensor([0] * 10 + [1] * 10)
    agreement = AgreementModule(num_classes=2, shared_dim=2)
    agreement.initialize(
        cca,
        torch.tensor(clip, dtype=torch.float32),
        torch.tensor(dino, dtype=torch.float32),
        labels,
    )
    out = agreement(
        torch.tensor(clip[:5], dtype=torch.float32),
        torch.tensor(dino[:5], dtype=torch.float32),
    )

    assert out['clip_proj'].shape == (5, 2)
    assert out['dino_proj'].shape == (5, 2)
    assert out['agree_top1'].shape == (5,)
    assert_close(out['clip_proj'], torch.tensor(expected_clip[:5], dtype=torch.float32), atol=1e-5, rtol=1e-5)
    assert_close(out['dino_proj'], torch.tensor(expected_dino[:5], dtype=torch.float32), atol=1e-5, rtol=1e-5)


def test_patch_aggregation_weights_sum_to_one() -> None:
    agg = SemanticAwareAggregation(dim=4)
    tokens = torch.randn(3, 7, 4)
    pooled, weights = agg(tokens)

    assert pooled.shape == (3, 4)
    assert weights.shape == (3, 7)
    assert_close(weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)


def test_concept_shared_kd_uses_per_concept_gate() -> None:
    """Per-concept gates should scale the shared KD loss."""
    criterion = AACDCriterion(lambda_shared=1.0, lambda_txt=0.0, lambda_feat=0.0)
    labels = torch.tensor([0, 1])

    outputs_full = _base_outputs()
    outputs_gated = _base_outputs()
    # Same targets/predictions but zero out the gate
    outputs_gated['student_concept_acts'] = outputs_full['student_concept_acts'].clone()
    outputs_gated['shared_target'] = outputs_full['shared_target'].clone()
    outputs_gated['per_concept_gate'] = torch.zeros(2, 4)

    loss_full = criterion(outputs_full, labels)
    loss_gated = criterion(outputs_gated, labels)

    assert loss_full['shared'] > loss_gated['shared']
    assert loss_gated['shared'] == 0.0


def test_unified_text_kd_uses_per_concept_gate() -> None:
    """Text KD should use the same per-concept gates."""
    criterion = AACDCriterion(lambda_shared=0.0, lambda_txt=1.0, lambda_feat=0.0)
    labels = torch.tensor([0, 1])

    outputs_full = _base_outputs()
    outputs_gated = _base_outputs()
    outputs_gated['student_concept_acts'] = outputs_full['student_concept_acts'].clone()
    outputs_gated['text_concept_targets'] = outputs_full['text_concept_targets'].clone()
    outputs_gated['per_concept_gate'] = torch.zeros(2, 4)

    loss_full = criterion(outputs_full, labels)
    loss_gated = criterion(outputs_gated, labels)

    assert loss_full['txt'] > loss_gated['txt']
    assert loss_gated['txt'] == 0.0


def test_agreement_module_state_dict_reloads() -> None:
    rng = np.random.default_rng(1)
    clip = rng.normal(size=(12, 4))
    dino = rng.normal(size=(12, 3))
    labels = torch.tensor([0] * 6 + [1] * 6)

    cca = CCAProjection(dim_c=4, dim_d=3, s=2, reg=1e-3)
    cca.fit(clip, dino)

    agreement = AgreementModule(num_classes=2, shared_dim=2, clip_dim=4, dino_dim=3)
    agreement.initialize(
        cca,
        torch.tensor(clip, dtype=torch.float32),
        torch.tensor(dino, dtype=torch.float32),
        labels,
    )

    restored = AgreementModule(num_classes=2, shared_dim=2, clip_dim=4, dino_dim=3)
    restored.load_state_dict(agreement.state_dict())

    assert restored._initialized
    assert_close(restored._A, agreement._A)
    assert_close(restored._B, agreement._B)
    assert_close(restored._mean_c, agreement._mean_c)
    assert_close(restored._mean_d, agreement._mean_d)
    assert_close(restored.prototypes, agreement.prototypes)


def test_preprocessing_branches_use_different_normalization() -> None:
    model = AACDTeacherStudent.__new__(AACDTeacherStudent)
    nn.Module.__init__(model)
    model.register_buffer('_clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
    model.register_buffer('_clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
    model.register_buffer('_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    model.register_buffer('_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    x = torch.full((1, 3, 2, 2), 0.5)
    clip_x = model.preprocess_for_clip(x)
    imagenet_x = model.preprocess_for_imagenet(x)

    assert not torch.allclose(clip_x, imagenet_x)


def test_ae_svc_produces_regularized_features() -> None:
    """AE-SVC should produce features with ~zero mean and ~unit variance."""
    torch.manual_seed(42)
    input_dim = 16
    latent_dim = 16
    n_samples = 256

    features = torch.randn(n_samples, input_dim) * 3.0 + 2.0  # non-zero mean, high variance
    model = AE_SVC(input_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(30):
        z, x_rec = model(features)
        loss, _ = AE_SVC.compute_losses(z, features, x_rec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z, _ = model(features)
    # After training, mean should be close to 0 and variance close to 1
    assert z.mean().abs() < 0.5, f"Mean too large: {z.mean().item()}"
    assert (z.var(dim=0).mean() - 1.0).abs() < 1.0, f"Variance too far from 1: {z.var(dim=0).mean().item()}"


def test_concept_basis_gating_and_calibration() -> None:
    """ConceptBasis should produce per-concept gates and calibrate alpha from data."""
    torch.manual_seed(42)
    shared_dim = 8
    num_concepts = 4

    cb = ConceptBasis(shared_dim=shared_dim, num_concepts=num_concepts)
    # Initialize with random projected text
    projected = F.normalize(torch.randn(shared_dim, num_concepts), dim=0)
    cb.init_from_projected_text(projected)

    # Forward
    c_tilde = torch.randn(10, shared_dim)
    d_tilde = torch.randn(10, shared_dim)
    z_c, z_d, gate, shared_target = cb(c_tilde, d_tilde)

    assert z_c.shape == (10, num_concepts)
    assert z_d.shape == (10, num_concepts)
    assert gate.shape == (10, num_concepts)
    assert shared_target.shape == (10, num_concepts)
    assert (gate >= 0).all() and (gate <= 1).all()

    # Calibrate
    cb.calibrate(c_tilde, d_tilde)
    assert cb.alpha.shape == (num_concepts,)
    assert (cb.alpha > 0).all()

    # Anchoring loss
    assert cb.anchoring_loss().item() >= 0
    # Orthogonality loss
    assert cb.orthogonality_loss().item() >= 0


def test_concept_basis_project_to_concepts() -> None:
    """project_to_concepts should use normalized A."""
    torch.manual_seed(0)
    shared_dim = 8
    num_concepts = 4
    cb = ConceptBasis(shared_dim=shared_dim, num_concepts=num_concepts)
    projected = F.normalize(torch.randn(shared_dim, num_concepts), dim=0)
    cb.init_from_projected_text(projected)

    features = torch.randn(5, shared_dim)
    concept_acts = cb.project_to_concepts(features)
    assert concept_acts.shape == (5, num_concepts)

    # Should match manual computation
    A_norm = F.normalize(cb.A, dim=0)
    expected = features @ A_norm
    assert_close(concept_acts, expected, atol=1e-6, rtol=1e-6)


def test_criterion_total_loss_includes_all_components() -> None:
    """Total loss should include cls, shared, txt, and concept reg."""
    criterion = AACDCriterion(
        lambda_shared=0.3,
        lambda_txt=0.2,
        lambda_anchor=0.01,
        lambda_orth=0.001,
    )
    labels = torch.tensor([0, 1])
    outputs = _base_outputs()

    loss_dict = criterion(outputs, labels)
    assert 'total' in loss_dict
    assert 'cls' in loss_dict
    assert 'shared' in loss_dict
    assert 'txt' in loss_dict
    assert 'concept_reg' in loss_dict
    assert loss_dict['total'].requires_grad
