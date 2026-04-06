import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from src.models.components.aacd_campus import AACDTeacherStudent
from src.models.components.aacd_criterion import AACDCriterion
from src.models.components.agreement import AgreementModule
from src.models.components.cca_module import CCAProjection
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
    def __init__(self, shared_dim: int):
        super().__init__()
        self.shared_dim = shared_dim
        self._initialized = False


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


def _base_outputs() -> dict:
    hidden = F.normalize(torch.randn(2, 5), dim=1)
    aligned_nlp = F.normalize(torch.randn(3, 5), dim=1)
    clip_img = F.normalize(torch.randn(2, 6), dim=1)
    frozen_nlp = F.normalize(torch.randn(3, 6), dim=1)
    return {
        'logits': torch.randn(2, 3),
        'student_shared': torch.randn(2, 4),
        'shared_target': torch.randn(2, 4),
        'hidden_features': hidden,
        'aligned_nlp': aligned_nlp,
        'clip_img_feats': clip_img,
        'frozen_nlp_feats': frozen_nlp,
        'agreement_w': torch.tensor([1.0, 1.0]),
        'kd_shared_weight': torch.tensor([1.0, 1.0]),
        'delta': torch.tensor([0.1, 0.2]),
        'clip_top1': torch.tensor([0, 1]),
        'dino_top1': torch.tensor([0, 1]),
        'clip_margin': torch.tensor([0.9, 0.9]),
        'dino_margin': torch.tensor([0.8, 0.8]),
        'agree_top1': torch.tensor([True, True]),
        'clip_margin_lo': torch.tensor(0.3),
        'clip_margin_hi': torch.tensor(0.8),
        'dino_margin_lo': torch.tensor(0.3),
        'dino_margin_hi': torch.tensor(0.8),
        'delta_hi': torch.tensor(0.5),
        'projected_intermediates': None,
        'clip_logit_scale': torch.tensor(3.0),
        'patch_entropy': torch.tensor(0.0),
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
    assert out['shared_target'].shape == (5, 2)
    assert out['agreement_w'].shape == (5,)
    assert out['delta'].shape == (5,)
    assert out['kd_shared_weight'].shape == (5,)
    assert torch.isfinite(out['clip_margin_lo'])
    assert torch.isfinite(out['clip_margin_hi'])
    assert torch.isfinite(out['delta_hi'])
    assert_close(out['clip_proj'], torch.tensor(expected_clip[:5], dtype=torch.float32), atol=1e-5, rtol=1e-5)
    assert_close(out['dino_proj'], torch.tensor(expected_dino[:5], dtype=torch.float32), atol=1e-5, rtol=1e-5)


def test_patch_aggregation_weights_sum_to_one() -> None:
    agg = SemanticAwareAggregation(dim=4)
    tokens = torch.randn(3, 7, 4)
    pooled, weights = agg(tokens)

    assert pooled.shape == (3, 4)
    assert weights.shape == (3, 7)
    assert_close(weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)


def test_text_kd_is_gated_by_clip_correctness_and_margin() -> None:
    criterion = AACDCriterion(lambda_shared=0.0, lambda_geom=0.0, lambda_feat=0.0)
    labels = torch.tensor([0, 1])

    outputs_full = _base_outputs()
    outputs_soft = dict(outputs_full)
    outputs_soft['clip_margin'] = torch.tensor([0.4, 0.4])
    outputs_wrong = dict(outputs_full)
    outputs_wrong['clip_top1'] = torch.tensor([2, 2])

    loss_full = criterion(outputs_full, labels)
    loss_soft = criterion(outputs_soft, labels)
    loss_wrong = criterion(outputs_wrong, labels)

    assert loss_full['txt'] > loss_soft['txt'] > loss_wrong['txt']
    assert loss_full['mean_text_kd_weight'] == 1.0
    assert loss_soft['mean_text_kd_weight'] == 0.5
    assert loss_wrong['mean_text_kd_weight'] == 0.0


def test_shared_loss_respects_hard_and_soft_gates() -> None:
    criterion = AACDCriterion(lambda_txt=0.0, lambda_geom=0.0, lambda_feat=0.0)
    labels = torch.tensor([0, 1])

    outputs_full = _base_outputs()
    outputs_soft = dict(outputs_full)
    outputs_off = dict(outputs_full)
    outputs_soft['kd_shared_weight'] = torch.tensor([0.3, 0.3])
    outputs_off['kd_shared_weight'] = torch.tensor([0.0, 0.0])

    loss_full = criterion(outputs_full, labels)
    loss_soft = criterion(outputs_soft, labels)
    loss_off = criterion(outputs_off, labels)

    assert loss_full['shared'] > loss_soft['shared'] > loss_off['shared']
    assert loss_off['label_only_frac'] == 1.0


def test_agreement_module_state_dict_reloads_with_threshold_buffers() -> None:
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
    assert_close(restored.clip_margin_lo, agreement.clip_margin_lo)
    assert_close(restored.clip_margin_hi, agreement.clip_margin_hi)
    assert_close(restored.dino_margin_lo, agreement.dino_margin_lo)
    assert_close(restored.dino_margin_hi, agreement.dino_margin_hi)
    assert_close(restored.delta_hi, agreement.delta_hi)


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


def test_mobilevit_logits_use_aggregated_hidden_features() -> None:
    model = AACDTeacherStudent.__new__(AACDTeacherStudent)
    nn.Module.__init__(model)
    model.use_mobilevit = True
    model.register_buffer('_clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
    model.register_buffer('_clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
    model.register_buffer('_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    model.register_buffer('_imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    model.clip_teacher = DummyTeacher(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
    model.dino_teacher = DummyTeacher(torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32))
    model.student = DummyMobileStudent()
    model.patch_agg = SemanticAwareAggregation(dim=4)
    with torch.no_grad():
        model.patch_agg.gate.weight.zero_()
        model.patch_agg.gate.bias.zero_()
    model.feat_distill = DummyFeatDistill(shared_dim=2)
    model.align_nlp = nn.Identity()
    model.condensation_shared = nn.Linear(4, 2, bias=False)
    model.agreement = DummyAgreement(shared_dim=2)
    model.frozen_nlp_features = F.normalize(torch.randn(3, 4), dim=1)

    outputs = model(torch.randn(2, 3, 4, 4))

    expected_logits = model.student.classify(outputs['hidden_features'])
    assert_close(outputs['logits'], expected_logits, atol=1e-6, rtol=1e-6)
    assert outputs['gap_logits'] is not None
    assert not torch.allclose(outputs['logits'], outputs['gap_logits'])
