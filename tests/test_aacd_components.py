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
        self.model = type("ClipStub", (), {"logit_scale": nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32), requires_grad=False)})()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]))

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        patch_tokens = torch.tensor(
            [[[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]],
            device=x.device,
        ).repeat(batch_size, 1, 1)
        gap_features = torch.zeros(batch_size, 4, device=x.device)
        gap_logits = -torch.ones(batch_size, 3, device=x.device)
        intermediates = [torch.ones(batch_size, 3, device=x.device)]
        return patch_tokens, gap_features, gap_logits, intermediates


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
    w, z_shared, delta, clip_proj_t, dino_proj_t = agreement(
        torch.tensor(clip[:5], dtype=torch.float32),
        torch.tensor(dino[:5], dtype=torch.float32),
    )

    assert clip_proj_t.shape == (5, 2)
    assert dino_proj_t.shape == (5, 2)
    assert z_shared.shape == (5, 2)
    assert w.shape == (5,)
    assert delta.shape == (5,)
    assert torch.isfinite(clip_proj_t).all()
    assert torch.isfinite(dino_proj_t).all()
    assert_close(clip_proj_t, torch.tensor(expected_clip[:5], dtype=torch.float32), atol=1e-5, rtol=1e-5)
    assert_close(dino_proj_t, torch.tensor(expected_dino[:5], dtype=torch.float32), atol=1e-5, rtol=1e-5)


def test_patch_aggregation_weights_sum_to_one() -> None:
    agg = SemanticAwareAggregation(dim=4)
    tokens = torch.randn(3, 7, 4)
    pooled, weights = agg(tokens)

    assert pooled.shape == (3, 4)
    assert weights.shape == (3, 7)
    assert_close(weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)


def test_text_kd_is_decoupled_from_agreement_weight() -> None:
    criterion = AACDCriterion(
        lambda_shared=0.0,
        lambda_vis=0.0,
        lambda_geom=0.0,
        lambda_feat=0.0,
    )

    hidden = F.normalize(torch.randn(2, 5), dim=1)
    aligned_nlp = F.normalize(torch.randn(3, 5), dim=1)
    clip_img = F.normalize(torch.randn(2, 6), dim=1)
    frozen_nlp = F.normalize(torch.randn(3, 6), dim=1)
    outputs = {
        "logits": torch.randn(2, 3),
        "student_shared": torch.randn(2, 4),
        "shared_target": torch.randn(2, 4),
        "hidden_features": hidden,
        "aligned_img": hidden.clone(),
        "aligned_dino": hidden.clone(),
        "aligned_nlp": aligned_nlp,
        "clip_img_feats": clip_img,
        "frozen_nlp_feats": frozen_nlp,
        "agreement_w": torch.tensor([1.0, 1.0]),
        "delta": torch.tensor([0.1, 0.2]),
        "projected_intermediates": None,
        "clip_logit_scale": torch.tensor(3.0),
    }
    outputs_low = dict(outputs)
    outputs_low["agreement_w"] = torch.tensor([0.05, 0.05])
    labels = torch.tensor([0, 1])

    loss_ref = criterion(outputs, labels)
    loss_low = criterion(outputs_low, labels)

    assert loss_ref["txt"] == loss_low["txt"]
    assert loss_ref["txt_gated"] != loss_low["txt_gated"]


def test_agreement_module_state_dict_reloads_with_fixed_dims() -> None:
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


def test_mobilevit_logits_use_aggregated_hidden_features() -> None:
    model = AACDTeacherStudent.__new__(AACDTeacherStudent)
    nn.Module.__init__(model)
    model.use_mobilevit = True
    model.clip_teacher = DummyTeacher(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
    model.dino_teacher = DummyTeacher(torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32))
    model.student = DummyMobileStudent()
    model.patch_agg = SemanticAwareAggregation(dim=4)
    with torch.no_grad():
        model.patch_agg.gate.weight.zero_()
        model.patch_agg.gate.bias.zero_()
    model.feat_distill = DummyFeatDistill(shared_dim=2)
    model.align_img = nn.Identity()
    model.align_nlp = nn.Identity()
    model.align_dino = nn.Identity()
    model.condensation_shared = nn.Linear(4, 2, bias=False)
    model.agreement = DummyAgreement(shared_dim=2)
    model.frozen_nlp_features = F.normalize(torch.randn(3, 4), dim=1)

    outputs = model(torch.randn(2, 3, 4, 4))

    expected_logits = model.student.classify(outputs["hidden_features"])
    assert_close(outputs["logits"], expected_logits, atol=1e-6, rtol=1e-6)
    assert outputs["gap_logits"] is not None
    assert not torch.allclose(outputs["logits"], outputs["gap_logits"])
