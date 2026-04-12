import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.ae_svc import AE_SVC
from src.models.components.aacd_campus import AACDTeacherStudent
from src.models.components.aacd_criterion import AACDKernelCriterion, kernel_mse
from src.models.components.kernel_ops import cosine_kernel


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


def _base_kernel_outputs(shared_dim: int = 8) -> dict:
    """Build a mock outputs dict matching the KUEA kernel AACD forward signature."""
    B = 2
    D = shared_dim
    # Build kernel matrices from random features so they are valid PSD matrices
    student_feat = torch.randn(B, D)
    clip_feat = torch.randn(B, D)
    dino_feat = torch.randn(B, D)
    text_feat = torch.randn(B, D)
    return {
        'logits': torch.randn(B, 3),
        'hidden_features': F.normalize(torch.randn(B, 5), dim=1),
        'clip_img_feats': F.normalize(torch.randn(B, 6), dim=1),
        'dino_img_feats': F.normalize(torch.randn(B, 6), dim=1),
        'frozen_nlp_feats': F.normalize(torch.randn(3, 6), dim=1),
        'student_kernel_feats': student_feat,
        'K_student': cosine_kernel(student_feat),
        'K_clip': cosine_kernel(clip_feat),
        'K_dino': cosine_kernel(dino_feat),
        'K_text': cosine_kernel(text_feat),
    }


def test_kernel_mse_is_zero_for_identical_matrices() -> None:
    """kernel_mse should be zero when student and teacher kernels match."""
    K = cosine_kernel(torch.randn(4, 8))
    assert kernel_mse(K, K).item() == 0.0


def test_kernel_mse_increases_with_divergence() -> None:
    """kernel_mse should grow when student and teacher kernels differ."""
    feat_a = torch.randn(4, 8)
    feat_b = torch.randn(4, 8) * 5.0  # very different features
    K_a = cosine_kernel(feat_a)
    K_b = cosine_kernel(feat_b)
    loss = kernel_mse(K_a, K_b)
    assert loss.item() > 0.0


def test_txt_kernel_loss_is_zero_when_K_text_is_none() -> None:
    """When K_text is None, txt_kernel loss should be 0."""
    criterion = AACDKernelCriterion(lambda_txt=1.0)
    labels = torch.tensor([0, 1])
    outputs = _base_kernel_outputs()
    outputs['K_text'] = None

    loss_dict = criterion(outputs, labels)
    assert loss_dict['txt_kernel'] == 0.0


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


def test_kernel_criterion_total_loss_includes_all_components() -> None:
    """Total loss should include cls, clip_kernel, dino_kernel, and txt_kernel."""
    criterion = AACDKernelCriterion(
        lambda_cls=1.0,
        lambda_clip=0.2,
        lambda_dino=0.6,
        lambda_txt=0.2,
    )
    labels = torch.tensor([0, 1])
    outputs = _base_kernel_outputs()

    loss_dict = criterion(outputs, labels)
    assert 'total' in loss_dict
    assert 'cls' in loss_dict
    assert 'clip_kernel' in loss_dict
    assert 'dino_kernel' in loss_dict
    assert 'txt_kernel' in loss_dict
    assert loss_dict['total'].requires_grad
