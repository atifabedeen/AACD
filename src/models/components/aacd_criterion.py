"""
AACD kernel-based loss function.

Active training loss:
  L = lambda_cls * L_cls
    + lambda_clip * kernel_mse(K_student, K_clip)
    + lambda_dino * kernel_mse(K_student, K_dino)
    + lambda_txt  * kernel_mse(K_student, K_text)   [if K_text is available]
"""

from __future__ import annotations

import torch
import torch.nn as nn


def kernel_mse(K_student: torch.Tensor, K_teacher: torch.Tensor) -> torch.Tensor:
    """Mean squared error between two kernel matrices.

    Args:
        K_student: Student kernel matrix [B, B].
        K_teacher: Teacher kernel matrix [B, B].

    Returns:
        Scalar MSE loss.
    """
    return ((K_student - K_teacher) ** 2).mean()


class AACDKernelCriterion:
    """Kernel-geometry distillation criterion for AACD.

    Replaces the CCA + ConceptBasis concept-space losses with pairwise
    kernel matrix MSE losses between student and each teacher.

    Args:
        lambda_cls: Weight for cross-entropy classification loss.
        lambda_clip: Weight for CLIP kernel MSE loss.
        lambda_dino: Weight for DINO kernel MSE loss.
        lambda_txt: Weight for text kernel MSE loss.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_clip: float = 0.2,
        lambda_dino: float = 0.6,
        lambda_txt: float = 0.2,
    ):
        self.lambda_cls = lambda_cls
        self.lambda_clip = lambda_clip
        self.lambda_dino = lambda_dino
        self.lambda_txt = lambda_txt
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def __call__(
        self,
        outputs: dict,
        labels: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 100,
    ) -> dict:
        """Compute the combined kernel-geometry distillation loss.

        Args:
            outputs: Dict from ``AACDTeacherStudent.forward()`` containing
                ``logits``, ``K_student``, ``K_clip``, ``K_dino``, and
                optionally ``K_text``.
            labels: Ground-truth class labels [B].
            epoch: Current training epoch (unused, kept for API compat).
            max_epochs: Maximum training epochs (unused, kept for API compat).

        Returns:
            Dict with keys ``"total"`` (loss tensor for backward),
            ``"cls"``, ``"clip_kernel"``, ``"dino_kernel"``, ``"txt_kernel"``
            (float scalars for logging).
        """
        # Classification loss
        loss_cls = self.ce(outputs["logits"], labels)

        # Kernel MSE losses
        loss_clip = kernel_mse(outputs["K_student"], outputs["K_clip"])
        loss_dino = kernel_mse(outputs["K_student"], outputs["K_dino"])

        K_text = outputs.get("K_text")
        if K_text is not None:
            loss_txt = kernel_mse(outputs["K_student"], K_text)
        else:
            loss_txt = torch.tensor(0.0, device=loss_cls.device)

        total = (
            self.lambda_cls * loss_cls
            + self.lambda_clip * loss_clip
            + self.lambda_dino * loss_dino
            + self.lambda_txt * loss_txt
        )

        return {
            "total": total,
            "cls": loss_cls.item(),
            "clip_kernel": loss_clip.item(),
            "dino_kernel": loss_dino.item(),
            "txt_kernel": loss_txt.item() if torch.is_tensor(loss_txt) else loss_txt,
        }
