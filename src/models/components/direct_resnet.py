from __future__ import annotations

import torch
from torch import nn

from src.models.components.campus import StudentNet


class DirectResNetClassifier(nn.Module):
    """Plain supervised ResNet path without any teacher or distillation branches."""

    def __init__(self, student, data_attributes) -> None:
        super().__init__()
        self.student = StudentNet(student, data_attributes.class_num, use_teacher=False)
        self.data_attributes = data_attributes

        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def preprocess_for_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._imagenet_mean.to(x.device)) / self._imagenet_std.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(self.preprocess_for_imagenet(x))
