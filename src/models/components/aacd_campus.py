"""
Agreement-Aware Correlation-Guided Distillation (AACD) model — KUEA kernel refactor.

Replaces CCA + ConceptBasis concept-space distillation with KUEA-style
pairwise kernel geometry losses.  Frozen CLIP/DINO teachers and optional
AE-SVC preprocessing are kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components.ae_svc import AE_SVC
from src.models.components.campus import TeacherNet, StudentNet
from src.models.components.dino_teacher import DINOv2Teacher
from src.models.components.kernel_ops import build_kernel

feature_norm = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-10)


class AACDTeacherStudent(nn.Module):
    def __init__(
        self,
        teacher,
        dino,
        student,
        data_attributes,
        shared_dim: int = 128,
        kernel_name: str = "poly",
        text_kernel_name: str = "cosine",
        kernel_degree: int = 3,
        text_preproc: str = "raw",
        use_ae_svc: bool = True,
        use_mobilevit: bool = False,
    ):
        super().__init__()
        self.use_mobilevit = use_mobilevit
        self.shared_dim = shared_dim
        self.kernel_name = kernel_name
        self.text_kernel_name = text_kernel_name
        self.kernel_degree = kernel_degree
        self.text_preproc = text_preproc
        self.use_ae_svc = use_ae_svc

        # ---- Normalization buffers ----
        self.register_buffer(
            "_clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        # ---- Teachers ----
        self.clip_teacher = TeacherNet(teacher)
        clip_dim = self.clip_teacher.last_features_dim
        self.dino_teacher = DINOv2Teacher(dino.model_name)
        dino_dim = self.dino_teacher.output_dim
        self.clip_feature_dim = clip_dim
        self.dino_feature_dim = dino_dim

        # ---- Student ----
        self.student = StudentNet(student, data_attributes.class_num, use_teacher=True)
        student_dim = self.student.num_features
        self.patch_agg = None
        self.feat_distill = None

        # ---- Student projection (Linear -> ReLU -> Linear) ----
        self.student_proj = nn.Sequential(
            nn.Linear(student_dim, student_dim),
            nn.ReLU(),
            nn.Linear(student_dim, shared_dim),
        )

        # ---- Kernel parameters ----
        if kernel_name == "poly":
            # Trainable student kernel params
            self.kernel_gamma_student = nn.Parameter(
                torch.tensor(1.0 / shared_dim)
            )
            self.kernel_coef0_student = nn.Parameter(torch.tensor(1.0))
            # Fixed teacher kernel params (plain floats, not nn.Parameter)
            self.kernel_gamma_clip: float = 1.0 / shared_dim
            self.kernel_coef0_clip: float = 1.0
        else:
            self.kernel_gamma_student = None
            self.kernel_coef0_student = None
            self.kernel_gamma_clip = None
            self.kernel_coef0_clip = None

        # ---- AE-SVC encoders (pre-registered for checkpoint restore) ----
        self.clip_ae_encoder = self._build_ae_encoder(clip_dim)
        self.dino_ae_encoder = self._build_ae_encoder(dino_dim)
        self._freeze_module(self.clip_ae_encoder)
        self._freeze_module(self.dino_ae_encoder)
        self.register_buffer(
            "_clip_ae_ready", torch.tensor(False, dtype=torch.bool)
        )
        self.register_buffer(
            "_dino_ae_ready", torch.tensor(False, dtype=torch.bool)
        )

        # ---- Frozen text embeddings ----
        self.data_attributes = data_attributes
        self.frozen_nlp_features = self._get_frozen_nlp_features(data_attributes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ae_encoder(feature_dim: int) -> nn.Module:
        return AE_SVC(feature_dim, feature_dim).encoder

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    @property
    def ae_encoders_ready(self) -> bool:
        return bool(self._clip_ae_ready.item() and self._dino_ae_ready.item())

    @property
    def initialized_for_distillation(self) -> bool:
        if not self.use_ae_svc:
            return True
        return self.ae_encoders_ready

    def set_ae_encoders(
        self, clip_encoder: nn.Module, dino_encoder: nn.Module
    ) -> None:
        """Load trained AE-SVC weights into the pre-registered frozen encoders."""
        self.clip_ae_encoder.load_state_dict(clip_encoder.state_dict())
        self.dino_ae_encoder.load_state_dict(dino_encoder.state_dict())
        self._freeze_module(self.clip_ae_encoder)
        self._freeze_module(self.dino_ae_encoder)
        self._clip_ae_ready.fill_(True)
        self._dino_ae_ready.fill_(True)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _normalize(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return (x - mean.to(x.device)) / std.to(x.device)

    def preprocess_for_clip(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(x, self._clip_mean, self._clip_std)

    def preprocess_for_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize(x, self._imagenet_mean, self._imagenet_std)

    def get_branch_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_x = self.preprocess_for_clip(x)
        imagenet_x = self.preprocess_for_imagenet(x)
        return clip_x, imagenet_x, imagenet_x

    # ------------------------------------------------------------------
    # Frozen text features
    # ------------------------------------------------------------------

    def _get_frozen_nlp_features(self, attributes) -> torch.Tensor:
        prompt_tmpl = attributes.prompt_tmpl
        classes_list = list(attributes.classes.values())
        tokens = self.clip_teacher.tokenizer(
            [prompt_tmpl.format(w) for w in classes_list]
        )
        nlp_feats = self.clip_teacher.encode_text(tokens).detach()
        return feature_norm(nlp_feats)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict:
        clip_x, dino_x, student_x = self.get_branch_inputs(x)

        # Teachers (frozen)
        clip_img = self.clip_teacher(clip_x)
        dino_img = self.dino_teacher(dino_x)
        frozen_nlp = self.frozen_nlp_features.to(x.device)

        # Student backbone
        hidden_features, logits = self.student(student_x)

        # Detach teacher features
        clip_feat = clip_img.detach()
        dino_feat = dino_img.detach()

        # Optional AE-SVC encoding
        if self.use_ae_svc and self._clip_ae_ready.item():
            clip_feat = self.clip_ae_encoder(clip_feat)
        if self.use_ae_svc and self._dino_ae_ready.item():
            dino_feat = self.dino_ae_encoder(dino_feat)

        # Student projection
        student_feat = self.student_proj(hidden_features)

        # Build kernel matrices
        K_clip = build_kernel(
            clip_feat,
            self.kernel_name,
            gamma=self.kernel_gamma_clip,
            coef0=self.kernel_coef0_clip,
            degree=self.kernel_degree,
            role="clip",
        )
        K_dino = build_kernel(
            dino_feat,
            self.kernel_name,
            gamma=None,
            coef0=None,
            degree=self.kernel_degree,
            role="dino",
        )
        K_student = build_kernel(
            student_feat,
            self.kernel_name,
            gamma=self.kernel_gamma_student,
            coef0=self.kernel_coef0_student,
            degree=self.kernel_degree,
            role="student",
        )

        # Text kernel (optional, only when labels are provided)
        K_text: torch.Tensor | None = None
        if labels is not None:
            text_feat_gt = frozen_nlp[labels]
            if self.text_preproc == "clip_ae" and self._clip_ae_ready.item():
                text_feat_gt = self.clip_ae_encoder(text_feat_gt)
            K_text = build_kernel(
                text_feat_gt,
                self.text_kernel_name,
                gamma=self.kernel_gamma_clip,
                coef0=self.kernel_coef0_clip,
                degree=self.kernel_degree,
                role="text",
            )

        return {
            "hidden_features": hidden_features,
            "logits": logits,
            "clip_img_feats": clip_img,
            "dino_img_feats": dino_img,
            "student_kernel_feats": student_feat,
            "K_clip": K_clip,
            "K_dino": K_dino,
            "K_student": K_student,
            "K_text": K_text,
            "frozen_nlp_feats": frozen_nlp,
        }
