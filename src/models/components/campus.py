import torch
from torch import nn
import torchvision.models as models

try:
    import open_clip
except ImportError:  # pragma: no cover - only needed for teacher-backed runs
    open_clip = None

feature_norm = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-10)

class TeacherNet(nn.Module):
    def __init__(self, teacher):
        super(TeacherNet, self).__init__()
        if open_clip is None:
            raise ImportError("open_clip is required to instantiate TeacherNet")
        self.arch = teacher.arch
        self.pretrained = teacher.pretrained
        self.model, _, _ = open_clip.create_model_and_transforms(teacher.arch, pretrained=teacher.pretrained)
        self.model.requires_grad_(False)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(teacher.arch)
        self.last_features_dim = self.model.transformer.resblocks[-1].mlp.c_proj.out_features

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def encode_image(self, x):
        return self.model.encode_image(x)

    def encode_text(self, x):
        return self.model.encode_text(x)

    def forward(self, x):
        with torch.no_grad():
            clip_img_features = self.encode_image(x).detach()
        clip_img_features = feature_norm(clip_img_features)
        return clip_img_features

class StudentNet(nn.Module):
    def __init__(self, student, class_num, use_teacher=True):
        super(StudentNet, self).__init__()
        self.use_teacher = use_teacher
        self.num_features = None
        if self.use_teacher:
            origin_model = models.__dict__[student.arch](pretrained=True)
            self.model = ModifiedResNet(origin_model, class_num)
            self.num_features = self.model.num_features
        else:
            self.model = models.__dict__[student.arch](pretrained=True)
            try:
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, class_num)
            except Exception:
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, class_num)

    def forward(self, x):
        if self.use_teacher:
            hidden_features, out = self.model(x)
            return feature_norm(hidden_features), out
        out = self.model(x)
        return out


class ModifiedResNet(torch.nn.Module):
    def __init__(self, origin_model, classnum, dropout: float = 0.3):
        super(ModifiedResNet, self).__init__()
        self.resnet = origin_model

        try:
            num_features = origin_model.fc.in_features
            self.resnet.fc = nn.Identity()
        except Exception:
            num_features = origin_model.classifier[1].in_features
            self.resnet.classifier = nn.Identity()
        self.drop = nn.Dropout(p=0)
        self.linear_cls = nn.Linear(num_features, classnum)
        self.num_features = num_features

    def forward(self, x):
        hidden_features = self.resnet(x)
        out = self.linear_cls(self.drop(hidden_features))
        return hidden_features, out

