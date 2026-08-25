"""
Ported verbatim from bowang-lab/BLEEP/modules.py, with the unused
alternative encoders (ViT / CLIP / resnet101 / resnet152) dropped and the
config import made explicit.

Do not "improve" this file. Architectural fidelity to the published model
is the point -- any change here has to be reported as a deviation.
"""
import timm
from torch import nn

import config_her2st as CFG


class ImageEncoder(nn.Module):
    """Encode a 224x224 patch to a fixed-size vector."""

    def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained,
                 trainable=CFG.trainable):
        super().__init__()
        # NOTE: BLEEP passes `pretrained` positionally. Keyword form here is
        # equivalent and survives timm API churn.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    """Linear -> GELU -> Linear -> Dropout -> residual -> LayerNorm."""

    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim,
                 dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
