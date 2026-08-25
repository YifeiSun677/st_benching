"""
Ported verbatim from bowang-lab/BLEEP/models.py (CLIPModel only).

Two properties worth remembering when reading the loss curve:

1. There is NO spot encoder. `spot_projection` is applied directly to the
   833-dim expression vector. The "expression tower" is one linear layer
   plus a residual MLP -- that is the whole thing.

2. The targets are SOFT: F.softmax((img_sim + spot_sim)/2 / tau). This is
   not standard InfoNCE with hard diagonal targets, so there is no
   ln(batch_size) chance floor to compare the loss against. Use
   diagnostics.py (val retrieval accuracy) to judge whether it learned.
"""
import torch
import torch.nn.functional as F
from torch import nn

import config_her2st as CFG
from modules import ImageEncoder, ProjectionHead


def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    raise ValueError(reduction)


class CLIPModel(nn.Module):
    def __init__(self, temperature=CFG.temperature,
                 image_embedding=CFG.image_embedding,
                 spot_embedding=CFG.spot_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding)
        self.temperature = temperature

    def forward(self, batch):
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]

        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            ((images_similarity + spots_similarity) / 2) / self.temperature,
            dim=-1,
        )
        spots_loss = cross_entropy(logits, targets, reduction="none")
        images_loss = cross_entropy(logits.T, targets.T, reduction="none")
        loss = (images_loss + spots_loss) / 2.0
        return loss.mean()

    @torch.no_grad()
    def embed_image(self, images):
        return self.image_projection(self.image_encoder(images))

    @torch.no_grad()
    def embed_spot(self, expression):
        return self.spot_projection(expression)
