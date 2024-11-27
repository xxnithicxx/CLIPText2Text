import torch
import torch.nn as nn
import torch.nn.functional as F
from models.text_encoder import TextEncoder
from models.projection_head import ProjectionHead
from config.cfg import CFG


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Encode text inputs
        text_features_1 = self.text_encoder(
            input_ids=batch["input_ids_1"], attention_mask=batch["attention_mask_1"]
        )
        text_features_2 = self.text_encoder(
            input_ids=batch["input_ids_2"], attention_mask=batch["attention_mask_2"]
        )

        # Project text features to embedding space
        text_embeddings_1 = self.text_projection(text_features_1)
        text_embeddings_2 = self.text_projection(text_features_2)

        # Calculate similarity logits
        logits = (text_embeddings_1 @ text_embeddings_2.T) / self.temperature

        # Generate targets
        targets = torch.eye(logits.size(0)).to(logits.device)

        # Calculate loss
        texts_loss_1 = F.cross_entropy(logits, targets, reduction="mean")
        texts_loss_2 = F.cross_entropy(logits.T, targets.T, reduction="mean")
        loss = (texts_loss_1 + texts_loss_2) / 2.0

        return loss

class SigLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        text_embedding=CFG.text_embedding
    ):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Encode text inputs
        text_features_1 = self.text_encoder(
            input_ids=batch["input_ids_1"], attention_mask=batch["attention_mask_1"]
        )
        text_features_2 = self.text_encoder(
            input_ids=batch["input_ids_2"], attention_mask=batch["attention_mask_2"]
        )

        # Project text features to embedding space
        text_embeddings_1 = self.text_projection(text_features_1)
        text_embeddings_2 = self.text_projection(text_features_2)

        # Calculate similarity logits
        logits = torch.matmul(text_embeddings_1, text_embeddings_2.T) / self.temperature

        # Compute Sigmoid Loss for pairs
        positive_mask = torch.eye(logits.size(0), device=logits.device)
        negative_mask = 1.0 - positive_mask
        positive_loss = -F.logsigmoid(logits) * positive_mask
        negative_loss = -F.logsigmoid(-logits) * negative_mask

        # Aggregate loss
        loss = positive_loss.sum() + negative_loss.sum()
        loss /= logits.size(0)  # Normalize by batch size

        return loss
