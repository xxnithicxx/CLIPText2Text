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

        if self.training:
            # Generate targets
            targets = torch.eye(logits.size(0)).to(logits.device)

            # Calculate loss
            texts_loss_1 = F.cross_entropy(logits, targets, reduction="mean")
            texts_loss_2 = F.cross_entropy(logits.T, targets.T, reduction="mean")
            loss = (texts_loss_1 + texts_loss_2) / 2.0

            return loss
        else:
            return logits

# Ref: https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py
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

        # Normalize embeddings
        text_embeddings_1 = F.normalize(text_embeddings_1, p=2, dim=-1)
        text_embeddings_2 = F.normalize(text_embeddings_2, p=2, dim=-1)

        # Calculate similarity logits
        logits = torch.matmul(text_embeddings_1, text_embeddings_2.T) / self.temperature
        
        if self.training:
            # The implementation of SigLIP logits is as follows:
            # logit_scale = torch.nn.Parameter(torch.tensor(1.0))
            # logit_bias = torch.nn.Parameter(torch.tensor(1.0))
            # logits = (
            #     torch.matmul(text_embeddings_1, text_embeddings_2.T) 
            #     * logit_scale.exp()
            #     + logit_bias
            #     / self.temperature
            # )
            
            # Create masks based on is_duplicate
            is_duplicate = batch["label"].float().view(-1, 1)  # Shape: (batch_size, 1)
            positive_mask = is_duplicate  # Positive if is_duplicate == 1
            negative_mask = 1.0 - is_duplicate  # Negative if is_duplicate == 0

            # Compute Sigmoid Loss for pairs
            # positive_mask = torch.eye(logits.size(0), device=logits.device)
            # negative_mask = 0.0 - positive_mask # Positive is 1, negative is -1
            
            positive_loss = -F.logsigmoid(logits) * positive_mask
            negative_loss = -F.logsigmoid(-logits) * negative_mask

            # Aggregate loss
            loss = positive_loss.sum() + negative_loss.sum()
            loss /= logits.size(0)  # Normalize by batch size

            return loss
        else:
            return logits

class SigLIPModelGram(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        text_embedding=CFG.text_embedding,
        gram_loss_weight=0.1  # Weight for the Gram-Matrix Loss
    ):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.gram_loss_weight = gram_loss_weight

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

        # Normalize embeddings
        text_embeddings_1 = F.normalize(text_embeddings_1, p=2, dim=-1)
        text_embeddings_2 = F.normalize(text_embeddings_2, p=2, dim=-1)

        # Calculate similarity logits
        logits = torch.matmul(text_embeddings_1, text_embeddings_2.T) / self.temperature

        if self.training:
            # Create masks based on is_duplicate
            is_duplicate = batch["is_duplicate"].float().view(-1, 1)  # Shape: (batch_size, 1)
            positive_mask = is_duplicate  # Positive if is_duplicate == 1
            negative_mask = 1.0 - is_duplicate  # Negative if is_duplicate == 0

            # Compute Sigmoid Loss for pairs
            positive_loss = -F.logsigmoid(logits) * positive_mask
            negative_loss = -F.logsigmoid(-logits) * negative_mask
            similarity_loss = positive_loss.sum() + negative_loss.sum()
            similarity_loss /= logits.size(0)  # Normalize by batch size

            # Compute Gram-Matrix Loss
            gram_matrix_1 = torch.matmul(text_embeddings_1.T, text_embeddings_1)  # Shape: (embedding_dim, embedding_dim)
            gram_matrix_2 = torch.matmul(text_embeddings_2.T, text_embeddings_2)  # Shape: (embedding_dim, embedding_dim)
            gram_loss = F.mse_loss(gram_matrix_1, gram_matrix_2)

            # Total loss (weighted sum)
            total_loss = similarity_loss + self.gram_loss_weight * gram_loss

            return total_loss
        else:
            return logits