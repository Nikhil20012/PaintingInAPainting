import torch
import torch.nn as nn

from src.models.encoder import ViTEncoder
from src.models.classifier import ClassificationHead
from src.models.detector import DetectionHead


class PaintingModel(nn.Module):
    def __init__(
        self,
        freeze_layers: int   = 6,
        n_styles:      int   = 27,
        n_artists:     int   = 23,
        n_genres:      int   = 10,
        dropout:       float = 0.3,
    ):
        super().__init__()
        self.encoder    = ViTEncoder(freeze_layers=freeze_layers)
        self.classifier = ClassificationHead(
            in_dim    = self.encoder.hidden_dim,
            n_styles  = n_styles,
            n_artists = n_artists,
            n_genres  = n_genres,
            dropout   = dropout,
        )
        self.detector = DetectionHead(in_dim=self.encoder.hidden_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_token, patch_tokens = self.encoder(x)
        clf_out                 = self.classifier(cls_token)
        heatmap                 = self.detector(patch_tokens)
        return {**clf_out, "heatmap": heatmap}