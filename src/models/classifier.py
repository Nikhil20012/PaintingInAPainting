import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_dim:    int   = 768,
        n_styles:  int   = 27,
        n_artists: int   = 23,
        n_genres:  int   = 10,
        dropout:   float = 0.3,
    ):
        super().__init__()

        # shared trunk — learns features common to all four tasks
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # four independent output branches
        self.style  = nn.Linear(256, n_styles)
        self.artist = nn.Linear(256, n_artists)
        self.genre  = nn.Linear(256, n_genres)
        self.hidden = nn.Linear(256, 1)   # single logit — BCEWithLogitsLoss handles sigmoid

    def forward(self, cls_token: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.trunk(cls_token)
        return {
            "style":  self.style(feat),            # (B, 27)
            "artist": self.artist(feat),            # (B, 23)
            "genre":  self.genre(feat),             # (B, 10)
            "hidden": self.hidden(feat).squeeze(1), # (B,)
        }