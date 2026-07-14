import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(logits, targets)

        prob = torch.sigmoid(logits)
        num  = 2 * (prob * targets).sum(dim=(-2, -1))
        den  = prob.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) + 1e-6
        dice = 1 - (num / den).mean()

        return bce + dice


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        w_style:   float = 1.0,
        w_artist:  float = 1.0,
        w_genre:   float = 1.0,
        w_hidden:  float = 2.0,
        w_heatmap: float = 3.0,
    ):
        super().__init__()
        self.w_style   = w_style
        self.w_artist  = w_artist
        self.w_genre   = w_genre
        self.w_hidden  = w_hidden
        self.w_heatmap = w_heatmap

        self.ce       = nn.CrossEntropyLoss()
        self.bce      = nn.BCEWithLogitsLoss()
        self.dice_bce = DiceBCELoss()

    def forward(self, preds: dict, targets: dict) -> dict[str, torch.Tensor]:
        l_style   = self.ce(preds["style"],  targets["style"])
        l_artist  = self.ce(preds["artist"], targets["artist"])
        l_genre   = self.ce(preds["genre"],  targets["genre"])
        l_hidden  = self.bce(preds["hidden"], targets["hidden"])
        l_heatmap = self.dice_bce(preds["heatmap"], targets["heatmap"])

        total = (
            self.w_style   * l_style   +
            self.w_artist  * l_artist  +
            self.w_genre   * l_genre   +
            self.w_hidden  * l_hidden  +
            self.w_heatmap * l_heatmap
        )

        return {
            "total":   total,
            "style":   l_style,
            "artist":  l_artist,
            "genre":   l_genre,
            "hidden":  l_hidden,
            "heatmap": l_heatmap,
        }