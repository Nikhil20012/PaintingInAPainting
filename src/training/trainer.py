from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import MultiTaskLoss


class Trainer:
    def __init__(
        self,
        model:        nn.Module,
        train_dl:     DataLoader,
        val_dl:       DataLoader,
        lr:           float,
        weight_decay: float,
        w_style:      float,
        w_artist:     float,
        w_genre:      float,
        w_hidden:     float,
        w_heatmap:    float,
        device:       str,
        ckpt_dir:     Path,
    ):
        self.model    = model.to(device)
        self.device   = device
        self.ckpt_dir = ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = MultiTaskLoss(w_style, w_artist, w_genre, w_hidden, w_heatmap)

        # only pass parameters that require gradients — frozen layers are excluded
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # cosine annealing smoothly reduces lr to near zero over training
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6,
        )

        self.train_dl = train_dl
        self.val_dl   = val_dl

    def _run_epoch(self, dl: DataLoader, train: bool) -> dict:
        self.model.train(train)
        totals = {}

        with torch.set_grad_enabled(train):
            for batch in tqdm(dl, leave=False):
                imgs    = batch["composite"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                preds  = self.model(imgs)
                losses = self.criterion(preds, targets)

                if train:
                    self.optimizer.zero_grad()
                    losses["total"].backward()
                    # gradient clipping prevents exploding gradients
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                for k, v in losses.items():
                    totals[k] = totals.get(k, 0.0) + v.item()

        n = len(dl)
        return {k: v / n for k, v in totals.items()}

    def fit(self, epochs: int) -> float:
        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_losses = self._run_epoch(self.train_dl, train=True)
            val_losses   = self._run_epoch(self.val_dl,   train=False)
            self.scheduler.step()

            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_losses.items()}, step=epoch
            )
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_losses.items()}, step=epoch
            )
            mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"train {train_losses['total']:.4f} | "
                f"val {val_losses['total']:.4f}"
            )

            if val_losses["total"] < best_val:
                best_val = val_losses["total"]
                ckpt = self.ckpt_dir / "best.pth"
                torch.save(self.model.state_dict(), ckpt)
                mlflow.log_artifact(str(ckpt))
                print(f"  saved best checkpoint (val loss {best_val:.4f})")

        return best_val