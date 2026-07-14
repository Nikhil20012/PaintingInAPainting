"""
Usage:
    python scripts/train.py                        # single training run
    python scripts/train.py --tune                 # Optuna hyperparameter search
    python scripts/train.py --tune --n-trials 30   # custom trial count
"""

import argparse
from pathlib import Path

import mlflow
import optuna
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import PaintingDataset
from src.models.model import PaintingModel
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    root = Path(cfg["data"]["synthetic_root"])
    size = cfg["data"]["image_size"]
    bs   = cfg["training"]["batch_size"]
    nw   = cfg["training"]["num_workers"]

    train_ds = PaintingDataset(root, "train", size)
    val_ds   = PaintingDataset(root, "val",   size)

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    return train_dl, val_dl


def run_trial(trial: optuna.Trial, cfg: dict, train_dl: DataLoader, val_dl: DataLoader) -> float:
    lr           = trial.suggest_float("lr",           1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    freeze       = trial.suggest_int("freeze_layers",  4, 10)
    dropout      = trial.suggest_float("dropout",      0.1, 0.5)
    w_hidden     = trial.suggest_float("w_hidden",     1.0, 5.0)
    w_heatmap    = trial.suggest_float("w_heatmap",    1.0, 6.0)

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(trial.params)

        model   = PaintingModel(freeze_layers=freeze, dropout=dropout)
        trainer = Trainer(
            model        = model,
            train_dl     = train_dl,
            val_dl       = val_dl,
            lr           = lr,
            weight_decay = weight_decay,
            w_style      = 1.0,
            w_artist     = 1.0,
            w_genre      = 1.0,
            w_hidden     = w_hidden,
            w_heatmap    = w_heatmap,
            device       = cfg["training"]["device"],
            ckpt_dir     = Path("checkpoints") / f"trial-{trial.number}",
        )
        return trainer.fit(epochs=cfg["optuna"]["epochs_per_trial"])


def train_single(cfg: dict, train_dl: DataLoader, val_dl: DataLoader) -> None:
    t  = cfg["training"]
    lw = t["loss_weights"]
    m  = cfg["model"]

    with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
        mlflow.log_params({
            "lr":           t["learning_rate"],
            "weight_decay": t["weight_decay"],
            "batch_size":   t["batch_size"],
            "epochs":       t["epochs"],
            "freeze_layers": m["freeze_layers"],
            "dropout":      m["dropout"],
            **{f"w_{k}": v for k, v in lw.items()},
        })

        model   = PaintingModel(
            freeze_layers = m["freeze_layers"],
            dropout       = m["dropout"],
        )
        trainer = Trainer(
            model        = model,
            train_dl     = train_dl,
            val_dl       = val_dl,
            lr           = t["learning_rate"],
            weight_decay = t["weight_decay"],
            w_style      = lw["style"],
            w_artist     = lw["artist"],
            w_genre      = lw["genre"],
            w_hidden     = lw["hidden"],
            w_heatmap    = lw["heatmap"],
            device       = t["device"],
            ckpt_dir     = Path("checkpoints/run"),
        )
        trainer.fit(epochs=t["epochs"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--tune",     action="store_true")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    train_dl, val_dl = build_loaders(cfg)

    if args.tune:
        n_trials = args.n_trials or cfg["optuna"]["n_trials"]
        study    = optuna.create_study(
            direction    = "minimize",
            study_name   = "painting-hparam-search",
            storage      = cfg["optuna"]["storage"],
            load_if_exists = True,
        )
        with mlflow.start_run(run_name="optuna-search"):
            study.optimize(
                lambda trial: run_trial(trial, cfg, train_dl, val_dl),
                n_trials = n_trials,
            )
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best val loss: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    else:
        train_single(cfg, train_dl, val_dl)


if __name__ == "__main__":
    main()