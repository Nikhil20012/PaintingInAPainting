"""Evaluate trained model on test set with metrics and Grad-CAM visualizations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from src.data.dataset import PaintingDataset
from src.models.model import PaintingModel


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, ckpt_path: str) -> PaintingModel:
    m = cfg["model"]
    model = PaintingModel(
        freeze_layers=m["freeze_layers"],
        n_styles=m["n_styles"],
        n_artists=m["n_artists"],
        n_genres=m["n_genres"],
        dropout=m["dropout"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    return model


def evaluate_classification(model, test_dl, device) -> dict:
    """Run inference on test set and compute per-task metrics."""
    all_preds = {"style": [], "artist": [], "genre": [], "hidden": []}
    all_targets = {"style": [], "artist": [], "genre": [], "hidden": []}

    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            imgs = batch["composite"].to(device)
            targets = batch["targets"]
            preds = model(imgs)

            all_preds["style"].extend(preds["style"].argmax(dim=1).cpu().tolist())
            all_preds["genre"].extend(preds["genre"].argmax(dim=1).cpu().tolist())
            all_preds["hidden"].extend(torch.sigmoid(preds["hidden"]).cpu().tolist())

            # skip artist predictions where target is -1
            artist_preds = preds["artist"].argmax(dim=1).cpu().tolist()
            artist_targets = targets["artist"].tolist()
            for p, t in zip(artist_preds, artist_targets):
                if t != -1:
                    all_preds["artist"].append(p)
                    all_targets["artist"].append(t)

            all_targets["style"].extend(targets["style"].tolist())
            all_targets["genre"].extend(targets["genre"].tolist())
            all_targets["hidden"].extend(targets["hidden"].tolist())

    metrics = {}

    # style metrics
    metrics["style_accuracy"] = accuracy_score(all_targets["style"], all_preds["style"])
    metrics["style_f1"] = f1_score(all_targets["style"], all_preds["style"], average="weighted")

    # artist metrics (excluding unknown artists)
    if all_targets["artist"]:
        metrics["artist_accuracy"] = accuracy_score(all_targets["artist"], all_preds["artist"])
        metrics["artist_f1"] = f1_score(all_targets["artist"], all_preds["artist"], average="weighted")

    # genre metrics
    metrics["genre_accuracy"] = accuracy_score(all_targets["genre"], all_preds["genre"])
    metrics["genre_f1"] = f1_score(all_targets["genre"], all_preds["genre"], average="weighted")

    # hidden detection metrics
    hidden_binary = [1 if p > 0.5 else 0 for p in all_preds["hidden"]]
    hidden_targets_int = [int(t) for t in all_targets["hidden"]]
    metrics["hidden_accuracy"] = accuracy_score(hidden_targets_int, hidden_binary)
    metrics["hidden_auc"] = roc_auc_score(hidden_targets_int, all_preds["hidden"])

    return metrics


def generate_gradcam(model, test_dl, output_dir: Path, n_samples: int = 10):
    """Generate Grad-CAM visualizations for sample test images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # target the last transformer block in the ViT encoder
    target_layer = model.encoder.vit.encoder.layers[-1].ln_1

    cam = GradCAM(model=model, target_layers=[target_layer])

    # inverse normalization for display
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    samples = 0
    for batch in test_dl:
        imgs = batch["composite"]
        fnames = batch["fname"]

        for i in range(imgs.shape[0]):
            if samples >= n_samples:
                return

            img_tensor = imgs[i].unsqueeze(0)
            grayscale_cam = cam(input_tensor=img_tensor)[0]

            # convert tensor back to displayable image
            img_display = inv_normalize(imgs[i]).permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)

            cam_image = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_display)
            axes[0].set_title("Input")
            axes[0].axis("off")

            axes[1].imshow(grayscale_cam, cmap="jet")
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            axes[2].imshow(cam_image)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / f"gradcam_{fnames[i]}", dpi=150, bbox_inches="tight")
            plt.close()

            samples += 1


def main():
    cfg = load_config()
    device = cfg["training"]["device"]
    ckpt_path = "checkpoints/run/best.pth"

    if not Path(ckpt_path).exists():
        print(f"No checkpoint found at {ckpt_path}")
        return

    print("Loading model...")
    model = load_model(cfg, ckpt_path)
    model.to(device)

    print("Loading test data...")
    test_ds = PaintingDataset(
        root=Path(cfg["data"]["synthetic_root"]),
        split="test",
        image_size=cfg["data"]["image_size"],
    )
    test_dl = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    print("Running evaluation...")
    metrics = evaluate_classification(model, test_dl, device)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # save metrics
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")

    print("Generating Grad-CAM visualizations...")
    generate_gradcam(model, test_dl, output_dir / "gradcam", n_samples=10)
    print(f"Grad-CAM images saved to {output_dir / 'gradcam'}/")


if __name__ == "__main__":
    main()