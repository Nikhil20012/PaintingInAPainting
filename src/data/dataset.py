import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def _img_transform(size: int, train: bool = False) -> T.Compose:
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _mask_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])


class PaintingDataset(Dataset):
    def __init__(self, root: Path, split: str = "train", image_size: int = 224):
        self.root    = root / split
        self.tf_img  = _img_transform(image_size, train=(split == "train"))
        self.tf_mask = _mask_transform(image_size)

        with open(root / "manifest.json") as f:
            all_entries = json.load(f)

        self.entries = [e for e in all_entries if e["split"] == split]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        e     = self.entries[idx]
        fname = e["filename"]
        mname = e["mask"]

        composite = self.tf_img(
            Image.open(self.root / "composite" / fname).convert("RGB")
        )
        mask = self.tf_mask(
            Image.open(self.root / "mask" / mname).convert("L")
        )
        mask = (mask > 0.5).float().squeeze(0)

        # labels come from the top (visible) painting
        targets = {
            "style":   torch.tensor(e["top_style_idx"],  dtype=torch.long),
            "artist":  torch.tensor(e["top_artist_idx"], dtype=torch.long),
            "genre":   torch.tensor(e["top_genre_idx"],  dtype=torch.long),
            "hidden":  torch.tensor(1.0,                 dtype=torch.float),
            "heatmap": mask,
        }

        return {"composite": composite, "targets": targets, "fname": fname}