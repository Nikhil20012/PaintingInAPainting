import csv
import json
import random
from pathlib import Path
from scipy.ndimage import gaussian_filter

import numpy as np
from PIL import Image
from tqdm import tqdm


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def aging_noise(img: Image.Image, strength: float = 0.015) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    arr = np.clip(arr + np.random.normal(0, strength * 255, arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def blend(
    top: Image.Image,
    hidden: Image.Image,
    alpha: float,
) -> tuple[Image.Image, np.ndarray]:
    top_arr    = np.array(top,    dtype=np.float32)
    hidden_arr = np.array(hidden, dtype=np.float32)
    h, w       = top_arr.shape[:2]

    # spatially varying alpha simulates uneven paint thickness
    noise = np.random.uniform(-0.05, 0.05, (h, w, 1))
    noise = gaussian_filter(noise, sigma=8)
    alpha_map = np.clip(alpha + noise, 0.05, 0.95)

    composite = np.clip(
        alpha_map * top_arr + (1 - alpha_map) * hidden_arr,
        0, 255,
    )
    composite = aging_noise(Image.fromarray(composite.astype(np.uint8)))

    # ground truth mask: white where hidden bleeds through more than 15%
    mask = ((1 - alpha_map.squeeze()) > 0.15).astype(np.uint8) * 255
    return composite, mask


def load_gold_entries(gold_csv: Path) -> dict[str, list[dict]]:
    """Load gold CSV and group entries by split."""
    entries_by_split = {"train": [], "val": [], "test": []}
    with open(gold_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split in entries_by_split:
                entries_by_split[split].append({
                    "filename":   row["filename"],
                    "style_idx":  int(row["style_idx"]),
                    "artist_idx": int(row["artist_idx"]),
                    "genre_idx":  int(row["genre_idx"]),
                })
    return entries_by_split


def generate(
    wikiart_root: Path,
    output_root:  Path,
    gold_csv:     Path,
    n_pairs:      int   = 50_000,
    image_size:   int   = 224,
    alpha_min:    float = 0.60,
    alpha_max:    float = 0.90,
    train_split:  float = 0.80,
    val_split:    float = 0.10,
    seed:         int   = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    n_train = int(n_pairs * train_split)
    n_val   = int(n_pairs * val_split)
    n_test  = n_pairs - n_train - n_val
    budgets = {"train": n_train, "val": n_val, "test": n_test}

    for split in budgets:
        for sub in ("composite", "top", "hidden", "mask"):
            (output_root / split / sub).mkdir(parents=True, exist_ok=True)

    # load gold entries grouped by split — pairs are sampled
    # within the same split to prevent data leakage
    gold_entries = load_gold_entries(gold_csv)
    for split, entries in gold_entries.items():
        print(f"  Gold {split}: {len(entries):,} images available")
        if len(entries) < 2:
            raise ValueError(f"Need at least 2 Gold images in {split} split")

    manifest = []
    idx = 0

    for split, budget in budgets.items():
        pool = gold_entries[split]

        with tqdm(total=budget, desc=f"Generating {split}") as pbar:
            generated = 0
            while generated < budget:
                entry_top, entry_hidden = random.sample(pool, 2)
                alpha = random.uniform(alpha_min, alpha_max)

                top_path    = wikiart_root / entry_top["filename"]
                hidden_path = wikiart_root / entry_hidden["filename"]

                try:
                    top_img    = resize_center_crop(
                        Image.open(top_path).convert("RGB"), image_size
                    )
                    hidden_img = resize_center_crop(
                        Image.open(hidden_path).convert("RGB"), image_size
                    )
                except Exception as e:
                    tqdm.write(f"Skipping pair: {e}")
                    continue

                composite_img, mask = blend(top_img, hidden_img, alpha)
                mask_img = Image.fromarray(mask)

                fname = f"{idx:06d}.jpg"
                mname = f"{idx:06d}.png"

                composite_img.save(output_root / split / "composite" / fname, quality=95)
                top_img.save(      output_root / split / "top"       / fname, quality=95)
                hidden_img.save(   output_root / split / "hidden"    / fname, quality=95)
                mask_img.save(     output_root / split / "mask"      / mname)

                manifest.append({
                    "id":               idx,
                    "split":            split,
                    "filename":         fname,
                    "mask":             mname,
                    "top_source":       entry_top["filename"],
                    "hidden_source":    entry_hidden["filename"],
                    "alpha":            round(alpha, 4),
                    "top_style_idx":    entry_top["style_idx"],
                    "top_artist_idx":   entry_top["artist_idx"],
                    "top_genre_idx":    entry_top["genre_idx"],
                    "hidden_style_idx": entry_hidden["style_idx"],
                    "hidden_artist_idx":entry_hidden["artist_idx"],
                    "hidden_genre_idx": entry_hidden["genre_idx"],
                })

                generated += 1
                idx += 1
                pbar.update(1)

    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Splits: { {s: budgets[s] for s in budgets} }")