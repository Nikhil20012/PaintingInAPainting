"""Generate synthetic composite dataset using Gold labels."""

from pathlib import Path

import yaml

from src.data.blend import generate


def main() -> None:
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    generate(
        wikiart_root=Path(data_cfg["local_root"]) / "wikiart",
        output_root=Path(data_cfg["local_root"]) / "synthetic",
        gold_csv=Path(data_cfg["local_root"]) / "gold" / "labels" / "gold_wikiart.csv",
        n_pairs=data_cfg["num_pairs"],
        image_size=data_cfg["image_size"],
        alpha_min=data_cfg["alpha_min"],
        alpha_max=data_cfg["alpha_max"],
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        seed=data_cfg["seed"],
    )


if __name__ == "__main__":
    main()