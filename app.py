"""Flask API for hidden painting detection with RAG narrative generation."""

import io
import csv
from pathlib import Path

import torch
import yaml
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms as T

from src.models.model import PaintingModel
from src.rag.graph import run_rag_pipeline

app = Flask(__name__)

# globals loaded once at startup
MODEL = None
DEVICE = None
TRANSFORM = None
STYLE_MAP = {}
ARTIST_MAP = {}
GENRE_MAP = {}


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_label_maps():
    """Load index-to-name mappings from Gold CSVs."""
    global STYLE_MAP, ARTIST_MAP, GENRE_MAP
    gold_dir = Path("data/gold/labels")

    with open(gold_dir / "gold_style_mapping.csv") as f:
        for row in csv.DictReader(f):
            STYLE_MAP[int(row["style_idx"])] = row["style"].replace("_", " ")

    with open(gold_dir / "gold_artist_mapping.csv") as f:
        for row in csv.DictReader(f):
            ARTIST_MAP[int(row["artist_idx"])] = row["artist"]

    with open(gold_dir / "gold_genre_mapping.csv") as f:
        for row in csv.DictReader(f):
            GENRE_MAP[int(row["genre_idx"])] = row["genre"].replace("_", " ")


def load_model(cfg: dict) -> PaintingModel:
    m = cfg["model"]
    model = PaintingModel(
        freeze_layers=m["freeze_layers"],
        n_styles=m["n_styles"],
        n_artists=m["n_artists"],
        n_genres=m["n_genres"],
        dropout=m["dropout"],
    )
    ckpt = cfg["api"]["model_checkpoint"]
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def get_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = MODEL(tensor)

    # classification predictions
    style_idx = preds["style"].argmax(dim=1).item()
    artist_idx = preds["artist"].argmax(dim=1).item()
    genre_idx = preds["genre"].argmax(dim=1).item()
    hidden_prob = torch.sigmoid(preds["hidden"]).item()
    has_hidden = hidden_prob > 0.5

    style_name = STYLE_MAP.get(style_idx, "unknown")
    artist_name = ARTIST_MAP.get(artist_idx, "unknown")
    genre_name = GENRE_MAP.get(genre_idx, "unknown")

    # heatmap as nested list
    heatmap = torch.sigmoid(preds["heatmap"]).squeeze().cpu().numpy().tolist()

    result = {
        "style": style_name,
        "artist": artist_name,
        "genre": genre_name,
        "hidden_detected": has_hidden,
        "hidden_confidence": round(hidden_prob, 4),
        "heatmap": heatmap,
    }

    # RAG narrative generation
    try:
        narrative = run_rag_pipeline(
            style=style_name,
            artist=artist_name,
            genre=genre_name,
            has_hidden=has_hidden,
            confidence=hidden_prob,
        )
        result["narrative"] = narrative
    except Exception as e:
        result["narrative"] = f"Narrative generation failed: {e}"

    return jsonify(result)


def create_app():
    global MODEL, DEVICE, TRANSFORM

    cfg = load_config()
    DEVICE = cfg["training"]["device"]

    load_label_maps()
    MODEL = load_model(cfg)
    MODEL.to(DEVICE)
    TRANSFORM = get_transform(cfg["data"]["image_size"])

    return app


if __name__ == "__main__":
    cfg = load_config()
    app = create_app()
    app.run(
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        debug=True,
    )