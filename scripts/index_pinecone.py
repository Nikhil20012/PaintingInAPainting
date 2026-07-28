"""Build art history context chunks from WikiArt metadata and index into Pinecone.

Generates descriptive text chunks for each artist, style, and genre
in the Gold dataset, embeds them with sentence-transformers, and
upserts into Pinecone for RAG retrieval.
"""

import csv
from collections import defaultdict
from pathlib import Path

from src.rag.retriever import ArtRetriever


def load_gold_metadata(gold_dir: Path) -> dict:
    """Load Gold CSVs and build metadata summaries."""
    # load style mapping
    styles = {}
    with open(gold_dir / "gold_style_mapping.csv") as f:
        for row in csv.DictReader(f):
            styles[int(row["style_idx"])] = row["style"]

    # load genre mapping
    genres = {}
    with open(gold_dir / "gold_genre_mapping.csv") as f:
        for row in csv.DictReader(f):
            genres[int(row["genre_idx"])] = row["genre"]

    # load artist mapping
    artists = {}
    with open(gold_dir / "gold_artist_mapping.csv") as f:
        for row in csv.DictReader(f):
            artists[int(row["artist_idx"])] = row["artist"]

    # load main dataset to get per-artist stats
    artist_styles = defaultdict(set)
    artist_genres = defaultdict(set)
    artist_counts = defaultdict(int)
    style_counts = defaultdict(int)

    with open(gold_dir / "gold_wikiart.csv") as f:
        for row in csv.DictReader(f):
            artist = row["artist"]
            style = row["style"]
            genre = row["primary_genre"]
            artist_styles[artist].add(style)
            artist_genres[artist].add(genre)
            artist_counts[artist] += 1
            style_counts[style] += 1

    return {
        "styles": styles,
        "genres": genres,
        "artists": artists,
        "artist_styles": dict(artist_styles),
        "artist_genres": dict(artist_genres),
        "artist_counts": dict(artist_counts),
        "style_counts": dict(style_counts),
    }


def build_chunks(metadata: dict) -> list[dict]:
    """Build text chunks from metadata for embedding."""
    chunks = []

    # style chunks
    for idx, style in metadata["styles"].items():
        clean_style = style.replace("_", " ")
        count = metadata["style_counts"].get(style, 0)
        text = (
            f"{clean_style} is an art movement represented by {count} paintings "
            f"in the WikiArt collection. Works in the {clean_style} style share "
            f"distinctive visual characteristics and historical significance."
        )
        chunks.append({
            "id": f"style-{idx}",
            "text": text,
            "metadata": {"type": "style", "style": clean_style},
        })

    # genre chunks
    for idx, genre in metadata["genres"].items():
        clean_genre = genre.replace("_", " ")
        text = (
            f"{clean_genre} is a genre of painting that encompasses works "
            f"focused on specific subject matter and artistic traditions."
        )
        chunks.append({
            "id": f"genre-{idx}",
            "text": text,
            "metadata": {"type": "genre", "genre": clean_genre},
        })

    # artist chunks
    for idx, artist in metadata["artists"].items():
        count = metadata["artist_counts"].get(artist, 0)
        styles = metadata["artist_styles"].get(artist, set())
        genres = metadata["artist_genres"].get(artist, set())

        clean_styles = ", ".join(s.replace("_", " ") for s in sorted(styles))
        clean_genres = ", ".join(g.replace("_", " ") for g in sorted(genres))

        text = (
            f"{artist} is a painter with {count} works in the collection. "
            f"Their work spans the following styles: {clean_styles}. "
            f"Their paintings cover genres including {clean_genres}."
        )
        chunks.append({
            "id": f"artist-{idx}",
            "text": text,
            "metadata": {
                "type": "artist",
                "artist": artist,
                "style": clean_styles,
                "genre": clean_genres,
            },
        })

    return chunks


def main() -> None:
    gold_dir = Path("data/gold/labels")

    print("Loading Gold metadata...")
    metadata = load_gold_metadata(gold_dir)
    print(f"  Styles: {len(metadata['styles'])}")
    print(f"  Genres: {len(metadata['genres'])}")
    print(f"  Artists: {len(metadata['artists'])}")

    print("Building text chunks...")
    chunks = build_chunks(metadata)
    print(f"  Total chunks: {len(chunks)}")

    print("Embedding and indexing into Pinecone...")
    retriever = ArtRetriever()
    upserted = retriever.upsert(chunks)
    print(f"  Upserted {upserted} vectors")

    # test retrieval
    print("\nTest query: 'Impressionism landscape'")
    results = retriever.retrieve("Impressionism landscape", top_k=3)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['text'][:100]}...")

    print("\nDone.")


if __name__ == "__main__":
    main()