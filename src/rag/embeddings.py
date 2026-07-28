"""Embed art history text chunks using sentence-transformers."""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class ArtEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = EMBEDDING_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text chunks into vectors."""
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.model.encode(query).tolist()