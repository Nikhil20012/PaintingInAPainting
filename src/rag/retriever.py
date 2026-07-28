"""Pinecone vector store for art history context retrieval."""

import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from src.rag.embeddings import ArtEmbedder, EMBEDDING_DIM

INDEX_NAME = "painting-art-history"


class ArtRetriever:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Missing PINECONE_API_KEY in .env")

        self.pc = Pinecone(api_key=api_key)
        self.embedder = ArtEmbedder()
        self.index_name = INDEX_NAME

        # create index if it doesn't exist
        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(self.index_name)

    def upsert(self, chunks: list[dict]) -> int:
        """Upsert text chunks with metadata into Pinecone.

        Each chunk: {"id": str, "text": str, "metadata": dict}
        """
        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed(texts)

        records = []
        for chunk, vector in zip(chunks, vectors):
            records.append({
                "id": chunk["id"],
                "values": vector,
                "metadata": {**chunk["metadata"], "text": chunk["text"]},
            })

        # upsert in batches of 100
        batch_size = 100
        upserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.index.upsert(vectors=batch)
            upserted += len(batch)

        return upserted

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k relevant art history chunks for a query."""
        query_vector = self.embedder.embed_query(query)

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        return [
            {
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "artist": match.metadata.get("artist", ""),
                "style": match.metadata.get("style", ""),
                "genre": match.metadata.get("genre", ""),
            }
            for match in results.matches
        ]