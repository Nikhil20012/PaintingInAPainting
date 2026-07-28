"""LangGraph RAG workflow for grounded narrative generation.

Flow: model predictions → retrieve art history context → Claude generates narrative.
"""

import os
from typing import TypedDict

import anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.rag.retriever import ArtRetriever


class RAGState(TypedDict):
    """State that flows through the RAG graph."""
    style: str
    artist: str
    genre: str
    has_hidden: bool
    confidence: float
    retrieved_context: list[dict]
    narrative: str


def build_retrieval_query(state: RAGState) -> str:
    """Build a search query from model predictions."""
    parts = []
    if state["artist"] and state["artist"] != "unknown":
        parts.append(state["artist"])
    if state["style"]:
        parts.append(state["style"])
    if state["genre"]:
        parts.append(state["genre"])
    return " ".join(parts)


def retrieve_context(state: RAGState) -> RAGState:
    """Retrieve relevant art history context from Pinecone."""
    retriever = ArtRetriever()
    query = build_retrieval_query(state)
    context = retriever.retrieve(query, top_k=5)
    return {**state, "retrieved_context": context}


def generate_narrative(state: RAGState) -> RAGState:
    """Generate a grounded narrative using Claude with retrieved context."""
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # format retrieved context for the prompt
    context_lines = []
    for chunk in state["retrieved_context"]:
        context_lines.append(f"- {chunk['text']} (relevance: {chunk['score']:.2f})")
    context_block = "\n".join(context_lines) if context_lines else "No additional context available."

    hidden_status = "a hidden painting was detected beneath the surface" if state["has_hidden"] else "no hidden painting was detected"

    prompt = f"""You are an art historian analyzing a painting. Based on the model's predictions and the retrieved art history context, write a concise, grounded narrative about this painting.

Model predictions:
- Style: {state["style"]}
- Artist: {state["artist"]}
- Genre: {state["genre"]}
- Hidden layer: {hidden_status}
- Confidence: {state["confidence"]:.1%}

Retrieved art history context:
{context_block}

Write a 3-4 sentence narrative that connects the model's findings with the historical context. If a hidden painting was detected, discuss what this might mean historically."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    narrative = response.content[0].text
    return {**state, "narrative": narrative}


def build_rag_graph() -> StateGraph:
    """Build the LangGraph RAG pipeline."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", generate_narrative)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def run_rag_pipeline(
    style: str,
    artist: str,
    genre: str,
    has_hidden: bool,
    confidence: float,
) -> str:
    """Run the full RAG pipeline and return the narrative."""
    graph = build_rag_graph()

    initial_state: RAGState = {
        "style": style,
        "artist": artist,
        "genre": genre,
        "has_hidden": has_hidden,
        "confidence": confidence,
        "retrieved_context": [],
        "narrative": "",
    }

    result = graph.invoke(initial_state)
    return result["narrative"]