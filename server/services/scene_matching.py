"""Shared scene-object matching helpers for queries and highlights."""

from __future__ import annotations

import re

import numpy as np

STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "around",
    "at",
    "can",
    "do",
    "find",
    "for",
    "from",
    "here",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "show",
    "that",
    "the",
    "there",
    "these",
    "this",
    "those",
    "to",
    "what",
    "where",
    "which",
    "with",
    "you",
}

TOKEN_CANONICAL = {
    "adult": "person",
    "backpack": "bag",
    "bags": "bag",
    "books": "book",
    "bookshelf": "shelf",
    "boy": "person",
    "boys": "person",
    "child": "person",
    "children": "person",
    "couch": "sofa",
    "cups": "cup",
    "display": "monitor",
    "displays": "monitor",
    "girl": "person",
    "girls": "person",
    "guy": "person",
    "guys": "person",
    "human": "person",
    "humans": "person",
    "lady": "person",
    "ladies": "person",
    "laptops": "laptop",
    "loveseat": "sofa",
    "men": "person",
    "monitors": "monitor",
    "mugs": "cup",
    "notebook": "book",
    "notebooks": "book",
    "people": "person",
    "screens": "monitor",
    "shelves": "shelf",
    "sofas": "sofa",
    "television": "monitor",
    "televisions": "monitor",
    "tv": "monitor",
    "tvs": "monitor",
    "woman": "person",
    "women": "person",
}

TOKEN_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for stable lexical matching."""
    return " ".join(TOKEN_RE.sub(" ", text.lower()).split())


def singularize(token: str) -> str:
    """Apply a light singularization pass for simple plural forms."""
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ses", "xes", "zes", "ches", "shes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_token(token: str) -> str:
    """Normalize a single token into its matching form."""
    token = singularize(token)
    return TOKEN_CANONICAL.get(token, token)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Normalize and tokenize free-form text."""
    tokens = [normalize_token(tok) for tok in normalize_text(text).split()]
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in STOPWORDS]
    return tokens


def lexical_score(query: str, label: str) -> float:
    """Score how directly a query names a label."""
    normalized_query = normalize_text(query)
    normalized_label = normalize_text(label)
    if not normalized_query or not normalized_label:
        return 0.0
    if normalized_query == normalized_label:
        return 1.0
    if normalized_query in normalized_label:
        return 0.95

    query_tokens = set(tokenize(query))
    label_tokens = set(tokenize(label, remove_stopwords=False))
    if not query_tokens or not label_tokens:
        return 0.0

    overlap = len(query_tokens & label_tokens)
    if overlap == 0:
        return 0.0
    return overlap / len(query_tokens)


def node_embedding(node: dict, gaussian_store) -> np.ndarray | None:
    """Return a cached mean CLIP embedding for a scene-graph node."""
    cached = node.get("_semantic_embedding")
    if cached is not None:
        return cached
    if gaussian_store is None or gaussian_store.decoded_embeddings is None:
        return None

    indices = np.asarray(node.get("gaussian_indices", []), dtype=np.int64)
    if len(indices) == 0:
        return None

    valid = indices[(indices >= 0) & (indices < len(gaussian_store.decoded_embeddings))]
    if len(valid) == 0:
        return None

    embedding = gaussian_store.decoded_embeddings[valid].mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-8:
        return None

    embedding /= norm
    node["_semantic_embedding"] = embedding
    return embedding
