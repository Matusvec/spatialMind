"""CLIP encoding endpoint: text query -> top-K matching Gaussian indices."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.services.similarity import top_k

router = APIRouter()


class EncodeRequest(BaseModel):
    """Request body for POST /clip/encode."""

    text: str
    k: int = 100


class EncodeResponse(BaseModel):
    """Response body for POST /clip/encode."""

    query: str
    k: int
    indices: list[int]
    scores: list[float]
    gaussian_count: int


@router.post("/clip/encode", response_model=EncodeResponse)
async def encode_query(request: EncodeRequest):
    """Encode a text query and return top-K matching Gaussian indices.

    Takes a natural language text query, encodes it via CLIP, computes
    cosine similarity against all decoded Gaussian embeddings, and returns
    the top-K indices with their similarity scores.
    """
    from server.main import get_app_state

    state = get_app_state()

    if not state["gaussian_store"] or not state["gaussian_store"].is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Gaussian store not loaded. Check /health for status.",
        )

    if not state["clip_encoder"]:
        raise HTTPException(
            status_code=503,
            detail="CLIP encoder not initialized. Check /health for status.",
        )

    text_embedding = state["clip_encoder"].encode_text(request.text)
    indices, scores = top_k(
        text_embedding,
        state["gaussian_store"].decoded_embeddings,
        k=request.k,
    )

    return EncodeResponse(
        query=request.text,
        k=request.k,
        indices=indices.tolist(),
        scores=scores.tolist(),
        gaussian_count=state["gaussian_store"].count,
    )
