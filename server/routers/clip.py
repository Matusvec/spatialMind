"""CLIP query endpoints: encode, highlight mask, and probability cloud."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.services.similarity import highlight_mask, probability_cloud, top_k

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


# --- Highlight mask endpoint ---


class HighlightRequest(BaseModel):
    """Request body for POST /clip/highlight."""

    text: str
    k: int = 100
    mode: str = "soft"  # "soft" or "hard"


class HighlightResponse(BaseModel):
    """Response body for POST /clip/highlight."""

    query: str
    gaussian_count: int
    mask: list[float]  # Per-Gaussian opacity values [0.0, 1.0]


@router.post("/clip/highlight", response_model=HighlightResponse)
async def highlight_query(request: HighlightRequest):
    """Return per-Gaussian opacity mask for semantic highlighting.

    The mask is a float array of length N (number of Gaussians) where
    each value is in [0.0, 1.0]. The frontend applies this as an opacity
    multiplier on each Gaussian's color in the Spark.js renderer.
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
    mask = highlight_mask(
        text_embedding,
        state["gaussian_store"].decoded_embeddings,
        k=request.k,
        mode=request.mode,
    )
    return HighlightResponse(
        query=request.text,
        gaussian_count=state["gaussian_store"].count,
        mask=mask.tolist(),
    )


# --- Probability cloud endpoint ---


class ProbabilityRequest(BaseModel):
    """Request body for POST /clip/probability."""

    text: str
    temperature: float = 0.1


class ProbabilityResponse(BaseModel):
    """Response body for POST /clip/probability."""

    query: str
    gaussian_count: int
    probabilities: list[float]  # Per-Gaussian soft assignment scores
    min_score: float
    max_score: float
    mean_score: float


@router.post("/clip/probability", response_model=ProbabilityResponse)
async def probability_query(request: ProbabilityRequest):
    """Return per-Gaussian soft assignment probabilities.

    This is the probability cloud -- a continuous semantic field over
    ALL Gaussians. Every Gaussian gets a score, enabling heatmap
    visualization at any granularity. Temperature controls sharpness:
    lower temperature = sharper peaks around matching Gaussians.
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
    probs = probability_cloud(
        text_embedding,
        state["gaussian_store"].decoded_embeddings,
        temperature=request.temperature,
    )
    return ProbabilityResponse(
        query=request.text,
        gaussian_count=state["gaussian_store"].count,
        probabilities=probs.tolist(),
        min_score=float(probs.min()),
        max_score=float(probs.max()),
        mean_score=float(probs.mean()),
    )
