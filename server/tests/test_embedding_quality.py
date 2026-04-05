"""Tests that decoded Gaussian embeddings are semantically meaningful.

These tests catch the #1 failure mode: autoencoder.pth not matching the PLY
latent space (different Colab training runs). When mismatched, all decoded
embeddings collapse to nearly the same direction, making LERF relevancy
scores cluster around 0.5 and returning ~8 Gaussians instead of thousands.

Run with: python -m pytest server/tests/test_embedding_quality.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENE_DIR = os.environ.get("SCENE_DIR", "./data/scene")


def _skip_if_no_scene():
    ae_path = os.path.join(SCENE_DIR, "autoencoder.pth")
    ply_path = os.path.join(SCENE_DIR, "semantic_0.ply")
    if not os.path.exists(ae_path) or not os.path.exists(ply_path):
        pytest.skip("Scene data not available (need autoencoder.pth + semantic_0.ply)")


@pytest.fixture(scope="module")
def decoded_state():
    """Load autoencoder + PLY and decode level 0 embeddings."""
    _skip_if_no_scene()

    from server.models.autoencoder import load_autoencoder
    from server.services.gaussian_store import GaussianStore

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ae_path = os.path.join(SCENE_DIR, "autoencoder.pth")
    autoencoder = load_autoencoder(ae_path, device)

    class FakeConfig:
        scene_dir = SCENE_DIR

    store = GaussianStore(FakeConfig())
    store.load_ply(os.path.join(SCENE_DIR, "semantic_0.ply"))
    store.decode_all(autoencoder, device)
    store.decoded_levels[0] = store.decoded_embeddings

    return {
        "store": store,
        "autoencoder": autoencoder,
        "embeddings": store.decoded_embeddings,
        "device": device,
    }


@pytest.fixture(scope="module")
def clip_encoder(decoded_state):
    """Load the CLIP text encoder."""
    from server.services.clip_encoder import CLIPEncoder

    return CLIPEncoder("ViT-B-32", "openai", decoded_state["device"])


@pytest.fixture(scope="module")
def negative_embeddings(clip_encoder):
    """Encode canonical negatives."""
    from server.services.similarity import CANONICAL_NEGATIVES

    return np.stack([clip_encoder.encode_text(neg) for neg in CANONICAL_NEGATIVES])


# ---------------------------------------------------------------------------
# Test 1: Embedding diversity (catches autoencoder mismatch)
# ---------------------------------------------------------------------------


class TestEmbeddingDiversity:
    """Verify decoded embeddings are diverse, not collapsed."""

    def test_centroid_norm_is_low(self, decoded_state):
        """Centroid of random embeddings should have low norm (diverse directions).

        If centroid_norm > 0.5, all embeddings point the same way = AE mismatch.
        A well-trained AE gives centroid_norm < 0.15 for 1000 random samples.
        """
        emb = decoded_state["embeddings"]
        sample = emb[np.random.choice(len(emb), min(1000, len(emb)), replace=False)]
        centroid = sample.mean(axis=0).astype(np.float32)
        centroid_norm = float(np.linalg.norm(centroid))
        print(f"Centroid norm: {centroid_norm:.4f} (want < 0.3)")
        assert centroid_norm < 0.3, (
            f"Decoded embeddings are collapsed (centroid_norm={centroid_norm:.3f}). "
            f"The autoencoder.pth does not match the PLY latent space. "
            f"Download the correct ae_ckpt/best_ckpt.pth from your Colab training run."
        )

    def test_pairwise_similarity_spread(self, decoded_state):
        """Random pairs of decoded embeddings should NOT all have the same similarity.

        If std(pairwise_cosine) < 0.01, embeddings are degenerate.
        """
        emb = decoded_state["embeddings"]
        idx = np.random.choice(len(emb), min(500, len(emb)), replace=False)
        sample = emb[idx].astype(np.float32)
        # Normalize
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        sample = sample / norms
        # Pairwise cosine similarity for first 200
        sub = sample[:200]
        sims = sub @ sub.T
        # Exclude diagonal
        mask = ~np.eye(len(sub), dtype=bool)
        off_diag = sims[mask]
        std = float(np.std(off_diag))
        mean = float(np.mean(off_diag))
        print(f"Pairwise cosine: mean={mean:.4f}, std={std:.4f} (want std > 0.03)")
        assert std > 0.03, (
            f"Pairwise cosine similarity has near-zero variance (std={std:.4f}). "
            f"All embeddings point in the same direction = AE mismatch."
        )

    def test_embeddings_are_float32(self, decoded_state):
        """Embeddings must be float32, not float16, for LERF precision."""
        emb = decoded_state["embeddings"]
        assert emb.dtype == np.float32, (
            f"Decoded embeddings are {emb.dtype}, expected float32. "
            f"Float16 destroys LERF score discrimination."
        )

    def test_embeddings_are_unit_normalized(self, decoded_state):
        """Each embedding should be approximately L2-normalized."""
        emb = decoded_state["embeddings"]
        sample = emb[np.random.choice(len(emb), min(500, len(emb)), replace=False)]
        norms = np.linalg.norm(sample, axis=1)
        mean_norm = float(norms.mean())
        print(f"Mean L2 norm: {mean_norm:.4f} (want ~1.0)")
        assert 0.9 < mean_norm < 1.1, (
            f"Embeddings are not unit-normalized (mean_norm={mean_norm:.3f}). "
            f"Cosine similarity requires L2-normalized vectors."
        )


# ---------------------------------------------------------------------------
# Test 2: LERF relevancy produces meaningful scores
# ---------------------------------------------------------------------------


class TestLERFRelevancy:
    """Verify LERF scores discriminate between objects and background."""

    def test_raw_cosine_has_discrimination(self, decoded_state, clip_encoder):
        """Raw cosine similarity should discriminate between queries.

        Even with collapsed AE embeddings, the top 0.1% of cosine scores for
        'chair' should be meaningfully higher than the median.
        """
        emb = decoded_state["embeddings"]
        text_emb = clip_encoder.encode_text("chair")
        raw_cos = emb @ text_emb
        p50 = float(np.median(raw_cos))
        p999 = float(np.percentile(raw_cos, 99.9))
        spread = p999 - p50
        print(f"Raw cosine: p50={p50:.5f}, p99.9={p999:.5f}, spread={spread:.5f} (want > 0.005)")
        assert spread > 0.005, (
            f"Raw cosine has no discrimination (spread={spread:.5f}). "
            f"AE embeddings are completely degenerate."
        )

    def test_different_queries_have_different_top_sets(self, decoded_state, clip_encoder):
        """Top 0.1% for 'chair' and 'floor' should be mostly different Gaussians."""
        emb = decoded_state["embeddings"]
        k = max(100, int(len(emb) * 0.001))
        sets = {}
        for query in ["chair", "floor"]:
            text_emb = clip_encoder.encode_text(query)
            cos = emb @ text_emb
            top_k_idx = np.argpartition(-cos, k)[:k]
            sets[query] = set(top_k_idx.tolist())
        overlap = len(sets["chair"] & sets["floor"]) / max(len(sets["chair"]), 1)
        print(f"chair/floor overlap: {overlap:.2%} (want < 50%)")
        assert overlap < 0.5, (
            f"Top Gaussians for 'chair' and 'floor' overlap {overlap:.0%}. "
            f"Embeddings have no semantic discrimination."
        )


# ---------------------------------------------------------------------------
# Test 3: Full pipeline produces enough highlighted Gaussians
# ---------------------------------------------------------------------------


class TestHighlightCoverage:
    """Verify the full grounding pipeline returns enough Gaussians for visible highlighting."""

    def test_ground_query_returns_many_indices(self, decoded_state, clip_encoder, negative_embeddings):
        """ground_query('chair') must return >1000 highlight indices."""
        from server.services.dense_query import ground_query

        state = {
            "gaussian_store": decoded_state["store"],
            "clip_encoder": clip_encoder,
            "negative_embeddings": negative_embeddings,
        }
        result = ground_query(text="chair", state=state, persist_instances=False)
        count = len(result.get("highlight_indices", []))
        print(f"ground_query('chair') returned {count} highlight indices")
        assert count > 1000, (
            f"ground_query returned only {count} highlight indices for 'chair'. "
            f"Expected >1000 for visible object highlighting."
        )

    def test_ground_query_person_returns_many_indices(self, decoded_state, clip_encoder, negative_embeddings):
        """ground_query('person') must return >1000 highlight indices."""
        from server.services.dense_query import ground_query

        state = {
            "gaussian_store": decoded_state["store"],
            "clip_encoder": clip_encoder,
            "negative_embeddings": negative_embeddings,
        }
        result = ground_query(text="person", state=state, persist_instances=False)
        count = len(result.get("highlight_indices", []))
        print(f"ground_query('person') returned {count} highlight indices")
        assert count > 1000, (
            f"ground_query returned only {count} highlight indices for 'person'. "
            f"Expected >1000 for visible object highlighting."
        )

    def test_highlight_match_indices_match_highlight_indices(self, decoded_state, clip_encoder, negative_embeddings):
        """highlight_match.indices should equal highlight_indices (no data loss in adapter)."""
        from server.services.dense_query import ground_query

        state = {
            "gaussian_store": decoded_state["store"],
            "clip_encoder": clip_encoder,
            "negative_embeddings": negative_embeddings,
        }
        result = ground_query(text="chair", state=state, persist_instances=False)
        hi = set(result.get("highlight_indices", []))
        hm = set(result.get("highlight_match", {}).get("indices", []))
        assert hi == hm, (
            f"highlight_indices ({len(hi)}) != highlight_match.indices ({len(hm)}). "
            f"Data is being lost between dense_query and the frontend."
        )


# ---------------------------------------------------------------------------
# Test 4: Autoencoder roundtrip quality
# ---------------------------------------------------------------------------


class TestAutoencoderQuality:
    """Verify the autoencoder can roundtrip CLIP vectors with reasonable fidelity."""

    def test_roundtrip_similarity(self, decoded_state, clip_encoder):
        """Encode a text vector through the AE and check roundtrip similarity."""
        autoencoder = decoded_state["autoencoder"]
        device = decoded_state["device"]

        text_emb = clip_encoder.encode_text("chair")
        text_tensor = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = autoencoder.encode(text_tensor)
            reconstructed = autoencoder.decode(latent)
            reconstructed = reconstructed / reconstructed.norm(dim=-1, keepdim=True)

        recon_np = reconstructed.cpu().numpy().flatten()
        sim = float(np.dot(text_emb, recon_np))
        print(f"AE roundtrip cosine similarity: {sim:.4f} (want > 0.5)")
        assert sim > 0.5, (
            f"Autoencoder roundtrip similarity is {sim:.3f} (expected > 0.5). "
            f"The AE cannot faithfully reconstruct CLIP vectors. "
            f"This means decoded PLY embeddings are meaningless."
        )

    def test_latent_scale_matches_ply(self, decoded_state, clip_encoder):
        """AE-encoded CLIP vectors should have similar scale to PLY latents."""
        autoencoder = decoded_state["autoencoder"]
        device = decoded_state["device"]
        store = decoded_state["store"]

        # Encode 10 text queries through AE encoder
        queries = ["chair", "person", "wall", "floor", "table"]
        ae_latents = []
        for q in queries:
            text_emb = clip_encoder.encode_text(q)
            text_tensor = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                latent = autoencoder.encode(text_tensor)
            ae_latents.append(latent.cpu().numpy().flatten())
        ae_latents = np.array(ae_latents)
        ae_scale = float(np.abs(ae_latents).mean())

        # PLY latent scale
        ply_latents = store.latent_features[:1000]
        ply_scale = float(np.abs(ply_latents).mean())

        ratio = max(ae_scale, ply_scale) / max(min(ae_scale, ply_scale), 1e-8)
        print(f"AE latent scale: {ae_scale:.4f}, PLY latent scale: {ply_scale:.4f}, ratio: {ratio:.1f}x")
        assert ratio < 10, (
            f"Latent scale mismatch: AE produces scale {ae_scale:.4f} but PLY has {ply_scale:.4f} "
            f"({ratio:.0f}x difference). The autoencoder was trained on different data."
        )
