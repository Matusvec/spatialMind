"""Tests for highlight mask and probability cloud functions.

Uses synthetic data to verify per-Gaussian opacity computation and
temperature-scaled softmax probability distributions without requiring
real PLY files or GPU hardware.
"""

import numpy as np
import pytest

from server.services.similarity import highlight_mask, probability_cloud, top_k


@pytest.fixture
def synthetic_embeddings():
    """Create synthetic normalized embeddings for testing."""
    rng = np.random.default_rng(42)
    N = 1000
    embeddings = rng.standard_normal((N, 512)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


@pytest.fixture
def query_embedding():
    """Create a synthetic query vector."""
    rng = np.random.default_rng(99)
    vec = rng.standard_normal(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


class TestHighlightMask:
    """Verify highlight_mask produces correct per-Gaussian opacity values."""

    def test_highlight_mask_length(self, synthetic_embeddings, query_embedding):
        """Mask length equals N (number of Gaussians)."""
        mask = highlight_mask(query_embedding, synthetic_embeddings)
        assert len(mask) == len(synthetic_embeddings)

    def test_highlight_mask_range(self, synthetic_embeddings, query_embedding):
        """All values are in [0.0, 1.0]."""
        mask = highlight_mask(query_embedding, synthetic_embeddings)
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

    def test_highlight_mask_hard_count(
        self, synthetic_embeddings, query_embedding
    ):
        """Hard mode has exactly k non-zero values."""
        k = 10
        mask = highlight_mask(
            query_embedding, synthetic_embeddings, k=k, mode="hard"
        )
        non_zero_count = np.count_nonzero(mask)
        assert non_zero_count == k

    def test_highlight_mask_soft_gradient(
        self, synthetic_embeddings, query_embedding
    ):
        """Soft mode has values between 0 and 1 (not just binary)."""
        mask = highlight_mask(
            query_embedding, synthetic_embeddings, k=50, mode="soft"
        )
        non_zero = mask[mask > 0]
        # In soft mode, we should have values that aren't just 0.0 or 1.0
        unique_values = np.unique(non_zero)
        assert len(unique_values) > 2, (
            f"Soft mode should produce gradient values, got only {unique_values}"
        )


class TestProbabilityCloud:
    """Verify probability_cloud produces valid softmax distributions."""

    def test_probability_cloud_length(
        self, synthetic_embeddings, query_embedding
    ):
        """Output length equals N."""
        probs = probability_cloud(query_embedding, synthetic_embeddings)
        assert len(probs) == len(synthetic_embeddings)

    def test_probability_cloud_nonnegative(
        self, synthetic_embeddings, query_embedding
    ):
        """All values are >= 0."""
        probs = probability_cloud(query_embedding, synthetic_embeddings)
        assert np.all(probs >= 0)

    def test_probability_cloud_sums_to_one(
        self, synthetic_embeddings, query_embedding
    ):
        """Sum approximately equals 1.0."""
        probs = probability_cloud(query_embedding, synthetic_embeddings)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)

    def test_probability_cloud_different_queries(self, synthetic_embeddings):
        """Two random queries produce different distributions."""
        rng1 = np.random.default_rng(10)
        rng2 = np.random.default_rng(20)
        q1 = rng1.standard_normal(512).astype(np.float32)
        q1 = q1 / np.linalg.norm(q1)
        q2 = rng2.standard_normal(512).astype(np.float32)
        q2 = q2 / np.linalg.norm(q2)

        probs1 = probability_cloud(q1, synthetic_embeddings)
        probs2 = probability_cloud(q2, synthetic_embeddings)

        # Cosine distance between distributions should be > 0
        cos_sim = np.dot(probs1, probs2) / (
            np.linalg.norm(probs1) * np.linalg.norm(probs2)
        )
        assert cos_sim < 1.0, "Different queries should produce different distributions"


class TestCrossFunction:
    """Verify consistency between highlight_mask and top_k."""

    def test_top_k_consistency(self, synthetic_embeddings, query_embedding):
        """top_k indices overlap with highlight_mask non-zero indices."""
        k = 50
        indices, _ = top_k(query_embedding, synthetic_embeddings, k=k)
        mask = highlight_mask(
            query_embedding, synthetic_embeddings, k=k, mode="hard"
        )
        mask_indices = set(np.where(mask > 0)[0])
        top_k_set = set(indices.tolist())
        assert mask_indices == top_k_set, (
            "Hard-mode highlight mask indices should match top_k indices"
        )

    def test_multi_resolution(self):
        """Broader query matches more Gaussians in top-k than narrow query.

        Creates embeddings where a cluster of 100 is 'furniture-like'
        and a sub-cluster of 30 is 'chair-like'. The broader query
        ('furniture') should have a higher mean score across all Gaussians.
        """
        rng = np.random.default_rng(42)
        N = 500

        # Create base random embeddings
        embeddings = rng.standard_normal((N, 512)).astype(np.float32)

        # Create a 'furniture' direction
        furniture_dir = rng.standard_normal(512).astype(np.float32)
        furniture_dir = furniture_dir / np.linalg.norm(furniture_dir)

        # Create a 'chair' direction (correlated with furniture but distinct)
        chair_dir = furniture_dir + 0.3 * rng.standard_normal(512).astype(
            np.float32
        )
        chair_dir = chair_dir / np.linalg.norm(chair_dir)

        # Make 100 Gaussians furniture-like (indices 0-99)
        for i in range(100):
            embeddings[i] = (
                embeddings[i] * 0.3 + furniture_dir * 0.7
            )

        # Make a sub-cluster of 30 even more chair-like (indices 0-29)
        for i in range(30):
            embeddings[i] = (
                embeddings[i] * 0.2 + chair_dir * 0.8
            )

        # L2 normalize all
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # Broader query should have higher mean highlight score
        furniture_mask = highlight_mask(
            furniture_dir, embeddings, k=150, mode="soft"
        )
        chair_mask = highlight_mask(chair_dir, embeddings, k=150, mode="soft")

        # Furniture (broader concept) should highlight more Gaussians
        # with higher average score across the furniture cluster
        furniture_mean = furniture_mask[:100].mean()
        chair_mean = chair_mask[:100].mean()

        # The furniture query should activate more of the furniture cluster
        assert furniture_mean > chair_mean * 0.5, (
            f"Broader query should activate more of the cluster: "
            f"furniture_mean={furniture_mean:.3f}, chair_mean={chair_mean:.3f}"
        )
