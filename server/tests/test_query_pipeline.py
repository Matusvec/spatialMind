"""Tests for the core semantic query pipeline.

Uses synthetic data to verify autoencoder architecture, Gaussian store,
CLIP encoder, and similarity computation without requiring real PLY files
or GPU hardware.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


class TestAutoencoder:
    """Verify autoencoder matches LangSplat architecture."""

    def test_decoder_output_dim_is_512(self):
        """Decoder input dim=3, output dim=512."""
        from server.models.autoencoder import Autoencoder

        ae = Autoencoder()
        dummy_input = torch.randn(5, 3)
        output = ae.decode(dummy_input)
        assert output.shape == (5, 512)

    def test_encoder_output_dim_is_3(self):
        """Encoder compresses to 3 dimensions."""
        from server.models.autoencoder import Autoencoder

        ae = Autoencoder()
        dummy_input = torch.randn(5, 512)
        output = ae.encode(dummy_input)
        assert output.shape == (5, 3)

    def test_decoder_layer_dims(self):
        """Decoder layers follow [16, 32, 64, 128, 256, 256, 512]."""
        from server.models.autoencoder import Autoencoder

        ae = Autoencoder()
        expected_dims = [3, 16, 32, 64, 128, 256, 256, 512]
        decoder_linears = [
            m for m in ae.decoder.modules() if isinstance(m, nn.Linear)
        ]
        for i, layer in enumerate(decoder_linears):
            assert layer.in_features == expected_dims[i], (
                f"Layer {i}: expected in_features={expected_dims[i]}, "
                f"got {layer.in_features}"
            )
            assert layer.out_features == expected_dims[i + 1], (
                f"Layer {i}: expected out_features={expected_dims[i + 1]}, "
                f"got {layer.out_features}"
            )

    def test_forward_roundtrip(self):
        """Forward pass encodes then decodes."""
        from server.models.autoencoder import Autoencoder

        ae = Autoencoder()
        dummy_input = torch.randn(5, 512)
        output = ae(dummy_input)
        assert output.shape == (5, 512)


class TestGaussianStore:
    """Verify Gaussian store loads PLY and decodes features."""

    def _create_test_ply(self, tmp_path, n_gaussians=100):
        """Create a minimal PLY file with positions and language features."""
        from plyfile import PlyData, PlyElement

        vertices = np.zeros(
            n_gaussians,
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("language_feature_0", "f4"),
                ("language_feature_1", "f4"),
                ("language_feature_2", "f4"),
            ],
        )
        vertices["x"] = np.random.randn(n_gaussians).astype(np.float32)
        vertices["y"] = np.random.randn(n_gaussians).astype(np.float32)
        vertices["z"] = np.random.randn(n_gaussians).astype(np.float32)
        vertices["language_feature_0"] = np.random.randn(n_gaussians).astype(
            np.float32
        )
        vertices["language_feature_1"] = np.random.randn(n_gaussians).astype(
            np.float32
        )
        vertices["language_feature_2"] = np.random.randn(n_gaussians).astype(
            np.float32
        )

        el = PlyElement.describe(vertices, "vertex")
        ply_path = str(tmp_path / "test.ply")
        PlyData([el], text=False).write(ply_path)
        return ply_path

    def test_load_ply_extracts_positions(self, tmp_path):
        """load_ply() extracts positions as (N, 3) array."""
        from server.services.gaussian_store import GaussianStore

        ply_path = self._create_test_ply(tmp_path, n_gaussians=50)
        store = GaussianStore(config=None)
        store.load_ply(ply_path)
        assert store.positions.shape == (50, 3)

    def test_load_ply_extracts_language_features(self, tmp_path):
        """load_ply() extracts language features as (N, 3) array."""
        from server.services.gaussian_store import GaussianStore

        ply_path = self._create_test_ply(tmp_path, n_gaussians=50)
        store = GaussianStore(config=None)
        store.load_ply(ply_path)
        assert store.latent_features.shape == (50, 3)

    def test_decode_all_produces_512_dim(self, tmp_path):
        """decode_all() produces (N, 512) normalized embeddings."""
        from server.models.autoencoder import Autoencoder
        from server.services.gaussian_store import GaussianStore

        ply_path = self._create_test_ply(tmp_path, n_gaussians=30)
        store = GaussianStore(config=None)
        store.load_ply(ply_path)

        ae = Autoencoder()
        ae.eval()
        store.decode_all(ae, "cpu")

        assert store.decoded_embeddings.shape == (30, 512)
        # Verify L2-normalized (each row norm ~1.0)
        norms = np.linalg.norm(store.decoded_embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_is_loaded_property(self, tmp_path):
        """is_loaded returns True after successful load and decode."""
        from server.models.autoencoder import Autoencoder
        from server.services.gaussian_store import GaussianStore

        store = GaussianStore(config=None)
        assert not store.is_loaded

        ply_path = self._create_test_ply(tmp_path, n_gaussians=10)
        store.load_ply(ply_path)

        ae = Autoencoder()
        ae.eval()
        store.decode_all(ae, "cpu")
        assert store.is_loaded


class TestSimilarity:
    """Verify top-K retrieval and soft scores."""

    def _make_embeddings(self, n=1000, dim=512):
        """Create random normalized embeddings."""
        emb = np.random.randn(n, dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    def _make_query(self, dim=512):
        """Create a random normalized query vector."""
        q = np.random.randn(dim).astype(np.float32)
        q = q / np.linalg.norm(q)
        return q

    def test_top_k_returns_correct_count(self):
        """top_k returns exactly k results."""
        from server.services.similarity import top_k

        embeddings = self._make_embeddings(1000)
        query = self._make_query()
        indices, scores = top_k(query, embeddings, k=10)
        assert len(indices) == 10
        assert len(scores) == 10

    def test_top_k_scores_sorted_descending(self):
        """top_k scores are in descending order."""
        from server.services.similarity import top_k

        embeddings = self._make_embeddings(1000)
        query = self._make_query()
        _, scores = top_k(query, embeddings, k=20)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score {i} ({scores[i]}) < Score {i+1} ({scores[i+1]})"
            )

    def test_top_k_scores_in_valid_range(self):
        """top_k scores are between -1.0 and 1.0."""
        from server.services.similarity import top_k

        embeddings = self._make_embeddings(500)
        query = self._make_query()
        _, scores = top_k(query, embeddings, k=50)
        assert np.all(scores >= -1.0)
        assert np.all(scores <= 1.0)

    def test_soft_scores_returns_all(self):
        """soft_scores returns N scores for N embeddings."""
        from server.services.similarity import soft_scores

        embeddings = self._make_embeddings(200)
        query = self._make_query()
        result = soft_scores(query, embeddings)
        assert result.shape == (200,)
        assert result.dtype == np.float32

    def test_different_queries_produce_different_results(self):
        """Different query vectors should produce different top-K index sets."""
        from server.services.similarity import top_k

        embeddings = self._make_embeddings(1000)
        query1 = self._make_query()
        query2 = self._make_query()
        indices1, _ = top_k(query1, embeddings, k=10)
        indices2, _ = top_k(query2, embeddings, k=10)
        # With random vectors in 512-dim, different queries almost certainly
        # produce different top-10 index sets
        assert not np.array_equal(indices1, indices2), (
            "Different queries should not produce identical index sets"
        )


class TestCLIPEncoder:
    """Verify CLIP encoder produces correct output shape."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CLIP test on CPU is slow; skipping for fast CI",
    )
    def test_encode_text_returns_512_dim(self):
        """encode_text returns a (512,) normalized float32 vector."""
        from server.services.clip_encoder import CLIPEncoder

        encoder = CLIPEncoder("ViT-B-16", "laion2b_s34b_b88k", "cpu")
        result = encoder.encode_text("chair")
        assert result.shape == (512,)
        assert result.dtype == np.float32
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-4
