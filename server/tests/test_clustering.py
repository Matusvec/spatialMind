"""Tests for DBSCAN clustering service with dynamic epsilon."""

import numpy as np
import pytest

from server.services.clustering import cluster_gaussians, compute_dynamic_epsilon


class TestComputeDynamicEpsilon:
    """Tests for compute_dynamic_epsilon function."""

    def test_returns_positive_float_with_random_points(self):
        """compute_dynamic_epsilon with 100 random 3D points returns a positive float."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 3)).astype(np.float32)
        eps = compute_dynamic_epsilon(positions)
        assert isinstance(eps, float)
        assert eps > 0.0

    def test_uses_k_equal_to_min_samples(self):
        """compute_dynamic_epsilon with min_samples=6 uses k=6 for k-distance graph."""
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((100, 3)).astype(np.float32)
        eps_6 = compute_dynamic_epsilon(positions, min_samples=6)
        eps_10 = compute_dynamic_epsilon(positions, min_samples=10)
        # Different k values should generally produce different epsilons
        assert eps_6 != eps_10
        assert eps_6 > 0.0
        assert eps_10 > 0.0


class TestClusterGaussians:
    """Tests for cluster_gaussians function."""

    def _make_cloud(self, center, n=20, spread=0.1, rng=None):
        """Generate a tight point cloud around a center."""
        if rng is None:
            rng = np.random.default_rng(0)
        return center + rng.standard_normal((n, 3)).astype(np.float32) * spread

    def test_three_well_separated_clouds(self):
        """cluster_gaussians with 3 well-separated point clouds returns 3 clusters."""
        rng = np.random.default_rng(42)
        cloud_a = self._make_cloud(np.array([0, 0, 0]), rng=rng)
        cloud_b = self._make_cloud(np.array([10, 0, 0]), rng=rng)
        cloud_c = self._make_cloud(np.array([0, 10, 0]), rng=rng)

        positions = np.vstack([cloud_a, cloud_b, cloud_c])
        n = len(positions)
        embeddings = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        # L2 normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        scores = np.ones(n, dtype=np.float32)
        indices = np.arange(n, dtype=np.int64)

        clusters = cluster_gaussians(positions, embeddings, scores, indices)
        assert len(clusters) == 3

    def test_noise_points_excluded(self):
        """cluster_gaussians returns noise label -1 for isolated outlier points."""
        rng = np.random.default_rng(42)
        cloud = self._make_cloud(np.array([0, 0, 0]), n=20, rng=rng)
        # Add distant outliers
        outliers = np.array([[100, 100, 100], [200, 200, 200]], dtype=np.float32)
        positions = np.vstack([cloud, outliers])
        n = len(positions)
        embeddings = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        scores = np.ones(n, dtype=np.float32)
        indices = np.arange(n, dtype=np.int64)

        clusters = cluster_gaussians(positions, embeddings, scores, indices)
        # All clustered indices should not include the outlier indices
        all_clustered = set()
        for c in clusters:
            all_clustered.update(c["gaussian_indices"])
        outlier_original_indices = set(range(20, 22))
        assert outlier_original_indices.isdisjoint(all_clustered)

    def test_cluster_dict_keys(self):
        """Each cluster dict has the required keys."""
        rng = np.random.default_rng(42)
        cloud = self._make_cloud(np.array([0, 0, 0]), n=20, rng=rng)
        n = len(cloud)
        embeddings = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        scores = np.ones(n, dtype=np.float32) * 0.9
        indices = np.arange(n, dtype=np.int64)

        clusters = cluster_gaussians(cloud, embeddings, scores, indices)
        assert len(clusters) >= 1
        c = clusters[0]
        assert "label" in c
        assert "centroid" in c
        assert len(c["centroid"]) == 3
        assert "bbox" in c
        assert len(c["bbox"]) == 6
        assert "gaussian_indices" in c
        assert isinstance(c["gaussian_indices"], list)
        assert "confidence" in c
        assert isinstance(c["confidence"], float)
        assert "mean_embedding" in c
        assert c["mean_embedding"].shape == (512,)

    def test_centroid_is_mean_of_positions(self):
        """Centroid is mean of cluster positions (within tolerance)."""
        rng = np.random.default_rng(42)
        cloud = self._make_cloud(np.array([5.0, 5.0, 5.0]), n=30, spread=0.05, rng=rng)
        n = len(cloud)
        embeddings = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        scores = np.ones(n, dtype=np.float32)
        indices = np.arange(n, dtype=np.int64)

        clusters = cluster_gaussians(cloud, embeddings, scores, indices)
        assert len(clusters) >= 1
        c = clusters[0]
        # Centroid should be close to [5, 5, 5]
        np.testing.assert_allclose(c["centroid"], [5.0, 5.0, 5.0], atol=0.1)

    def test_bbox_encompasses_all_cluster_points(self):
        """bbox is [x_min, y_min, z_min, x_max, y_max, z_max] encompassing all cluster points."""
        rng = np.random.default_rng(42)
        cloud = self._make_cloud(np.array([0, 0, 0]), n=30, rng=rng)
        n = len(cloud)
        embeddings = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        scores = np.ones(n, dtype=np.float32)
        indices = np.arange(n, dtype=np.int64)

        clusters = cluster_gaussians(cloud, embeddings, scores, indices)
        assert len(clusters) >= 1
        c = clusters[0]
        bbox = c["bbox"]
        # All x, y, z of cluster points must be within bbox
        cluster_positions = cloud[np.array(c["gaussian_indices"])]
        assert np.all(cluster_positions[:, 0] >= bbox[0])  # x >= x_min
        assert np.all(cluster_positions[:, 1] >= bbox[1])  # y >= y_min
        assert np.all(cluster_positions[:, 2] >= bbox[2])  # z >= z_min
        assert np.all(cluster_positions[:, 0] <= bbox[3])  # x <= x_max
        assert np.all(cluster_positions[:, 1] <= bbox[4])  # y <= y_max
        assert np.all(cluster_positions[:, 2] <= bbox[5])  # z <= z_max

    def test_confidence_is_mean_similarity(self):
        """Confidence is mean of similarity scores for cluster members."""
        rng = np.random.default_rng(42)
        cloud = self._make_cloud(np.array([0, 0, 0]), n=20, rng=rng)
        n = len(cloud)
        embeddings = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        scores = np.linspace(0.5, 1.0, n).astype(np.float32)
        indices = np.arange(n, dtype=np.int64)

        clusters = cluster_gaussians(cloud, embeddings, scores, indices)
        assert len(clusters) >= 1
        c = clusters[0]
        # Confidence should be the mean of scores for the cluster members
        # We use local indices since these are the local (0..n-1) indices mapped to original
        member_local_indices = [
            i for i in range(n)
            if int(indices[i]) in c["gaussian_indices"]
        ]
        expected_confidence = float(np.mean(scores[member_local_indices]))
        assert abs(c["confidence"] - expected_confidence) < 1e-5

    def test_identical_points_single_cluster(self):
        """cluster_gaussians with all identical points returns 1 cluster (not N singletons)."""
        n = 30
        positions = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (n, 1))
        embeddings = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        scores = np.ones(n, dtype=np.float32)
        indices = np.arange(n, dtype=np.int64)

        clusters = cluster_gaussians(positions, embeddings, scores, indices)
        assert len(clusters) == 1

    def test_sorted_by_confidence_descending(self):
        """Clusters are sorted by confidence descending."""
        rng = np.random.default_rng(42)
        cloud_a = self._make_cloud(np.array([0, 0, 0]), rng=rng)
        cloud_b = self._make_cloud(np.array([10, 0, 0]), rng=rng)

        positions = np.vstack([cloud_a, cloud_b])
        n = len(positions)
        embeddings = np.random.default_rng(1).standard_normal((n, 512)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Give different scores to each cloud
        scores = np.concatenate([
            np.full(20, 0.5, dtype=np.float32),
            np.full(20, 0.9, dtype=np.float32),
        ])
        indices = np.arange(n, dtype=np.int64)

        clusters = cluster_gaussians(positions, embeddings, scores, indices)
        if len(clusters) >= 2:
            for i in range(len(clusters) - 1):
                assert clusters[i]["confidence"] >= clusters[i + 1]["confidence"]
