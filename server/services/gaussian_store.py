"""Gaussian data store: loads PLY files and decodes latent features.

Handles loading LangSplat-trained PLY files containing per-Gaussian
positions and 3-dim language feature latents. Decodes all latents to
512-dim CLIP space at startup for fast cosine similarity queries.
"""

import logging

import numpy as np
import torch
from plyfile import PlyData

logger = logging.getLogger(__name__)


class GaussianStore:
    """In-memory store for Gaussian positions and decoded CLIP embeddings."""

    def __init__(self, config):
        """Initialize store.

        Args:
            config: Server Settings object (or None for testing).
        """
        self.config = config
        self.positions: np.ndarray | None = None
        self.latent_features: np.ndarray | None = None
        self.decoded_embeddings: np.ndarray | None = None
        self.count: int = 0

    @property
    def is_loaded(self) -> bool:
        """True if positions and decoded embeddings are available."""
        return (
            self.positions is not None
            and self.decoded_embeddings is not None
        )

    def load_ply(self, ply_path: str) -> None:
        """Load Gaussian data from a PLY file.

        Extracts vertex positions (x, y, z) and language feature latents
        (language_feature_0/1/2 or f_language_feature_0/1/2).

        Args:
            ply_path: Path to the PLY file.

        Raises:
            FileNotFoundError: If PLY file does not exist.
            ValueError: If language feature properties are not found.
        """
        logger.info("Loading PLY from: %s", ply_path)
        ply_data = PlyData.read(ply_path)
        vertex = ply_data["vertex"]

        # Extract positions
        x = np.array(vertex["x"], dtype=np.float32)
        y = np.array(vertex["y"], dtype=np.float32)
        z = np.array(vertex["z"], dtype=np.float32)
        self.positions = np.stack([x, y, z], axis=1)

        # Extract language features — try both naming conventions
        feat_names = [
            ("language_feature_0", "language_feature_1", "language_feature_2"),
            (
                "f_language_feature_0",
                "f_language_feature_1",
                "f_language_feature_2",
            ),
        ]

        latents = None
        for names in feat_names:
            try:
                f0 = np.array(vertex[names[0]], dtype=np.float32)
                f1 = np.array(vertex[names[1]], dtype=np.float32)
                f2 = np.array(vertex[names[2]], dtype=np.float32)
                latents = np.stack([f0, f1, f2], axis=1)
                logger.info(
                    "Found language features using naming: %s", names[0]
                )
                break
            except ValueError:
                continue

        if latents is None:
            raise ValueError(
                "PLY file missing language feature properties. "
                "Tried: language_feature_0/1/2 and f_language_feature_0/1/2"
            )

        self.latent_features = latents
        self.count = len(self.positions)
        logger.info("Loaded %d Gaussians from PLY", self.count)

    def decode_all(
        self, autoencoder: torch.nn.Module, device: str
    ) -> None:
        """Decode all latent features to 512-dim CLIP space.

        Processes in batches of 10000 for memory safety.
        Results are L2-normalized and stored as float32 numpy array.

        Args:
            autoencoder: Loaded autoencoder model with .decode() method.
            device: Torch device string ("cpu" or "cuda").

        Raises:
            RuntimeError: If latent features have not been loaded yet.
        """
        if self.latent_features is None:
            raise RuntimeError(
                "Cannot decode: latent features not loaded. Call load_ply first."
            )

        logger.info(
            "Decoding %d Gaussians to 512-dim (device=%s)...",
            self.count,
            device,
        )

        batch_size = 10000
        all_decoded = []

        latent_tensor = torch.tensor(
            self.latent_features, dtype=torch.float32
        ).to(device)

        autoencoder_device = next(autoencoder.parameters()).device
        if str(autoencoder_device) != device:
            autoencoder = autoencoder.to(device)

        with torch.no_grad():
            for i in range(0, len(latent_tensor), batch_size):
                batch = latent_tensor[i : i + batch_size]
                decoded = autoencoder.decode(batch)
                # L2 normalize each row
                decoded = decoded / decoded.norm(dim=-1, keepdim=True)
                all_decoded.append(decoded.cpu().numpy())

        self.decoded_embeddings = np.concatenate(all_decoded, axis=0).astype(
            np.float32
        )
        logger.info(
            "Decoded embeddings shape: %s", self.decoded_embeddings.shape
        )
