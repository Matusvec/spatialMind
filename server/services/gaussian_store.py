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
    """In-memory store for Gaussian positions and decoded CLIP embeddings.

    Supports multi-level semantic features (LangSplat sem0/1/2):
      - sem0: whole objects (chair, table, couch)
      - sem1: parts (chair leg, armrest, table top)
      - sem2: subparts (hair, button, finger)
    """

    def __init__(self, config):
        """Initialize store.

        Args:
            config: Server Settings object (or None for testing).
        """
        self.config = config
        self.positions: np.ndarray | None = None
        self.latent_features: np.ndarray | None = None
        self.decoded_embeddings: np.ndarray | None = None
        # Multi-level semantic embeddings: {0: array, 1: array, 2: array}
        self.semantic_levels: dict[int, np.ndarray] = {}
        self.decoded_levels: dict[int, np.ndarray] = {}
        self.count: int = 0

    @property
    def is_loaded(self) -> bool:
        """True if positions and language features have been loaded from PLY."""
        return self.positions is not None and self.latent_features is not None

    @property
    def is_decoded(self) -> bool:
        """True if latent features have been decoded to 512-dim CLIP space."""
        return self.decoded_embeddings is not None or len(self.decoded_levels) > 0

    def get_embeddings(self, level: int = 0, autoencoder=None, device: str = "cpu") -> np.ndarray | None:
        """Get decoded embeddings for a specific semantic level.

        Level 0 is always pre-decoded at startup. Levels 1/2 are decoded
        on-demand from 3-dim latents (takes ~4s on GPU, cached after first call).

        Args:
            level: 0=whole objects, 1=parts, 2=subparts
            autoencoder: Required for on-demand decode of levels 1/2.
            device: Torch device for on-demand decode.

        Returns:
            Decoded 512-dim embeddings array, or None if not available.
        """
        if level in self.decoded_levels:
            return self.decoded_levels[level]
        # On-demand decode for levels 1/2
        if level in self.semantic_levels and autoencoder is not None:
            logger.info("On-demand decode of level %d...", level)
            latents = self.semantic_levels[level]
            batch_size = 50000
            all_decoded = []
            latent_tensor = torch.tensor(latents, dtype=torch.float32).to(device)
            with torch.no_grad():
                for i in range(0, len(latent_tensor), batch_size):
                    batch = latent_tensor[i : i + batch_size]
                    decoded = autoencoder.decode(batch)
                    decoded = decoded / decoded.norm(dim=-1, keepdim=True)
                    all_decoded.append(decoded.cpu().numpy())
            self.decoded_levels[level] = np.concatenate(all_decoded, axis=0).astype(np.float32)
            logger.info(
                "Decoded level %d: %s (%s, %.1f GB)",
                level, self.decoded_levels[level].shape,
                self.decoded_levels[level].dtype,
                self.decoded_levels[level].nbytes / 1e9,
            )
            return self.decoded_levels[level]
        return self.decoded_embeddings

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
        import gc

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

        # Free the 5GB PlyData object immediately
        del ply_data, vertex
        gc.collect()

        logger.info("Loaded %d Gaussians from PLY", self.count)

    def load_semantic_level(self, ply_path: str, level: int) -> None:
        """Load only language features from a semantic PLY (no positions).

        Extracts language_feature_0/1/2, frees the PlyData immediately.

        Args:
            ply_path: Path to semantic_N.ply file.
            level: Semantic level (1=parts, 2=subparts).
        """
        import gc

        logger.info("Loading semantic level %d from: %s", level, ply_path)
        ply_data = PlyData.read(ply_path)
        data = ply_data["vertex"]

        f0 = np.array(data["language_feature_0"], dtype=np.float32)
        f1 = np.array(data["language_feature_1"], dtype=np.float32)
        f2 = np.array(data["language_feature_2"], dtype=np.float32)
        self.semantic_levels[level] = np.stack([f0, f1, f2], axis=1)

        level_names = {1: "parts", 2: "subparts"}
        count = len(f0)

        # Free the PlyData immediately
        del ply_data, data, f0, f1, f2
        gc.collect()

        logger.info(
            "Loaded semantic_%d (%s): %d features",
            level, level_names.get(level, ""), count,
        )

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

        batch_size = 50000
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
                decoded = decoded / decoded.norm(dim=-1, keepdim=True)
                all_decoded.append(decoded.cpu().numpy())

        self.decoded_embeddings = np.concatenate(all_decoded, axis=0).astype(np.float32)
        logger.info(
            "Decoded embeddings: %s (%s, %.1f GB)",
            self.decoded_embeddings.shape,
            self.decoded_embeddings.dtype,
            self.decoded_embeddings.nbytes / 1e9,
        )
