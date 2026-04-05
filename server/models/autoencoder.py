"""Autoencoder model definition matching LangSplat architecture.

The autoencoder compresses 512-dim CLIP embeddings to 3-dim latents (encoder)
and reconstructs them back to 512-dim (decoder). The decoder is used at query
time to project per-Gaussian latent features into CLIP space for cosine
similarity against text queries.

Architecture (from actual autoencoder.pth state_dict):
- Encoder: 512 -> 256 (BN) -> 128 (BN) -> 64 (BN) -> 32 (BN) -> 3
  Uses Linear + BatchNorm1d + ReLU blocks
- Decoder: 3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 256 -> 512
  Uses Linear + ReLU blocks (no BatchNorm)
"""

import logging
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """LangSplat autoencoder: 512-dim CLIP <-> 3-dim latent."""

    def __init__(self):
        super().__init__()

        # Encoder: Linear + BatchNorm1d + ReLU blocks
        # state_dict keys: encoder.0 (Linear 512->256), encoder.1 (BN 256),
        #   encoder.3 (Linear 256->128), encoder.4 (BN 128),
        #   encoder.6 (Linear 128->64), encoder.7 (BN 64),
        #   encoder.9 (Linear 64->32), encoder.10 (BN 32),
        #   encoder.12 (Linear 32->3)
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),   # 0
            nn.BatchNorm1d(256),   # 1
            nn.ReLU(),             # 2
            nn.Linear(256, 128),   # 3
            nn.BatchNorm1d(128),   # 4
            nn.ReLU(),             # 5
            nn.Linear(128, 64),    # 6
            nn.BatchNorm1d(64),    # 7
            nn.ReLU(),             # 8
            nn.Linear(64, 32),     # 9
            nn.BatchNorm1d(32),    # 10
            nn.ReLU(),             # 11
            nn.Linear(32, 3),      # 12
        )

        # Decoder: Linear + ReLU blocks (no BatchNorm)
        # 3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 256 -> 512
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),      # 0
            nn.ReLU(),             # 1
            nn.Linear(16, 32),     # 2
            nn.ReLU(),             # 3
            nn.Linear(32, 64),     # 4
            nn.ReLU(),             # 5
            nn.Linear(64, 128),    # 6
            nn.ReLU(),             # 7
            nn.Linear(128, 256),   # 8
            nn.ReLU(),             # 9
            nn.Linear(256, 256),   # 10
            nn.ReLU(),             # 11
            nn.Linear(256, 512),   # 12
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress 512-dim input to 3-dim latent."""
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct 3-dim latent to 512-dim CLIP space."""
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode roundtrip."""
        return self.decode(self.encode(x))


def load_autoencoder(
    path: str, device: str = "cpu"
) -> Autoencoder:
    """Load autoencoder from file, supporting both full model and state_dict.

    Args:
        path: Path to autoencoder checkpoint file.
        device: Device to load the model onto.

    Returns:
        Loaded Autoencoder in eval mode.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {path}")

    logger.info("Loading autoencoder from: %s", path)

    loaded = torch.load(path, map_location=device, weights_only=False)

    # Pattern 1: Full model object (has .decode method)
    if hasattr(loaded, "decode"):
        logger.info("Loaded autoencoder as full model object")
        loaded.eval()
        return loaded

    # Pattern 2: State dict (OrderedDict)
    if isinstance(loaded, (dict, OrderedDict)):
        logger.info("Loaded autoencoder from state_dict")
        ae = Autoencoder()
        ae.load_state_dict(loaded, strict=True)
        ae.eval()
        ae.to(device)
        return ae

    raise RuntimeError(
        f"Unexpected autoencoder checkpoint format: {type(loaded)}"
    )
