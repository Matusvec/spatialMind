"""Autoencoder model definition matching LangSplat architecture.

The autoencoder compresses 512-dim CLIP embeddings to 3-dim latents (encoder)
and reconstructs them back to 512-dim (decoder). The decoder is used at query
time to project per-Gaussian latent features into CLIP space for cosine
similarity against text queries.

Architecture (from LangSplat paper):
- Encoder: 512 -> 256 -> 128 -> 64 -> 32 -> 3 (ReLU between all except last)
- Decoder: 3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 256 -> 512 (ReLU between all except last)
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

        # Encoder: 512 -> 256 -> 128 -> 64 -> 32 -> 3
        encoder_dims = [512, 256, 128, 64, 32, 3]
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(
                nn.Linear(encoder_dims[i], encoder_dims[i + 1])
            )
            if i < len(encoder_dims) - 2:
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: 3 -> 16 -> 32 -> 64 -> 128 -> 256 -> 256 -> 512
        decoder_dims = [3, 16, 32, 64, 128, 256, 256, 512]
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(
                nn.Linear(decoder_dims[i], decoder_dims[i + 1])
            )
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

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

    LangSplat's training may save the model as torch.save(model) (full object)
    or as a state_dict. This function handles both patterns.

    Args:
        path: Path to autoencoder checkpoint file.
        device: Device to load the model onto.

    Returns:
        Loaded Autoencoder in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint cannot be loaded.
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
        ae.load_state_dict(loaded)
        ae.eval()
        ae.to(device)
        return ae

    raise RuntimeError(
        f"Unexpected autoencoder checkpoint format: {type(loaded)}"
    )
