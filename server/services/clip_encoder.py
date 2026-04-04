"""CLIP text encoding service.

Wraps OpenCLIP to encode text queries into 512-dim normalized vectors
for cosine similarity against decoded Gaussian embeddings.
"""

import functools
import logging

import numpy as np
import open_clip
import torch

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """CLIP text encoder using OpenCLIP.

    Loads the CLIP model at initialization and provides text-to-vector
    encoding with LRU caching for repeated queries.
    """

    def __init__(
        self, model_name: str, pretrained: str, device: str
    ) -> None:
        """Initialize CLIP model and tokenizer.

        Args:
            model_name: OpenCLIP model name (e.g., "ViT-B-16").
            pretrained: Pretrained weights identifier.
            device: Torch device string ("cpu" or "cuda").
        """
        logger.info(
            "Loading CLIP model: %s (pretrained=%s, device=%s)",
            model_name,
            pretrained,
            device,
        )
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(device)
        logger.info("CLIP model loaded successfully")

    @functools.lru_cache(maxsize=256)
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query to a 512-dim normalized CLIP vector.

        Args:
            text: Natural language query string.

        Returns:
            Normalized (512,) float32 numpy array.
        """
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
