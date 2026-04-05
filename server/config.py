"""Server configuration via Pydantic BaseSettings.

Loads from environment variables and .env file.
"""

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SpatialMind server configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    scene_dir: str = "./data/scene"
    host: str = "0.0.0.0"
    port: int = 8001
    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "laion2b_s34b_b88k"
    default_top_k: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fallback_scene_dir: str = "./data/fallback_scene"
    backboard_api_key: str = ""
    openai_api_key: str = ""
