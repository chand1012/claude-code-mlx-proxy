import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Configuration for the Claude Code MLX Proxy"""

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mlx-community/GLM-4.5-Air-3bit")
    TRUST_REMOTE_CODE: bool = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"
    EOS_TOKEN: Optional[str] = os.getenv("EOS_TOKEN")

    # Generation settings
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))
    DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "1.0"))

    # API settings
    API_MODEL_NAME: str = os.getenv("API_MODEL_NAME", "claude-4-sonnet-20250514")

    # Logging
    VERBOSE: bool = os.getenv("VERBOSE", "false").lower() == "true"


config = Config()
