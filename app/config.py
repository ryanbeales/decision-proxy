"""
Configuration settings for decision-proxy.
Loads from environment variables or .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Upstream OpenAI-compatible backend configuration
    upstream_endpoint: str = "http://localhost:8000/v1"
    upstream_model: str = "llama-bonsai-2-27b-2bit"
    upstream_api_key: str = "sk-no-key"

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Decision engine defaults
    default_enable_thinking: bool = False
    default_max_thinking_tokens: int = 128
    request_timeout_seconds: float = 60.0
    top_logprobs: int = 25


settings = Settings()
