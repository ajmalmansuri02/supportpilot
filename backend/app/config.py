"""All settings come from environment variables (or the .env file at the repo root).

See .env.example for every option with an explanation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

Provider = Literal["ollama", "gemini", "openai_compatible", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM provider -------------------------------------------------------
    # "ollama" = local models, "gemini" = Google AI Studio free tier,
    # "openai_compatible" = any other OpenAI-style endpoint (Vertex, Azure, Groq...),
    # "fake" = deterministic offline model used by tests and CI.
    llm_provider: Provider = "ollama"
    chat_model: str = "qwen2.5:7b"
    # A smaller, cheaper model for simple jobs such as classification (week 9 routing).
    fast_model: str = "qwen2.5:3b"

    ollama_base_url: str = "http://localhost:11434/v1"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_compatible_base_url: str = ""
    openai_compatible_api_key: str = ""

    # --- Embeddings ---------------------------------------------------------
    # Kept separate from the chat provider: you can chat with Gemini but embed locally.
    embed_provider: Provider = "ollama"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    llm_timeout_seconds: float = 120.0
    temperature: float = 0.2

    # --- Database -----------------------------------------------------------
    database_url: str = "postgresql://supportpilot:supportpilot@localhost:5432/supportpilot"

    # --- Conversation memory ------------------------------------------------
    # When a conversation passes this many messages, older ones are summarised.
    memory_max_messages: int = 12

    # --- Pricing (USD per 1M tokens) for cost tracking ------------------------
    # Local models are free; set these to your hosted model's list price to see
    # what a conversation *would* cost. Check the provider's pricing page.
    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0

    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
