"""Centralized, typed application configuration.

Why pydantic-settings instead of scattered `os.getenv(...)` calls: every setting is declared
once, with a type and a default, and pydantic validates all of them *at process startup*. A
missing or malformed `DATABASE_URL` becomes an immediate crash with a clear message instead of
a confusing `TypeError` three requests later when something finally tries to use it. That
fail-fast behavior matters more once this runs unattended on a VPS instead of a dev machine.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `env_file=".env"` is a local-dev convenience: pydantic-settings reads it via
    # python-dotenv, but only as a *fallback* for values not already present in the real
    # environment. In Docker/CI/the VPS, real env vars (from docker-compose `environment:` or
    # GitHub Actions secrets) always win — there is no .env file shipped in the image at all
    # (see .dockerignore), so production never silently depends on a file that shouldn't exist
    # there.
    #
    # `extra="ignore"` matters because the *process* environment in a container carries plenty
    # of variables we don't declare here (PATH, HOSTNAME, etc.). Without it, any settings
    # source enumerating unknown keys would error; "ignore" means "validate the keys I know
    # about, don't care about the rest."
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "local"
    """One of "local" / "ci" / "production". Drives things like docs exposure and log format
    rather than branching on hostnames or other guesswork."""

    # A `Literal`, not a bare `str`, for the same fail-fast reasoning as the rest of this class
    # — passed straight to `logging.basicConfig(level=...)` (see app.py's create_app()), and a
    # typo here (e.g. "verbose") should crash at startup with a clear pydantic error, not get
    # silently accepted by `logging` and produce no logs at all.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Single DSN string (the 12-factor convention: `postgresql+asyncpg://user:pass@host:port/db`)
    # rather than separate POSTGRES_HOST/PORT/USER/... fields on this model. The docker-compose
    # `db` service still takes the individual POSTGRES_USER/PASSWORD/DB vars Postgres itself
    # expects, but the *app* only ever needs one already-assembled connection string — that's
    # also exactly the shape most managed Postgres providers (Render, Railway, RDS, etc.) hand
    # you, so this setting doesn't need to change if we ever move off the VPS.
    database_url: SecretStr

    # Set True only for local dev/CI convenience (e.g. auto-creating tables without a
    # migration). Never true in production — Alembic migrations are the only thing allowed to
    # change production schema, so drift between "what the code expects" and "what's actually
    # in the database" stays impossible to introduce by accident.
    debug: bool = False

    # --- agent (Phase 2) ---

    # "provider:model" — the format langchain's init_chat_model() expects, so switching LLM
    # provider later (e.g. to Anthropic or OpenAI) is a one-line env var change, not a code
    # change. Kept as a plain setting rather than hardcoded in agent/llm.py for that reason.
    llm_model: str = "openai:gpt-4o-mini"

    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _validate_llm_api_keys(self) -> Settings:
        is_openai = "openai" in self.llm_model or "gpt" in self.llm_model
        is_gemini = "google_genai" in self.llm_model or "gemini" in self.llm_model
        if is_openai and self.openai_api_key is None:
            raise ValueError("openai_api_key is required when using an OpenAI model")
        if is_gemini and self.gemini_api_key is None:
            raise ValueError("gemini_api_key is required when using a Gemini model")
        return self

    # Langfuse is intentionally soft-optional, unlike every other setting on this class: it's
    # observability, not a functional dependency. Wiring must never make tracing a hard
    # requirement to run the app — an unconfigured contributor/CI environment should still be
    # able to boot and serve requests, just without traces. See infra/observability/langfuse.py
    # for the None-means-disabled handling this enables.
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- actions (Actions phase) ---

    # Fail-fast, same reasoning as gemini_api_key above: these back a tool the agent can
    # actually invoke, not observability, so an unconfigured environment shouldn't boot
    # believing it can create Notion pages when it can't.
    notion_api_key: SecretStr
    # The Notion page the create-page tool creates new pages under. A page (not a database)
    # specifically — see infra/notion/adapter.py for why that distinction matters.
    notion_parent_page_id: str


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton, re-used across requests instead of re-parsing env vars each time.

    Exposed as a plain function (not a module-level constant) specifically so it can be used as
    a FastAPI dependency and overridden in tests via `app.dependency_overrides[get_settings]` —
    a module-level `settings = Settings()` would get baked in at import time and be much
    harder to swap out for a test configuration.
    """
    return Settings()
