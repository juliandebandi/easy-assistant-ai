"""Centralized, typed application configuration.

Why pydantic-settings instead of scattered `os.getenv(...)` calls: every setting is declared
once, with a type and a default, and pydantic validates all of them *at process startup*. A
missing or malformed `DATABASE_URL` becomes an immediate crash with a clear message instead of
a confusing `TypeError` three requests later when something finally tries to use it. That
fail-fast behavior matters more once this runs unattended on a VPS instead of a dev machine.
"""

from functools import lru_cache

from pydantic import SecretStr
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

    log_level: str = "INFO"

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


@lru_cache
def get_settings() -> Settings:
    """Process-wide singleton, re-used across requests instead of re-parsing env vars each time.

    Exposed as a plain function (not a module-level constant) specifically so it can be used as
    a FastAPI dependency and overridden in tests via `app.dependency_overrides[get_settings]` —
    a module-level `settings = Settings()` would get baked in at import time and be much
    harder to swap out for a test configuration.
    """
    return Settings()
