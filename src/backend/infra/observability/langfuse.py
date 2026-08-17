"""Langfuse tracing wiring.

Two functions, two different lifetimes, mirroring how the Langfuse v3 SDK actually works:

- `configure_langfuse()` runs once, at app startup. Constructing `Langfuse(public_key=...,
  secret_key=...)` registers that client in Langfuse's own internal singleton registry —
  confirmed by reading the SDK source rather than assumed, since the public v3 docs describe
  the OTel-based architecture at a level that doesn't make this obvious. `CallbackHandler()`
  later resolves that same registered client with no arguments needed, *as long as exactly one
  client was ever registered* — which is exactly what calling this once at startup guarantees.
- `get_langfuse_handler()` is called per-request (see routes/messages.py) and just builds a new
  lightweight handler each time. This isn't wasteful: constructing it does no I/O, it just looks
  up the already-registered client. Handing the graph a fresh handler per call, rather than one
  shared instance reused across concurrent requests, sidesteps needing to know how CallbackHandler
  handles concurrent runs internally — a question not worth answering when the safe version costs
  nothing.

Whether this ever gets used is soft-optional (see Settings.langfuse_public_key) — an
unconfigured environment must still boot and serve requests. `configure_langfuse()` returning
`False` is that "not configured" case, not an error.
"""

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from backend.config import Settings


def configure_langfuse(settings: Settings) -> bool:
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return False

    Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    return True


def get_langfuse_handler() -> CallbackHandler:
    """Only call this when `configure_langfuse()` returned True — see app.py's lifespan."""
    return CallbackHandler()
