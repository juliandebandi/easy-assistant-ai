"""Chat model construction.

Kept separate from graph.py specifically so the graph builder never has an opinion about
*where* its model comes from — `build_graph()` takes a `BaseChatModel` as a parameter (see
graph.py) rather than calling `get_chat_model()` itself. `get_chat_model()` is what the real
app wires up at startup (app.py's lifespan); tests construct a `FakeListChatModel` instead and
pass that to the same `build_graph()` untouched. That seam is what makes the graph testable
without a real API key or a network call.
"""

from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from backend.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    kwargs: dict[str, Any] = {}
    is_openai = "openai" in settings.llm_model or "gpt" in settings.llm_model
    is_gemini = "google_genai" in settings.llm_model or "gemini" in settings.llm_model
    if is_openai and settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key.get_secret_value()
    elif is_gemini and settings.gemini_api_key:
        kwargs["api_key"] = settings.gemini_api_key.get_secret_value()

    # `**kwargs` (needed since the api_key kwarg varies by provider) breaks init_chat_model's
    # overload resolution, so mypy strict sees `Any` back instead of the `BaseChatModel` its
    # signature promises for the non-configurable-fields case actually used here.
    return cast(BaseChatModel, init_chat_model(settings.llm_model, **kwargs))
