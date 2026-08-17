"""Classifies a plain-text reply to a pending HITL confirmation.

Confirmations for gated tools (currently just `send_email`) aren't a structured API field
anymore — the user just replies normally, in conversation. This module is what turns that
plain text into an actual decision: `classify_reply()` is called by the `/messages` route
*before* it resumes the paused graph, using the same configured chat model (no second model
config needed) with `.with_structured_output()` so the result is always one of exactly three
outcomes, not free text the route would have to parse itself.

Deliberately reuses whatever model `get_chat_model()` already constructed (see
`interfaces/api/app.py`'s `app.state.chat_model`) rather than introducing a second, "cheaper"
classifier model — one fewer moving part, at the cost of one full-size model call per
confirmation turn. Worth revisiting if that ever shows up as a real cost/latency concern.
"""

from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


class ConfirmationDecision(BaseModel):
    decision: Literal["approve", "decline", "unclear"]
    # Only meaningful when decision == "unclear" — a direct answer to whatever the user asked,
    # grounded in the pending action's own recorded args (see the prompt below), so the route
    # can hand it straight back as this turn's reply without touching the paused graph at all.
    reply_if_unclear: str | None = None


async def classify_reply(
    model: BaseChatModel, pending_action: dict[str, Any], reply_text: str
) -> ConfirmationDecision:
    prompt = (
        "The user was asked to confirm this pending action before it runs:\n"
        f"tool: {pending_action['tool']}\n"
        f"args: {pending_action['args']}\n\n"
        f"Their reply: {reply_text!r}\n\n"
        'Classify the reply as "approve" (a clear yes, proceed), "decline" (a clear no, '
        'cancel), or "unclear" (a question, an edit request, or anything else that isn\'t a '
        "clear yes/no). If unclear, set reply_if_unclear to a short, direct answer using only "
        "the pending action's args above — never invent information that isn't there."
    )
    structured_model = model.with_structured_output(ConfirmationDecision)
    result = await structured_model.ainvoke(prompt)
    return cast(ConfirmationDecision, result)
