"""POST /messages — the primary conversational entrypoint.

Text in, text out, one exchange at a time, over a `thread_id`-scoped conversation — plus one
more shape since the Actions phase: a turn can pause instead of replying, when the agent wants
to call `send_email` (see agent/tools.py's HITL gate). No audio yet (that's a channel-edge
concern — see the architecture discussion: transcription happens before this endpoint is ever
reached, so this route doesn't need to change when audio input is added).

Confirm flow: `graph.aget_state(config)` is the single source of truth for "is this thread
currently paused waiting for confirmation" — checked before *and* after every `ainvoke`, rather
than tracking a second local flag that could drift from what the checkpointer actually has
persisted. That's two checkpointer reads per request instead of one; a deliberate trade-off,
since the extra local Postgres round trip is cheap and a second, possibly-stale source of truth
is not.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, model_validator

from backend.infra.observability.langfuse import get_langfuse_handler

router = APIRouter(tags=["messages"])


class MessageRequest(BaseModel):
    text: str | None = None
    # Omit on the first call of a new conversation; the server mints one and returns it. Pass
    # it back on every following call to continue that same conversation — it's the key the
    # Postgres checkpointer uses to load prior state, so a wrong or reused thread_id
    # transparently continues (or collides with) whatever conversation already owns it.
    thread_id: str | None = None
    # Only set when answering a `pending_confirmation` from a prior response on this same
    # thread_id — True to proceed with the gated action, False to decline it. `text` is
    # irrelevant on a confirming call; the graph resumes from where it paused, not from a new
    # human turn.
    confirm: bool | None = None

    @model_validator(mode="after")
    def _text_required_unless_confirming(self) -> MessageRequest:
        if self.confirm is None and not (self.text and self.text.strip()):
            raise ValueError("text is required unless confirm is set")
        return self


class PendingConfirmation(BaseModel):
    tool: str
    args: dict[str, Any]


class MessageResponse(BaseModel):
    thread_id: str
    # `None` exactly when `pending_confirmation` is set instead — the turn produced a question
    # for the human, not a reply from the model.
    reply: str | None = None
    pending_confirmation: PendingConfirmation | None = None


@router.post("/messages")
async def send_message(request: Request, body: MessageRequest) -> MessageResponse:
    thread_id = body.thread_id or str(uuid.uuid4())

    # thread_id travels through `config`, not the graph state itself — LangGraph's
    # checkpointer keys all persistence off `config["configurable"]["thread_id"]`
    # automatically. It never needs to be a field on AgentState.
    config: dict[str, object] = {"configurable": {"thread_id": thread_id}}
    if request.app.state.langfuse_enabled:
        config["callbacks"] = [get_langfuse_handler()]

    graph = request.app.state.graph
    is_paused = bool((await graph.aget_state(config)).interrupts)

    if is_paused and body.confirm is None:
        raise HTTPException(
            409, "thread has a pending confirmation; resend with confirm=true/false"
        )
    if not is_paused and body.confirm is not None:
        raise HTTPException(409, "no pending confirmation on this thread")

    graph_input: Command[Any] | dict[str, list[HumanMessage]] = (
        Command(resume={"approved": body.confirm})
        if is_paused
        else {"messages": [HumanMessage(content=body.text)]}
    )
    result = await graph.ainvoke(graph_input, config=config)

    still_pending = (await graph.aget_state(config)).interrupts
    if still_pending:
        value = still_pending[0].value
        return MessageResponse(
            thread_id=thread_id,
            pending_confirmation=PendingConfirmation(tool=value["tool"], args=value["args"]),
        )

    # .text (not .content): content is `str | list[str | dict]` depending on provider/response
    # shape (see agent/graph.py's reasoning for why nodes don't assume a shape either); .text
    # is LangChain's own normalized accessor for "give me this message as plain text."
    reply = result["messages"][-1].text
    return MessageResponse(thread_id=thread_id, reply=reply)
