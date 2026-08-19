"""POST /messages — the primary conversational entrypoint, plus POST /messages/audio for
voice input.

Text in, text out, one exchange at a time, over a `thread_id`-scoped conversation — plus one
more shape since the Actions phase: a turn can pause instead of replying, when the agent wants
to call `send_email` (see agent/tools.py's HITL gate). Audio is a channel-edge concern, not a
different conversational shape: transcription happens *before* either route reaches the shared
turn logic below, so `_handle_turn()` (and therefore `/messages` itself) never needs to know
whether a given turn's text originated as typing or speech.

Confirming a paused thread is ordinary conversation, not a special API field — the caller just
sends another `{text, thread_id}` like any other turn. `graph.aget_state(config)` is the single
source of truth for "is this thread currently paused" — checked before *and* after every
`ainvoke`, rather than tracking a second local flag that could drift from what the checkpointer
actually has persisted. When paused, the reply's plain text goes through `classify_reply()`
(see agent/confirmation.py) to decide approve/decline/unclear *before* touching the graph at
all — an "unclear" reply (a question, an edit request) never calls `ainvoke`, since a throwaway
script confirmed `aupdate_state()` on a paused thread clears the pending interrupt rather than
coexisting with it. Only a clear approve/decline ever resumes.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request, Response, UploadFile
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, model_validator

from backend.agent.confirmation import classify_reply
from backend.infra.observability.langfuse import get_langfuse_handler

router = APIRouter(tags=["messages"])

# OpenAI's Whisper API hard-rejects anything larger — checked here, before the transcription
# adapter is ever called, so an oversized upload fails with a clear 413 rather than an opaque
# error surfacing from inside infra/transcription/whisper_adapter.py.
_MAX_AUDIO_BYTES = 25 * 1024 * 1024


class MessageRequest(BaseModel):
    text: str
    # Omit on the first call of a new conversation; the server mints one and returns it. Pass
    # it back on every following call to continue that same conversation — it's the key the
    # Postgres checkpointer uses to load prior state, so a wrong or reused thread_id
    # transparently continues (or collides with) whatever conversation already owns it. Also
    # what a paused thread is resumed on — there's no separate "confirm" field; a reply to a
    # pending confirmation looks exactly like any other message.
    thread_id: str | None = None

    @model_validator(mode="after")
    def _text_is_not_blank(self) -> MessageRequest:
        if not self.text.strip():
            raise ValueError("text must not be blank")
        return self


class PendingConfirmation(BaseModel):
    tool: str
    args: dict[str, Any]


class MessageResponse(BaseModel):
    thread_id: str
    # `None` only when a normal reply never happened — currently unreachable in practice since
    # both the "paused, unclear" and "resumed" paths always produce a reply, but kept optional
    # rather than a fake empty string for whichever future case genuinely has none.
    reply: str | None = None
    pending_confirmation: PendingConfirmation | None = None


async def _handle_turn(request: Request, thread_id: str, text: str) -> MessageResponse:
    # thread_id travels through `config`, not the graph state itself — LangGraph's
    # checkpointer keys all persistence off `config["configurable"]["thread_id"]`
    # automatically. It never needs to be a field on AgentState.
    config: dict[str, object] = {"configurable": {"thread_id": thread_id}}
    if request.app.state.langfuse_enabled:
        config["callbacks"] = [get_langfuse_handler()]

    graph = request.app.state.graph
    pending = (await graph.aget_state(config)).interrupts

    graph_input: Command[Any] | dict[str, list[HumanMessage]]
    if pending:
        pending_action = pending[0].value
        decision = await classify_reply(request.app.state.chat_model, pending_action, text)
        if decision.decision == "unclear":
            # Graph untouched — still paused on the exact same interrupt it was before this
            # request, per this module's docstring above.
            return MessageResponse(
                thread_id=thread_id,
                reply=decision.reply_if_unclear,
                pending_confirmation=PendingConfirmation(
                    tool=pending_action["tool"], args=pending_action["args"]
                ),
            )
        graph_input = Command(resume={"approved": decision.decision == "approve"})
    else:
        graph_input = {"messages": [HumanMessage(content=text)]}

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

    # A tool can't safely delete its own thread's checkpoint history mid-run (see
    # agent/tools.py's start_new_conversation) — this is where that actually happens instead,
    # *after* this turn's reply is already built from `result`, so the acknowledgment still
    # reaches the user before the thread goes empty on the next message.
    if any(
        isinstance(m, ToolMessage) and m.name == "start_new_conversation"
        for m in result["messages"]
    ):
        await graph.checkpointer.adelete_thread(thread_id)

    return MessageResponse(thread_id=thread_id, reply=reply)


@router.post("/messages")
async def send_message(request: Request, body: MessageRequest) -> MessageResponse:
    thread_id = body.thread_id or str(uuid.uuid4())
    return await _handle_turn(request, thread_id, body.text)


@router.post("/messages/audio")
async def send_audio_message(
    request: Request, audio: UploadFile, thread_id: str | None = Form(None)
) -> MessageResponse:
    content = await audio.read()
    if len(content) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio file exceeds 25MB limit")

    mime_type = audio.content_type or "application/octet-stream"
    try:
        text = await request.app.state.transcription_port.transcribe(content, mime_type=mime_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Same "no blank turns" rule MessageRequest enforces for typed text (see
    # _text_is_not_blank above) — silence or unrecognizable audio shouldn't spend a graph run
    # on an empty human turn.
    if not text.strip():
        raise HTTPException(status_code=422, detail="transcription produced no text")

    return await _handle_turn(request, thread_id or str(uuid.uuid4()), text)


@router.delete("/messages/{thread_id}", status_code=204)
async def reset_conversation(request: Request, thread_id: str) -> Response:
    """Explicit, non-conversational reset — for a caller (e.g. a future UI's "New
    Conversation" button) that wants a deterministic wipe with no model involved at all. Same
    underlying mechanism as the agent-triggered path above: `adelete_thread` on a thread_id
    that doesn't exist yet is a harmless no-op, so this is safe to call defensively too.
    """
    await request.app.state.graph.checkpointer.adelete_thread(thread_id)
    return Response(status_code=204)
