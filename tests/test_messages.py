"""Tests for POST /messages: the endpoint contract, and that conversation state actually
persists across calls sharing a thread_id via the Postgres checkpointer — that persistence is
the one piece of real behavior worth testing at this phase. Response *content* isn't, since the
fake model returns canned replies regardless of what it's shown.
"""

import uuid

from fastapi import FastAPI
from httpx import AsyncClient


async def test_first_message_mints_a_thread_id(client: AsyncClient) -> None:
    response = await client.post("/messages", json={"text": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"]
    assert body["reply"] == "first reply"


async def test_conversation_persists_across_calls(client: AsyncClient, app: FastAPI) -> None:
    first = await client.post("/messages", json={"text": "hello"})
    thread_id = first.json()["thread_id"]

    second = await client.post("/messages", json={"text": "follow up", "thread_id": thread_id})

    assert second.json()["thread_id"] == thread_id
    assert second.json()["reply"] == "second reply"

    # The interesting assertion: did the graph actually load and extend prior state, rather
    # than starting fresh each call? A fake model can't prove that through its responses alone
    # (it ignores input) — read the checkpointer's own record of the thread directly instead.
    state = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    human_turns = [m.content for m in state.values["messages"] if m.type == "human"]
    assert human_turns == ["hello", "follow up"]


async def test_unknown_thread_id_starts_fresh(client: AsyncClient) -> None:
    # A random id, not a fixed literal: this suite runs against a real, persistent Postgres
    # locally (docker-compose's db, not torn down between runs), so a fixed thread_id would
    # only be "unseen" on the very first run and silently start asserting the wrong thing on
    # every run after that.
    thread_id = str(uuid.uuid4())

    response = await client.post("/messages", json={"text": "hi", "thread_id": thread_id})

    # A thread_id the checkpointer has never seen isn't an error — the graph just starts that
    # conversation from empty state, same as omitting thread_id entirely except the caller
    # supplies the id instead of the server minting one. Worth asserting explicitly since it'd
    # be easy to accidentally turn this into a 404 later without noticing the behavior changed.
    assert response.status_code == 200
    assert response.json()["thread_id"] == thread_id
