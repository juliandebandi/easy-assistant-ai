"""The graph's state schema.

`AgentState` is currently identical to LangGraph's built-in `MessagesState` — no new fields yet.
Tool dispatch (this phase's "answer directly vs. take an action" decision) doesn't need one:
`tools_condition` reads `tool_calls` straight off the last message (see graph.py). Human-in-
the-loop confirmation doesn't need one either — `interrupt()`'s pause point is already captured
by the checkpointer itself, which is the whole point of that mechanism. RAG's retrieved context
(a future phase) is the first thing likely to actually need a field here.

It's still defined as our own type now, rather than importing `MessagesState` directly
everywhere a state type is needed, for one reason: every node function, the graph builder, and
(later) tests all reference this type by name. Adding a field later is a one-line change here;
if everything imported `MessagesState` directly instead, the same change would mean hunting down
and updating every one of those call sites to point at a new type instead. One trivial subclass
now is cheaper than that refactor later.
"""

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    pass
