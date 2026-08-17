"""Graph topology.

Two nodes: `chat`, which calls a tool-bound model on the accumulated message history, and
`tools`, LangGraph's prebuilt `ToolNode`, which executes whatever tool calls `chat` requested.
`tools_condition` (also prebuilt) inspects `state["messages"][-1].tool_calls` after every
`chat` turn and routes to `tools` when there are any, or to `END` when the model replied
directly instead of calling a tool — that's the "answer directly vs. take an action" decision,
made by the model itself rather than a hand-written router node. No new `AgentState` field was
needed for this: `tools_condition` reads `tool_calls` straight off the last message, so there's
nothing to add to the state schema just to support routing.

An earlier draft of this file sketched a multi-agent supervisor instead — a router node
dispatching to a separate node per integration (mailing, calendar, Notion, Telegram,
WhatsApp). That was deliberately rejected: with the action surface at two tools, decomposing
into independent agents (each with its own prompt/state/loop) would be structure built for a
decision that doesn't exist yet, the same reasoning this file already applied to routing itself
before this phase. It gets reconsidered if and when a specific action proves it needs a genuinely
independent multi-turn loop that `tools_condition`'s single loop can't express.

`build_graph()` returns the *uncompiled* `StateGraph` rather than a ready-to-use compiled graph,
deliberately: compiling requires a checkpointer, and the checkpointer's connection pool has a
lifecycle (opened/closed) that belongs to whoever owns that lifecycle — the FastAPI lifespan in
interfaces/api/app.py — not to this module. Keeping graph.py's only job "define the topology"
means it has zero opinions about persistence or process lifetime, and can be compiled with a
throwaway in-memory checkpointer in a unit test just as easily as with Postgres in production.
"""

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.agent.state import AgentState


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool]) -> StateGraph[AgentState]:
    model_with_tools = model.bind_tools(tools)

    async def chat(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = await model_with_tools.ainvoke(state["messages"])
        # Node functions return a *partial* state update, not the full new state — LangGraph
        # merges it into the existing state using each field's reducer. `messages` uses
        # `add_messages` (via MessagesState), whose reducer appends rather than overwrites, so
        # returning a single-item list here correctly means "add this one reply," not "replace
        # the whole conversation history with just this."
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("chat", chat)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "chat")
    builder.add_conditional_edges("chat", tools_condition)
    builder.add_edge("tools", "chat")
    return builder
