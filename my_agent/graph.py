from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from my_agent.utils.state import AgentState
from my_agent.utils.nodes import call_model, call_tool_node, should_continue


class GraphConfig(TypedDict):
    """Configuration for the graph."""
    model_name: Literal["anthropic", "openai"]


def create_graph():
    """Create and compile the agent graph."""
    # Define a new graph with state and config
    workflow = StateGraph(AgentState, config_schema=GraphConfig)

    # Add nodes to the graph
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool_node)

    # Set the entry point
    workflow.add_edge(START, "agent")

    # Add conditional edge from agent node
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # Add edge from action back to agent
    workflow.add_edge("action", "agent")

    # Compile the graph
    graph = workflow.compile()

    return graph


# Create the compiled graph
graph = create_graph()
