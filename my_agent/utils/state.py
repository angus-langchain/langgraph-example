from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AnyMessage


def replace_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """Keep only the last message."""
    messages = left + right
    return messages[-1:] if messages else []


class AgentState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[Sequence[BaseMessage], replace_messages]
