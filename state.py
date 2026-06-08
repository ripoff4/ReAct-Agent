from typing import TypedDict, Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

    intent: Literal["ignore", "memory"]
