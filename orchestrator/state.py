from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: Literal["Crypto_Agent", "Weather_Agent", "FINISH"]
