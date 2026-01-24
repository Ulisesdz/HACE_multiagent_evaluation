from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Historial de mensajes (se van acumulando)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Próximo paso en el grafo: Los 3 especialistas financieros o FINISH
    next: Literal["Technical_Analyst", "Fundamental_Analyst", "Risk_Officer", "FINISH"]