from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Literal

from orchestrator.config import get_llm
from orchestrator.tools import (
    crypto_history_tool, crypto_prediction_tool, crypto_rag_tool,
    weather_history_tool, weather_prediction_tool, weather_rag_tool
)

llm = get_llm()

# --- 1. SUB-AGENTE CRYPTO ---
crypto_agent = create_react_agent(
    llm, 
    tools=[crypto_history_tool, crypto_prediction_tool, crypto_rag_tool],
    state_modifier="Eres un experto en Criptomonedas. Tienes herramientas para ver el PASADO (history), el FUTURO (prediction) y TEORÍA (rag). Usa la que corresponda según la pregunta."
)

# --- 2. SUB-AGENTE WEATHER ---
weather_agent = create_react_agent(
    llm, 
    tools=[weather_history_tool, weather_prediction_tool, weather_rag_tool],
    state_modifier="Eres un experto Meteorológico. Tienes herramientas para ver el PASADO (history), el FUTURO (prediction) y DATOS GEOGRÁFICOS (rag). Usa la que corresponda."
)

# --- 3. SUPERVISOR ---
class RouterOutput(BaseModel):
    """Decide a qué experto enviar la consulta."""
    next: Literal["Crypto_Agent", "Weather_Agent", "FINISH"]
    reasoning: str = Field(description="Razón de la elección")

supervisor_llm = llm.with_structured_output(RouterOutput)

def supervisor_node(state):
    messages = state["messages"]
    system_prompt = (
        "Eres un Supervisor IA. Tu trabajo es enrutar preguntas a dos expertos:\n"
        "1. Crypto_Agent: Para Bitcoin, Ethereum, Solana, precios, inversiones.\n"
        "2. Weather_Agent: Para clima, temperatura, ciudades (Madrid, Tokio, etc).\n"
        "Si la conversación ha terminado o solo saludan, responde FINISH."
    )
    # Lógica simple de invocación
    response = supervisor_llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    # Devolvemos el next para que el grafo sepa a dónde ir
    return {"next": response.next}