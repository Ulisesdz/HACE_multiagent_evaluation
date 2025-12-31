from pydantic import BaseModel, Field
from typing import Literal

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from orchestrator.config import get_llm, CRYPTO_DB, WEATHER_DB
from orchestrator.utils import get_available_entities
from orchestrator.tools import (
    crypto_history_tool, crypto_prediction_tool, crypto_rag_tool,
    weather_history_tool, weather_prediction_tool, weather_rag_tool
)
from orchestrator.prompts import (
    SUPERVISOR_SYSTEM_PROMPT, 
    get_crypto_agent_prompt,
    get_weather_agent_prompt
)

llm = get_llm()

# --- 1. SUB-AGENTE CRYPTO ---
# Obtenemos las listas reales desde los archivos .db
available_coins = get_available_entities(CRYPTO_DB)
crypto_prompt_text = get_crypto_agent_prompt(available_coins)

crypto_agent = create_react_agent(
    llm,
    tools=[crypto_history_tool, crypto_prediction_tool, crypto_rag_tool],
    prompt=crypto_prompt_text 
)

# --- 2. SUB-AGENTE WEATHER ---
# Obtenemos las listas reales desde los archivos .db
available_cities = get_available_entities(WEATHER_DB)
weather_prompt_text = get_weather_agent_prompt(available_cities)

weather_agent = create_react_agent(
    llm,
    tools=[weather_history_tool, weather_prediction_tool, weather_rag_tool],
    prompt=weather_prompt_text
)

# --- 3. SUPERVISOR ---
class RouterOutput(BaseModel):
    """Decide a qué experto enviar la consulta."""
    next: Literal["Crypto_Agent", "Weather_Agent", "FINISH"]
    reasoning: str = Field(description="Razón de la elección")

supervisor_llm = llm.with_structured_output(RouterOutput)

def supervisor_node(state):
    messages = state["messages"]

    response = supervisor_llm.invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + messages
    )
    
    # Devolvemos el next para que el grafo sepa a dónde ir
    return {"next": response.next}