from pydantic import BaseModel, Field
from typing import Literal

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from orchestrator.config import get_llm, CRYPTO_DB, WEATHER_DB
from orchestrator.utils import get_available_entities,log_execution
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
# Listas reales desde los archivos .db
available_coins = get_available_entities(CRYPTO_DB)
crypto_prompt_text = get_crypto_agent_prompt(available_coins)

_crypto_agent_executor = create_react_agent(
    llm,
    tools=[crypto_history_tool, crypto_prediction_tool, crypto_rag_tool],
    prompt=crypto_prompt_text 
)

@log_execution
def crypto_node(state):
    """
    Nodo que ejecuta el agente de Cripto.
    FILTRO: Solo le pasa el último mensaje del usuario para evitar contaminación.
    """
    # Última instrucción del usuario
    last_message = state["messages"][-1]
    
    # El agente 'olvide' todo lo anterior.
    result = _crypto_agent_executor.invoke({"messages": [last_message]})
    
    # Devuelve el resultado para que LangGraph lo añada al historial global
    # (El usuario ve el historial, pero el agente NO lo usa para pensar)
    return {"messages": result["messages"]}

# --- 2. SUB-AGENTE WEATHER ---
# Listas reales desde los archivos .db
available_cities = get_available_entities(WEATHER_DB)
weather_prompt_text = get_weather_agent_prompt(available_cities)

_weather_agent_executor = create_react_agent(
    llm,
    tools=[weather_history_tool, weather_prediction_tool, weather_rag_tool],
    prompt=weather_prompt_text
)

@log_execution
def weather_node(state):
    """
    Nodo que ejecuta el agente de Clima.
    FILTRO: Solo le pasa el último mensaje.
    """
    last_message = state["messages"][-1]
    result = _weather_agent_executor.invoke({"messages": [last_message]})
    return {"messages": result["messages"]}


# --- 3. SUPERVISOR ---
class RouterOutput(BaseModel):
    """Decide a qué experto enviar la consulta."""
    next: Literal["Crypto_Agent", "Weather_Agent", "FINISH"]
    reasoning: str = Field(description="Razón de la elección")

supervisor_llm = llm.with_structured_output(RouterOutput)

@log_execution
def supervisor_node(state):
    messages = state["messages"]
    
    # --- ESTRATEGIA DE "RESET DE MEMORIA" PARA ENRUTAMIENTO ---
    # Para evitar que el Supervisor se confunda con las tablas SQL o conversaciones
    # anteriores, le mostramos SOLO el último mensaje del usuario.
    # Esto fuerza al modelo a evaluar la intención ACTUAL desde cero.
    # print(f"   └─ MESSAGES: {messages}")
    if not messages:
        return {"next": "FINISH"}
        
    last_user_message = messages[-1]

    response = supervisor_llm.invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + [last_user_message]
    )
    
    # Next para que el grafo sepa a dónde ir
    return {"next": response.next}