from pydantic import BaseModel, Field
from typing import Literal

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from orchestrator.config import get_llm, CRYPTO_DB
from orchestrator.utils import get_available_entities, log_execution

# --- IMPORTACIÓN DE TOOLS --- 
from orchestrator.tools import (
    crypto_history_tool,
    crypto_prediction_tool,
    crypto_chart_tool,
    crypto_rag_tool,
    crypto_news_tool,
    crypto_volatility_tool
)

# --- IMPORTACIÓN DE PROMPTS ---
from orchestrator.prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    get_technical_agent_prompt,
    get_fundamental_agent_prompt,
    get_risk_agent_prompt
)

llm = get_llm()

# Monedas disponibles para dar contexto al Technical Agent
available_coins = get_available_entities(CRYPTO_DB)

# ============================================================
# 1. AGENTE TÉCNICO (THE QUANT)
# ============================================================
technical_prompt_text = get_technical_agent_prompt(available_coins)

_technical_agent = create_react_agent(
    llm,
    tools=[crypto_history_tool, crypto_prediction_tool, crypto_chart_tool],
    prompt=technical_prompt_text, 
)

@log_execution
def technical_node(state):
    """Nodo del Analista Técnico."""
    last_message = state["messages"][-1]
    result = _technical_agent.invoke({"messages": [last_message]})
    return {"messages": result["messages"]}


# ============================================================
# 2. AGENTE FUNDAMENTAL (THE RESEARCHER)
# ============================================================
fundamental_prompt_text = get_fundamental_agent_prompt()

_fundamental_agent = create_react_agent(
    llm,
    tools=[crypto_rag_tool, crypto_news_tool],
    prompt=fundamental_prompt_text,
)

@log_execution
def fundamental_node(state):
    """Nodo del Investigador Fundamental."""
    last_message = state["messages"][-1]
    result = _fundamental_agent.invoke({"messages": [last_message]})
    return {"messages": result["messages"]}


# ============================================================
# 3. AGENTE DE RIESGOS (RISK OFFICER)
# ============================================================
risk_prompt_text = get_risk_agent_prompt()

_risk_agent = create_react_agent(
    llm,
    tools=[crypto_volatility_tool],
    prompt=risk_prompt_text,
)

@log_execution
def risk_node(state):
    """Nodo del Gestor de Riesgos."""
    last_message = state["messages"][-1]
    result = _risk_agent.invoke({"messages": [last_message]})
    return {"messages": result["messages"]}


# ============================================================
# 4. SUPERVISOR (CHIEF INVESTMENT OFFICER)
# ============================================================
class RouterOutput(BaseModel):
    """Decide a qué miembro del comité de inversión enviar la consulta."""
    next: Literal["Technical_Analyst", "Fundamental_Analyst", "Risk_Officer", "FINISH"]
    reasoning: str = Field(description="Por qué elegiste a este experto.")

supervisor_llm = llm.with_structured_output(RouterOutput)

@log_execution
def supervisor_node(state):
    messages = state["messages"]
    if not messages:
        return {"next": "FINISH"}

    last_user_message = messages[-1]

    response = supervisor_llm.invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + [last_user_message]
    )

    return {"next": response.next}