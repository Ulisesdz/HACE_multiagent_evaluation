from typing import Literal, List
from pydantic import BaseModel, Field
from typing import Literal

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from orchestrator.config import get_llm, CRYPTO_DB
from orchestrator.utils import (
    get_available_entities,
    log_execution,
    extract_reports,
    extract_asset_from_task,
    get_current_turn_messages,
    get_last_user_message
)

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
    PLANNER_SYSTEM_PROMPT,
    SUPERVISOR_ROUTER_PROMPT,
    SUPERVISOR_SUMMARY_PROMPT,
    get_technical_agent_prompt,
    get_fundamental_agent_prompt,
    get_risk_agent_prompt
)


llm = get_llm()
available_coins = get_available_entities(CRYPTO_DB)


# AGENTES REACT - NODOS DEL GRAFO
# --- ANALISTA TÉCNICO ---
technical_prompt_text = get_technical_agent_prompt(available_coins)
_technical_agent = create_react_agent(
    llm,
    tools=[crypto_history_tool, crypto_prediction_tool, crypto_chart_tool],
    prompt=technical_prompt_text, 
)

@log_execution
def technical_node(state):
    """
    Ejecuta el agente de análisis técnico.
    
    Lee la tarea asignada desde state["current_task"] y la procesa de forma
    aislada, sin considerar el historial completo de la conversación.
    
    Args:
        state: Estado del grafo con mensajes y tareas.
        
    Returns:
        Diccionario con los mensajes generados, incluyendo el reporte firmado.
    """
    # Leer la tarea específica asignada por el supervisor
    clean_instruction = state.get("current_task")
    
    if not clean_instruction:
        # Fallback: usar el último mensaje del usuario si no hay tarea específica
        user_msg = get_last_user_message(state["messages"])
    else:
        # Crear mensaje limpio con solo la instrucción específica
        user_msg = HumanMessage(content=clean_instruction, name="supervisor_instruction")

    # Invocar el agente React con la tarea aislada
    result = _technical_agent.invoke({"messages": [user_msg]})
    
    # Firmar el reporte con el identificador del agente
    generated_messages = result["messages"]
    last_resp = generated_messages[-1].content
    
    # Extraer el activo de la tarea para incluirlo en el reporte
    asset_name = extract_asset_from_task(clean_instruction) if clean_instruction else "Activo no especificado"
    
    signed_response = f"### REPORTE DEL TECHNICAL_ANALYST ###\n**Activo:** {asset_name}\n\n{last_resp}"
    generated_messages[-1] = AIMessage(content=signed_response)

    return {"messages": generated_messages}


# --- ANALISTA FUNDAMENTAL ---
fundamental_prompt_text = get_fundamental_agent_prompt()
_fundamental_agent = create_react_agent(
    llm,
    tools=[crypto_rag_tool, crypto_news_tool],
    prompt=fundamental_prompt_text,
)

@log_execution
def fundamental_node(state):
    """
    Ejecuta el agente de análisis fundamental.
    
    Procesa tareas relacionadas con noticias y contexto del mercado de forma
    aislada, sin interferencia del historial conversacional.
    
    Args:
        state: Estado del grafo con mensajes y tareas.
        
    Returns:
        Diccionario con los mensajes generados, incluyendo el reporte firmado.
    """
    clean_instruction = state.get("current_task")
    
    if not clean_instruction:
        user_msg = get_last_user_message(state["messages"])
    else:
        user_msg = HumanMessage(content=clean_instruction, name="supervisor_instruction")

    result = _fundamental_agent.invoke({"messages": [user_msg]})
    
    # Firmar el reporte
    generated_messages = result["messages"]
    last_resp = generated_messages[-1].content
    
    # Extraer el activo de la tarea
    asset_name = extract_asset_from_task(clean_instruction) if clean_instruction else "Activo no especificado"
    
    signed_response = f"### REPORTE DEL FUNDAMENTAL_ANALYST ###\n**Activo:** {asset_name}\n\n{last_resp}"
    generated_messages[-1] = AIMessage(content=signed_response)

    return {"messages": generated_messages}


# --- OFICIAL DE RIESGOS ---
risk_prompt_text = get_risk_agent_prompt()
_risk_agent = create_react_agent(
    llm,
    tools=[crypto_volatility_tool],
    prompt=risk_prompt_text,
)

@log_execution
def risk_node(state):
    """
    Ejecuta el agente de análisis de riesgos.
    
    Procesa tareas relacionadas con volatilidad y métricas de riesgo de forma
    aislada del historial completo.
    
    Args:
        state: Estado del grafo con mensajes y tareas.
        
    Returns:
        Diccionario con los mensajes generados, incluyendo el reporte firmado.
    """
    clean_instruction = state.get("current_task")
    
    if not clean_instruction:
        user_msg = get_last_user_message(state["messages"])
    else:
        user_msg = HumanMessage(content=clean_instruction, name="supervisor_instruction")

    result = _risk_agent.invoke({"messages": [user_msg]})
    
    # Firmar el reporte
    generated_messages = result["messages"]
    last_resp = generated_messages[-1].content
    
    # Extraer el activo de la tarea
    asset_name = extract_asset_from_task(clean_instruction) if clean_instruction else "Activo no especificado"
    
    signed_response = f"### REPORTE DEL RISK_OFFICER ###\n**Activo:** {asset_name}\n\n{last_resp}"
    generated_messages[-1] = AIMessage(content=signed_response)

    return {"messages": generated_messages}


# PLANIFICADOR - GENERACIÓN DE LISTA DE TAREAS
class PlanningOutput(BaseModel):
    """Esquema de salida estructurada para el planificador."""
    tasks: List[str] = Field(description="Lista de tareas.")


planner_llm = llm.with_structured_output(PlanningOutput)


@log_execution
def planner_node(state):
    """
    Genera la lista de tareas a partir del mensaje más reciente del usuario.
    
    Analiza únicamente el último mensaje humano (ignorando instrucciones internas
    del supervisor) y descompone la solicitud en tareas atómicas.
    
    Args:
        state: Estado del grafo con el historial de mensajes.
        
    Returns:
        Diccionario con la lista de tareas pendientes y campos reseteados.
    """
    messages = state["messages"]
    user_msg = None
    
    # Buscar el último mensaje humano real (no supervisor_instruction)
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.name != "supervisor_instruction":
            user_msg = m
            break
    
    # Fallback: Devuelver lista vacía si no hay mensaje de usuario
    if not user_msg:
        return {
            "pending_tasks": [],
            "completed_outputs": []
        }
    
    # Generar plan de tareas usando salida estructurada
    plan = planner_llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT), 
        user_msg
    ])

    print(f"[PLANNER DEBUG] Plan generado: {plan}\n")
    
    return {
        "pending_tasks": plan.tasks,
        "completed_outputs": [],
        "current_task": ""  # Resetear tarea actual
    }


# SUPERVISOR - COORDINACIÓN Y ENRUTAMIENTO
class RouterOutput(BaseModel):
    """Esquema de salida estructurada para el router del supervisor."""
    agent: Literal["Technical_Analyst", "Fundamental_Analyst", "Risk_Officer"]


router_llm = llm.with_structured_output(RouterOutput)


@log_execution
def supervisor_node(state):
    """
    Coordina la ejecución del flujo de trabajo mediante un sistema de cola.
    
    Comportamiento basado en el estado de las tareas pendientes:
    - Si la lista está vacía: consolida reportes y genera resumen final (FINISH).
    - Si hay tareas pendientes: asigna la siguiente tarea al agente apropiado.
    
    Args:
        state: Estado del grafo con tareas pendientes y mensajes.
        
    Returns:
        Diccionario con el siguiente nodo a ejecutar y actualizaciones de estado.
    """
    pending = state.get("pending_tasks", [])
    
    # CASO 1: Sin tareas pendientes -> Finalizar y generar resumen
    if not pending:
        # Extraer y consolidar reportes del turno actual
        reports = extract_reports(get_current_turn_messages(state["messages"]))

        # Pregunta del usuario para dar contexto al resumen de lo que se ha generado
        last_user_msg = get_last_user_message(state["messages"])
        
        # Generar resumen final ejecutivo
        summary = llm.invoke(
                [SystemMessage(content=SUPERVISOR_SUMMARY_PROMPT)] + 
                [last_user_msg] + 
                reports
            )
        
        return {"next": "FINISH", "messages": [summary]}

    # CASO 2: Hay tareas pendientes -> Despachar siguiente tarea
    next_task = pending[0]
    remaining_tasks = pending[1:]
    
    # Determinar qué agente debe ejecutar la tarea
    decision = router_llm.invoke([
        SystemMessage(content=SUPERVISOR_ROUTER_PROMPT),
        HumanMessage(content=f"Tarea a asignar: {next_task}")
    ])
    
    return {
        "next": decision.agent,
        "current_task": next_task,        # Tarea específica para el agente
        "pending_tasks": remaining_tasks  # Avanzar la cola
    }