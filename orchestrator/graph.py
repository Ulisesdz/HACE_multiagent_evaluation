from langgraph.graph import StateGraph, END, START
from orchestrator.state import AgentState
from orchestrator.agents import (
    supervisor_node,
    technical_node,
    fundamental_node,
    risk_node,
    planner_node,
)


def build_graph():
    workflow = StateGraph(AgentState)

    # --- 1. NODOS ---
    workflow.add_node("Planner", planner_node)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Technical_Analyst", technical_node)
    workflow.add_node("Fundamental_Analyst", fundamental_node)
    workflow.add_node("Risk_Officer", risk_node)

    # --- 2. INICIO ---
    workflow.add_edge(START, "Planner")
    workflow.add_edge("Planner", "Supervisor")

    # --- 3. LOGICA CONDICIONAL (El Supervisor decide) ---
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {
            "Technical_Analyst": "Technical_Analyst",
            "Fundamental_Analyst": "Fundamental_Analyst",
            "Risk_Officer": "Risk_Officer",
            "FINISH": END,
        },
    )

    # --- 4. CICLO DE RETORNO (Iterativo) ---
    # En lugar de ir a END, los agentes vuelven al Supervisor
    # para que este decida si falta algo más o si ya termina.
    workflow.add_edge("Technical_Analyst", "Supervisor")
    workflow.add_edge("Fundamental_Analyst", "Supervisor")
    workflow.add_edge("Risk_Officer", "Supervisor")

    return workflow.compile()
