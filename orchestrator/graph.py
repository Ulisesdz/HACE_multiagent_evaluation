from langgraph.graph import StateGraph, END, START
from orchestrator.state import AgentState
from orchestrator.agents import (
    supervisor_node, 
    technical_node, 
    fundamental_node, 
    risk_node
)

def build_graph():
    workflow = StateGraph(AgentState)

    # --- 1. DEFINICIÓN DE NODOS ---
    # El Jefe
    workflow.add_node("Supervisor", supervisor_node)
    
    # Los Empleados (Specialists)
    workflow.add_node("Technical_Analyst", technical_node)
    workflow.add_node("Fundamental_Analyst", fundamental_node)
    workflow.add_node("Risk_Officer", risk_node)

    # --- 2. PUNTO DE ENTRADA ---
    workflow.add_edge(START, "Supervisor")

    # --- 3. LÓGICA DE ENRUTAMIENTO (ROUTING) ---
    # El Supervisor decide a quién llamar basándose en el output "next"
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

    # --- 4. CIERRE DEL FLUJO ---
    # En esta versión V1, los agentes responden y terminan el turno.
    # (En una V2, podrían devolver el trabajo al Supervisor para que redacte un informe final)
    workflow.add_edge("Technical_Analyst", END)
    workflow.add_edge("Fundamental_Analyst", END)
    workflow.add_edge("Risk_Officer", END)

    return workflow.compile()