from langgraph.graph import StateGraph, END, START
from orchestrator.state import AgentState
from orchestrator.agents import supervisor_node, crypto_node, weather_node


def build_graph():
    workflow = StateGraph(AgentState)

    # Nodos
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Crypto_Agent", crypto_node)
    workflow.add_node("Weather_Agent", weather_node)

    # Edges
    workflow.add_edge(START, "Supervisor")

    # Lógica Condicional del Supervisor
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {
            "Crypto_Agent": "Crypto_Agent",
            "Weather_Agent": "Weather_Agent",
            "FINISH": END,
        },
    )

    # Una vez que el subagente termina su trabajo, vuelve al supervisor
    workflow.add_edge("Crypto_Agent", END)
    workflow.add_edge("Weather_Agent", END)

    return workflow.compile()
