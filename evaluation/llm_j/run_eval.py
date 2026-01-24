import json
import pandas as pd
from langchain_core.messages import ToolMessage
from orchestrator.graph import build_graph
from evaluation.llm_j.judge import evaluate_response

FILE_PATH = "evaluation/llm_j/dataset"

def run_evaluation():
    app = build_graph()

    # 1. Cargar Dataset
    try:
        with open(f"{FILE_PATH}.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"El archivo {FILE_PATH}.json no existe.")
        return

    results = []

    print("INICIANDO EVALUACIÓN FINANCIERA (LLM-AS-A-JUDGE)...\n")

    for case in dataset:
        qid = case.get("id")
        category = case.get("category", "General")
        expected_behavior = case.get("expected_behavior", "Sin especificar")
        question = case.get("question")
        
        print(f"--- Evaluando Caso {qid}: '{question}' ---")

        try:
            initial_state = {"messages": [("user", question)]}
            
            # Variables para capturar la traza
            agent_selected = "None"
            tool_outputs = []
            final_answer = "No answer"

            # Paso a paso para capturar el Supervisor
            for event in app.stream(initial_state):
                for node_name, node_output in event.items():
                    
                    # 1. Capturar decisión del Supervisor
                    if node_name == "Supervisor":
                        agent_selected = node_output.get("next", "FINISH")
                        print(f"   ↳ Supervisor eligió: {agent_selected}")

                    # 2. Capturar outputs de los agentes
                    elif "messages" in node_output:
                        for m in node_output["messages"]:
                            # Si es un mensaje de herramienta
                            if isinstance(m, ToolMessage):
                                tool_outputs.append(f"[Tool: {m.name}] Output: {m.content}")
                            # Si es la respuesta final
                            if m.type == "ai" and not m.tool_calls:
                                final_answer = m.content

            # Preparar contexto para el juez
            context_str = "\n".join(tool_outputs) if tool_outputs else "No tools used."

            # Llamada al Juez con el dato del AGENTE SELECCIONADO
            eval_result = evaluate_response(
                question=question,
                agent_selected=agent_selected,
                context=context_str,
                answer=final_answer,
                expected_behavior=expected_behavior,
            )

            print(f"   Juez: {eval_result.score}/10 | {eval_result.error_type}")
            print(f"   Análisis: {eval_result.step_by_step_analysis[:100]}...\n")

            results.append({
                "id": qid,
                "category": category,
                "question": question,
                "agent_selected": agent_selected,
                "score": eval_result.score,
                "error_type": eval_result.error_type,
                "judge_analysis": eval_result.step_by_step_analysis,
                "agent_answer": final_answer,
                "expected": expected_behavior
            })

        except Exception as e:
            print(f"Error crítico en el caso {qid}: {e}")

    # Exportar a CSV
    df = pd.DataFrame(results)
    df.to_csv(f"{FILE_PATH}_results.csv", index=False)
    print(f"Evaluación finalizada. Reporte guardado en {FILE_PATH}_results.csv")

if __name__ == "__main__":
    run_evaluation()