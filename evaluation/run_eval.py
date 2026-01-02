import json
import pandas as pd
from langchain_core.messages import ToolMessage, AIMessage
from orchestrator.graph import build_graph
from evaluation.judge import evaluate_response

def run_evaluation():
    
    app = build_graph()

    # 1. Cargar Dataset
    with open("evaluation/dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    results = []

    print("INICIANDO EVALUACIÓN LLM-AS-A-JUDGE...\n")

    for case in dataset:
        qid = case["id"]
        question = case["question"]
        print(f"--- Evaluando Caso {qid}: '{question}' ---")

        try:
            initial_state = {"messages": [("user", question)]}
            output_state = app.invoke(initial_state)
            messages = output_state["messages"]
            
            # Extraer (Parsing) la respuesta y el contexto
            final_answer = messages[-1].content
            
            # Mensajes de Herramienta (ToolMessage) para ver el contexto
            tool_outputs = [m.content for m in messages if isinstance(m, ToolMessage)]
            
            if not tool_outputs:
                context = "No se usó ninguna herramienta (Respuesta directa)."
            else:
                context = "\n".join(tool_outputs)

            # Juez
            eval_result = evaluate_response(question, context, final_answer)
            
            # Guardar resultados
            print(f" Veredicto: {eval_result.score}/10 | {eval_result.hallucination_type}")
            print(f" Razón: {eval_result.reasoning}\n")
            
            results.append({
                "id": qid,
                "question": question,
                "agent_answer": final_answer,
                "tool_context": context[:200] + "...",
                "score": eval_result.score,
                "error_type": eval_result.hallucination_type,
                "reasoning": eval_result.reasoning
            })

        except Exception as e:
            print(f"Error crítico en el caso {qid}: {e}")

    # Exportar a CSV
    df = pd.DataFrame(results)
    df.to_csv("evaluation/results_report.csv", index=False)
    print("Evaluación finalizada. Reporte guardado en 'evaluation/results_report.csv'")

if __name__ == "__main__":
    run_evaluation()