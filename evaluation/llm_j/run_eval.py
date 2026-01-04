import json
import pandas as pd
from langchain_core.messages import ToolMessage, AIMessage
from orchestrator.graph import build_graph
from evaluation.judge import evaluate_response

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

    print("INICIANDO EVALUACIÓN LLM-AS-A-JUDGE...\n")

    for case in dataset:
        qid = case.get("id")
        category = case.get("category", "General")
        difficulty = case.get("difficulty", "Unknown")
        expected_behavior = case.get("expected_behavior", "Sin especificar")
        question = case.get("question")
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
            eval_result = evaluate_response(
                question=question,
                context=context,
                answer=final_answer,
                expected_behavior=expected_behavior,
            )

            # Guardar resultados
            print(f" Análisis: {eval_result.step_by_step_analysis[:150]}...")
            print(f" Juez: {eval_result.score}/10 [{eval_result.error_type}]\n")

            results.append(
                {
                    "id": qid,
                    "category": category,
                    "difficulty": difficulty,
                    "question": question,
                    "score": eval_result.score,
                    "error_type": eval_result.error_type,
                    "agent_answer": final_answer,
                    "judge_analysis": eval_result.step_by_step_analysis,
                    "expected_behavior": expected_behavior,
                    "real_context": context[:1000],
                }
            )

        except Exception as e:
            print(f"Error crítico en el caso {qid}: {e}")

    # Exportar a CSV
    df = pd.DataFrame(results)
    cols_order = [
        "id",
        "category",
        "difficulty",
        "score",
        "error_type",
        "question",
        "agent_answer",
        "judge_analysis",
    ]
    existing_cols = [c for c in cols_order if c in df.columns] + [
        c for c in df.columns if c not in cols_order
    ]
    df = df[existing_cols]
    df.to_csv(f"{FILE_PATH}.csv", index=False)
    print(f"Evaluación finalizada. Reporte guardado en {FILE_PATH}.csv")


if __name__ == "__main__":
    run_evaluation()
