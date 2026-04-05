"""
Script de evaluación offline para MACE
Ejecuta evaluación híbrida sobre dataset completo
"""

import sys
import json
import pandas as pd
from pathlib import Path
from langchain_core.messages import ToolMessage, AIMessage
from orchestrator.graph import build_graph
from evaluation.hybrid import HybridEvaluator
from evaluation.metrics_accumulator.logger import MetricsLogger

# Paths
DATASET_PATH = "evaluation/metrics_accumulator/dataset.json"
OUTPUT_PREFIX = "evaluation/hybrid/dataset"


class TraceCollector:
    """Colector de trazas del sistema (mismo que en otros evaluadores)"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reiniciar estado del colector"""
        self.planner_tasks = []
        self.routing_trace = []
        self.agent_executions = []
        self.sql_queries = []
        self.final_answer = ""
        self.user_question = ""

    def capture_planner(self, state: dict):
        """
        Capturar tareas generadas por el Planner

        Args:
            state: Dict del state de LangGraph con 'pending_tasks'
        """
        self.planner_tasks = state.get("pending_tasks", [])

    def capture_supervisor_decision(self, state: dict):
        """
        Capturar decisión de routing del Supervisor

        Args:
            state: Dict con 'current_task' y 'next' (agente)
        """
        current_task = state.get("current_task", "")
        next_agent = state.get("next", "")

        if next_agent != "FINISH" and current_task:
            self.routing_trace.append({"task": current_task, "agent": next_agent})

    def capture_agent_execution(
        self, agent_name: str, messages: list, current_task: str = ""
    ):
        """
        Capturar ejecución de un agente especialista

        Args:
            agent_name: Nombre del agente
            messages: Lista de mensajes de LangChain
            current_task: Tarea actual asignada
        """
        tools_used = []
        tool_outputs_text = []
        agent_response = ""

        for msg in messages:
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tools_used.append(tool_call.get("name", "unknown"))

                if msg.content and not msg.tool_calls:
                    agent_response = msg.content

            elif isinstance(msg, ToolMessage):
                tool_outputs_text.append(f"[{msg.name}]: {msg.content}")

        if tools_used or agent_response:
            self.agent_executions.append(
                {
                    "agent": agent_name,
                    "task": current_task,
                    "tools_used": tools_used,
                    "tool_outputs": "\n".join(tool_outputs_text),
                    "agent_response": agent_response,
                }
            )

    def capture_sql_query(self, task: str, tool_call_args: dict):
        """
        Capturar queries SQL ejecutadas

        Args:
            task: Tarea asociada
            tool_call_args: Argumentos del tool call
        """
        if "query" in tool_call_args:
            self.sql_queries.append({"task": task, "sql": tool_call_args["query"]})

    def capture_final_output(self, state: dict):
        """
        Capturar mensaje final del sistema

        Args:
            state: Dict con 'messages' finales
        """
        messages = state.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                self.final_answer = msg.content
                break


def run_hybrid_evaluation():
    """
    Ejecutar evaluación híbrida MACE sobre dataset completo

    Flujo:
    1. Cargar dataset de evaluación
    2. Por cada caso:
       - Ejecutar sistema multi-agente
       - Capturar trazas
       - Evaluar con HybridEvaluator
    3. Exportar resultados a CSV
    4. Mostrar estadísticas agregadas

    Output:
        CSV en evaluation/hybrid/dataset_hybrid_results.csv
    """
    app = build_graph()
    evaluator = HybridEvaluator()
    metrics_logger = MetricsLogger()

    # Cargar dataset
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"El archivo {DATASET_PATH} no existe.")
        return

    all_results = []

    print("=" * 80)
    print("MACE - Evaluación Híbrida (BATCH MODE)")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total de casos: {len(dataset)}")
    print()

    for idx, case in enumerate(dataset, 1):
        qid = case.get("id")
        category = case.get("category")
        difficulty = case.get("difficulty", "Medium")
        question = case.get("question")
        focus_area = case.get("focus_area", "General")

        print(f"\n{'='*80}")
        print(
            f"CASO {idx}/{len(dataset)}: {qid} | {category} | Dificultad: {difficulty}"
        )
        print(f"   Pregunta: {question}")
        print(f"{'='*80}")

        try:
            # Ejecutar sistema
            trace = TraceCollector()
            trace.user_question = question

            initial_state = {"messages": [("user", question)]}
            final_state = None

            print("Ejecutando sistema...")

            for event in app.stream(initial_state):
                for node_name, node_output in event.items():
                    final_state = node_output

                    if node_name == "Planner":
                        trace.capture_planner(node_output)
                        print(f"   ✓ Planner: {len(trace.planner_tasks)} tareas")

                    elif node_name == "Supervisor":
                        trace.capture_supervisor_decision(node_output)
                        next_step = node_output.get("next", "")
                        if next_step != "FINISH":
                            print(f"   → Routing: {next_step}")

                    elif node_name in [
                        "Technical_Analyst",
                        "Fundamental_Analyst",
                        "Risk_Officer",
                    ]:
                        current_task = node_output.get("current_task", "")

                        if not current_task and trace.routing_trace:
                            current_task = trace.routing_trace[-1].get("task", "")

                        trace.capture_agent_execution(
                            node_name, node_output.get("messages", []), current_task
                        )

                        # Capturar SQL si aplica
                        for msg in node_output.get("messages", []):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    if tool_call["name"] == "crypto_history_tool":
                                        if "query" in tool_call["args"]:
                                            query_value = tool_call["args"]["query"]
                                            if (
                                                isinstance(query_value, str)
                                                and query_value.strip()
                                            ):
                                                trace.capture_sql_query(
                                                    current_task, tool_call["args"]
                                                )

                        print(f"   ✓ {node_name}: ejecutado")

            if final_state:
                trace.capture_final_output(final_state)
            else:
                raise RuntimeError("El grafo no produjo ningún estado final.")

            print(f"Ejecución completada")

            # Preparar trace data
            trace_data = {
                "user_question": trace.user_question,
                "planner_tasks": trace.planner_tasks,
                "routing_trace": trace.routing_trace,
                "agent_executions": trace.agent_executions,
                "sql_queries": trace.sql_queries,
                "final_answer": trace.final_answer,
            }

            # Evaluar con MACE
            print("Evaluando con MACE...")
            evaluation = evaluator.evaluate(trace_data)

            # Resumen en consola
            print(f"\n{'─'*60}")
            print(f" RESULTADO:")
            print(f"   MACE Score:      {evaluation['final_score']:.3f}")
            print(f"   Quality Label:   {evaluation['quality_label']}")
            print(f"   Confidence:      {evaluation['confidence']}")
            print(f"   • Layer 1 Score: {evaluation['layer1_score']:.3f}")
            print(f"   • Layer 2 Score: {evaluation['layer2_score']:.3f}")
            if evaluation["layer3_score"] is not None:
                print(f"   • Layer 3 Score: {evaluation['layer3_score']:.3f}")
            print(f"   • Layer 3 Used:  {'Yes' if evaluation['layer3_used'] else 'No'}")
            print(f"   • Eval Time:     {evaluation['evaluation_time']:.3f}s")
            print(f"{'─'*60}\n")

            # GUARDAR EN METRICS ACCUMULATOR
            metrics_logger.log_offline_evaluation(
                test_case={
                    "id": qid,
                    "query": question,
                    "difficulty": difficulty,
                    "category": category,
                    "expected_tasks": case.get("expected_tasks", []),
                },
                trace_data=trace_data,
                baseline_eval=None,
                llm_judge_data=None,
                hybrid_eval=evaluation,
            )

            # Guardar resultados para CSV
            result = {
                # Metadata
                "id": qid,
                "category": category,
                "difficulty": difficulty,
                "focus_area": focus_area,
                "question": question,
                # Trazas de Ejecución
                "planner_tasks_count": len(trace.planner_tasks),
                "planner_tasks": "; ".join(trace.planner_tasks),
                "routing_decisions": len(trace.routing_trace),
                "agents_invoked": "; ".join(
                    [e["agent"] for e in trace.agent_executions]
                ),
                "sql_queries_count": len(trace.sql_queries),
                # Scores MACE (TODOS NUMÉRICOS)
                "hybrid_score": float(evaluation["final_score"]),
                "layer1_score": float(evaluation["layer1_score"]),
                "layer2_score": float(evaluation["layer2_score"]),
                "layer3_score": (
                    float(evaluation["layer3_score"])
                    if evaluation["layer3_score"] is not None
                    else None
                ),
                # Metadata MACE
                "quality_label": str(evaluation["quality_label"]),
                "confidence": str(evaluation["confidence"]),
                "layer3_used": int(evaluation["layer3_used"]),  # 0 o 1
                "escalation_reason": str(evaluation.get("escalation_reason", "")),
                "critical_failures": "; ".join(evaluation.get("critical_failures", [])),
                # Tiempos
                "evaluation_time": float(evaluation["evaluation_time"]),
                "layer1_time": float(evaluation["layer1_time"]),
                "layer2_time": float(evaluation["layer2_time"]),
                "layer3_time": float(evaluation["layer3_time"]),
                # Output
                "final_answer": trace.final_answer[:500],
            }

            all_results.append(result)

        except Exception as e:
            print(f"\nERROR CRÍTICO en {qid}: {str(e)}\n")
            import traceback

            traceback.print_exc()

            all_results.append(
                {
                    "id": qid,
                    "category": category,
                    "difficulty": difficulty,
                    "question": question,
                    "hybrid_score": 0.0,
                    "quality_label": "Error",
                    "confidence": "N/A",
                    "layer3_used": 0,
                    "evaluation_time": 0.0,
                    "error": str(e),
                }
            )

    # EXPORTAR RESULTADOS
    print(f"\n{'='*80}")
    print("Exportando resultados...")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)

    # Guardar CSV completo
    full_path = f"{OUTPUT_PREFIX}_hybrid_results.csv"
    df.to_csv(full_path, index=False)
    print(f"Resultados completos: {full_path}")

    # Guardar resumen ejecutivo
    summary_cols = [
        "id",
        "category",
        "difficulty",
        "focus_area",
        "hybrid_score",
        "quality_label",
        "confidence",
        "layer1_score",
        "layer2_score",
        "layer3_score",
        "layer3_used",
        "evaluation_time",
    ]
    summary = df[[col for col in summary_cols if col in df.columns]].copy()
    summary_path = f"{OUTPUT_PREFIX}_hybrid_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Resumen ejecutivo: {summary_path}")

    # Mostrar info del accumulator
    print(f"\nMétricas acumuladas guardadas en:")
    print(f"   evaluation/accumulated_data/offline_metrics.csv")

    # ESTADÍSTICAS GLOBALES
    print(f"\n{'='*80}")
    print("ESTADÍSTICAS GLOBALES (MACE)")
    print(f"{'='*80}\n")
    print(f"Total de casos evaluados: {len(df)}")
    print(f"MACE Score promedio:      {df['hybrid_score'].mean():.3f}")
    print(f"Tiempo promedio:          {df['evaluation_time'].mean():.3f}s")

    print(f"\nScores por Capa:")
    print(f"  • Layer 1 (Guardrails): {df['layer1_score'].mean():.3f}")
    print(f"  • Layer 2 (Semantic):   {df['layer2_score'].mean():.3f}")
    if df["layer3_score"].notna().any():
        print(
            f"  • Layer 3 (LLM):        {df['layer3_score'].mean():.3f} (cuando se usa)"
        )

    print(f"\nUso de Layer 3:")
    layer3_count = df["layer3_used"].sum()
    print(f"  • Casos con Layer 3:  {layer3_count} ({layer3_count/len(df)*100:.1f}%)")
    print(
        f"  • Casos sin Layer 3:  {len(df)-layer3_count} ({(len(df)-layer3_count)/len(df)*100:.1f}%)"
    )

    print(f"\nDistribución de Calidad:")
    quality_dist = df["quality_label"].value_counts()
    for quality, count in quality_dist.items():
        print(f"  • {quality}: {count} ({count/len(df)*100:.1f}%)")

    print(f"\nPor Dificultad:")
    diff_stats = (
        df.groupby("difficulty")
        .agg({"hybrid_score": ["mean", "std", "count"], "layer3_used": "sum"})
        .round(3)
    )
    print(diff_stats)

    print(f"\nPor Categoría (Top 5):")
    cat_stats = (
        df.groupby("category")["hybrid_score"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(5)
    )
    print(cat_stats)

    print(f"\nTiempos por Capa:")
    print(f"  • Layer 1: {df['layer1_time'].mean():.3f}s (avg)")
    print(f"  • Layer 2: {df['layer2_time'].mean():.3f}s (avg)")
    if layer3_count > 0:
        layer3_cases = df[df["layer3_used"] == 1]
        print(
            f"  • Layer 3: {layer3_cases['layer3_time'].mean():.3f}s (avg cuando se usa)"
        )

    print(f"\n{'='*80}")
    print("EVALUACIÓN MACE COMPLETADA")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    run_hybrid_evaluation()
