import sys
import json
import time
import pandas as pd
import traceback
from langchain_core.messages import ToolMessage, AIMessage

from orchestrator.graph import build_graph
from evaluation.metrics_accumulator.logger import MetricsLogger

# Importar los 3 evaluadores
from evaluation.baseline.evaluator import evaluate_baseline
from evaluation.hybrid import HybridEvaluator
from evaluation.llm_j.judge import (
    evaluate_planner,
    evaluate_supervisor,
    evaluate_agent,
    evaluate_final_output,
    evaluate_comprehensive,
)

# Paths
DATASET_PATH = "evaluation/metrics_accumulator/dataset.json"
OUTPUT_PREFIX = "evaluation/accumulated_data/dataset_unified"


class TraceCollector:
    """Colector unificado de trazas del sistema multi-agente"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.planner_tasks = []
        self.routing_trace = []
        self.agent_executions = []
        self.sql_queries = []
        self.final_answer = ""
        self.user_question = ""

    def capture_planner(self, state: dict):
        self.planner_tasks = state.get("pending_tasks", [])

    def capture_supervisor_decision(self, state: dict):
        current_task = state.get("current_task", "")
        next_agent = state.get("next", "")
        if next_agent != "FINISH" and current_task:
            self.routing_trace.append({"task": current_task, "agent": next_agent})

    def capture_agent_execution(
        self, agent_name: str, messages: list, current_task: str = ""
    ):
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
        if "query" in tool_call_args:
            self.sql_queries.append({"task": task, "sql": tool_call_args["query"]})

    def capture_final_output(self, state: dict):
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                self.final_answer = msg.content
                break


def run_unified_batch_evaluation():
    """
    Evaluación Batch Unificada.
    Ejecuta el grafo UNA sola vez por caso y lo evalúa con:
    1. Baseline (Determinista)
    2. LLM-Judge (Cualitativo 1-4)
    3. HACE (Híbrido)
    """
    app = build_graph()
    metrics_logger = MetricsLogger()
    HACE_evaluator = HybridEvaluator()

    # Cargar Dataset
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"El archivo {DATASET_PATH} no existe.")
        return

    all_results = []

    print("=" * 80)
    print("INICIANDO EVALUACIÓN UNIFICADA (BASELINE + LLM-J + HACE)")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Total de casos: {len(dataset)}\n")

    for idx, case in enumerate(dataset, 1):
        qid = case.get("id")
        category = case.get("category")
        difficulty = case.get("difficulty", "Medium")
        question = case.get("question")
        focus_area = case.get("focus_area", "General")
        expected_behavior = case.get(
            "expected_behavior", "Procesar la solicitud fielmente."
        )

        print(f"\n{'='*80}")
        print(f"CASO {idx}/{len(dataset)}: {qid} | Dificultad: {difficulty}")
        print(f"Pregunta: {question}")
        print(f"{'='*80}")

        try:

            # FASE 1: EJECUCIÓN DEL SISTEMA

            trace = TraceCollector()
            trace.user_question = question
            initial_state = {"messages": [("user", question)]}
            final_state = None

            print("\n[1/5] Ejecutando el pipeline Multi-Agente...")
            for event in app.stream(initial_state):
                for node_name, node_output in event.items():
                    final_state = node_output

                    if node_name == "Planner":
                        trace.capture_planner(node_output)
                        print(
                            f"  ✓ Planner: {len(trace.planner_tasks)} tareas generadas"
                        )

                    elif node_name == "Supervisor":
                        trace.capture_supervisor_decision(node_output)
                        next_step = node_output.get("next", "")
                        if next_step != "FINISH":
                            print(f"  → Routing: {next_step}")
                        else:
                            print(f"  ✓ Supervisor: finalizando")

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

                        for msg in node_output.get("messages", []):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    if (
                                        tc["name"] == "crypto_history_tool"
                                        and "query" in tc["args"]
                                    ):
                                        if (
                                            isinstance(tc["args"]["query"], str)
                                            and tc["args"]["query"].strip()
                                        ):
                                            trace.capture_sql_query(
                                                current_task, tc["args"]
                                            )
                        print(f"  ✓ {node_name}: ejecutado")

            if final_state:
                trace.capture_final_output(final_state)
            else:
                raise RuntimeError("El grafo no produjo ningún estado final.")

            trace_data = {
                "user_question": trace.user_question,
                "planner_tasks": trace.planner_tasks,
                "routing_trace": trace.routing_trace,
                "agent_executions": trace.agent_executions,
                "sql_queries": trace.sql_queries,
                "final_answer": trace.final_answer,
            }

            # FASE 2: EVALUACIÓN BASELINE

            print("\n[2/5] Evaluando: Baseline Metrics...")
            baseline_eval = evaluate_baseline(trace_data)
            print(f"  ↳ Baseline Score: {baseline_eval.baseline_score:.3f}")

            # FASE 3: EVALUACIÓN LLM-JUDGE

            print("\n[3/5] Evaluando: LLM-Judge...")
            llm_start_time = time.perf_counter()

            planner_eval = evaluate_planner(
                user_message=question,
                generated_tasks=trace.planner_tasks,
                expected_behavior=expected_behavior,
            )
            planner_score = (
                planner_eval.correctness
                + planner_eval.completeness
                + planner_eval.precision
                + planner_eval.task_decomposition
            ) / 4

            supervisor_eval = evaluate_supervisor(
                pending_tasks=trace.planner_tasks,
                routing_trace=trace.routing_trace,
                expected_behavior=expected_behavior,
            )
            supervisor_score = (
                supervisor_eval.routing_accuracy + supervisor_eval.task_completion
            ) / 2

            agents_eval = []
            tools_map = {
                "Technical_Analyst": [
                    "crypto_history_tool",
                    "crypto_prediction_tool",
                    "crypto_chart_tool",
                ],
                "Fundamental_Analyst": ["crypto_rag_tool", "crypto_news_tool"],
                "Risk_Officer": ["crypto_volatility_tool"],
            }
            for execution in trace.agent_executions:
                a_name = execution["agent"]
                ag_eval = evaluate_agent(
                    agent_name=a_name,
                    current_task=execution["task"],
                    available_tools=tools_map.get(a_name, []),
                    tools_used=execution["tools_used"],
                    tool_outputs=execution["tool_outputs"],
                    agent_response=execution["agent_response"],
                    expected_behavior=expected_behavior,
                )
                agents_eval.append(ag_eval)

            agents_avg_score = (
                sum(
                    (
                        a.tool_selection
                        + a.tool_execution
                        + a.output_fidelity
                        + a.output_completeness
                        + a.hallucination_check
                    )
                    / 5
                    for a in agents_eval
                )
                / len(agents_eval)
                if agents_eval
                else 0
            )

            final_eval = evaluate_final_output(
                original_tasks=trace.planner_tasks,
                agent_outputs=[e["agent_response"] for e in trace.agent_executions],
                final_report=trace.final_answer,
                expected_behavior=expected_behavior,
            )
            final_score = (
                final_eval.completeness
                + final_eval.accuracy
                + final_eval.structure
                + final_eval.chart_attribution
            ) / 4

            comprehensive_eval = evaluate_comprehensive(
                planner_eval, supervisor_eval, agents_eval, final_eval
            )
            llm_elapsed_time = time.perf_counter() - llm_start_time

            llm_judge_data = {
                "comprehensive_eval": comprehensive_eval,
                "planner_score": planner_score,
                "supervisor_score": supervisor_score,
                "agents_avg_score": agents_avg_score,
                "final_output_score": final_score,
                "elapsed_time": llm_elapsed_time,
            }
            print(
                f"  ↳ LLM-Judge Score: {comprehensive_eval.overall_score}/4 ({comprehensive_eval.overall_score/4:.3f} norm)"
            )

            # FASE 4: EVALUACIÓN HACE (HYBRID)

            print("\n[4/5] Evaluando: HACE (Híbrido)...")
            hybrid_eval = HACE_evaluator.evaluate(trace_data)
            print(
                f"  ↳ HACE Score: {hybrid_eval['final_score']:.3f} | Capas: {'3' if hybrid_eval['layer3_used'] else '2'}"
            )

            # FASE 5: REGISTRO ACUMULADO

            print("\n[5/5] Guardando en Accumulator...")
            metrics_logger.log_offline_evaluation(
                test_case={
                    "id": qid,
                    "query": question,
                    "difficulty": difficulty,
                    "category": category,
                    "expected_tasks": case.get("expected_tasks", []),
                },
                trace_data=trace_data,
                baseline_eval=baseline_eval,
                llm_judge_data=llm_judge_data,
                hybrid_eval=hybrid_eval,
            )

            # Consolidar todo para el CSV local
            result = {
                # Metadata
                "id": qid,
                "category": category,
                "difficulty": difficulty,
                "focus_area": focus_area,
                "question": question,
                # Ejecución
                "planner_tasks_count": len(trace.planner_tasks),
                "routing_decisions": len(trace.routing_trace),
                "agents_invoked": "; ".join(
                    [e["agent"] for e in trace.agent_executions]
                ),
                # BASELINE
                "baseline_score": baseline_eval.baseline_score,
                "baseline_time": baseline_eval.evaluation_time,
                "routing_f1": baseline_eval.routing_metrics.f1,
                "numeric_f1": baseline_eval.numeric_metrics.f1,
                "hallucination_rate": baseline_eval.numeric_metrics.hallucination_rate,
                "task_coverage": baseline_eval.task_coverage_metrics.coverage,
                # LLM-JUDGE
                "llmj_overall": comprehensive_eval.overall_score,
                "llmj_norm": comprehensive_eval.overall_score / 4,
                "llmj_time": llm_elapsed_time,
                "llmj_error_cat": comprehensive_eval.error_category,
                "llmj_planner": planner_score,
                "llmj_supervisor": supervisor_score,
                "llmj_agents": agents_avg_score,
                # HACE
                "HACE_score": float(hybrid_eval["final_score"]),
                "HACE_time": float(hybrid_eval["evaluation_time"]),
                "HACE_layer1": float(hybrid_eval["layer1_score"]),
                "HACE_layer2": float(hybrid_eval["layer2_score"]),
                "HACE_layer3": (
                    float(hybrid_eval["layer3_score"])
                    if hybrid_eval["layer3_score"]
                    else None
                ),
                "HACE_layer3_used": int(hybrid_eval["layer3_used"]),
                "HACE_quality": str(hybrid_eval["quality_label"]),
            }
            all_results.append(result)
            print(f"  ✓ Caso {qid} completado con éxito.")

        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO en {qid}: {str(e)}\n")
            traceback.print_exc()
            all_results.append(
                {
                    "id": qid,
                    "category": category,
                    "difficulty": difficulty,
                    "question": question,
                    "error": str(e),
                }
            )

    # EXPORTACIÓN DE RESULTADOS
    print(f"\n{'='*80}")
    print("Exportando CSV Combinado...")
    print(f"{'='*80}\n")

    df = pd.DataFrame(all_results)
    full_path = f"{OUTPUT_PREFIX}_results.csv"
    df.to_csv(full_path, index=False)
    print(f"Resultados completos guardados en: {full_path}")

    # Mostrar info del accumulator real
    print(f"\nMétricas oficiales acumuladas guardadas en:")
    print(f"   evaluation/accumulated_data/offline_metrics.csv")

    # Resumen global en consola
    print(f"\n{'='*80}")
    print("RESUMEN GLOBAL UNIFICADO")
    print(f"{'='*80}\n")
    print(f"Total de casos evaluados: {len(df)}")
    if "baseline_score" in df.columns:
        print(f"Score Promedio Baseline:  {df['baseline_score'].mean():.3f}")
        print(f"Score Promedio LLM-Judge: {df['llmj_norm'].mean():.3f} (Normalizado)")
        print(f"Score Promedio HACE:      {df['HACE_score'].mean():.3f}")
        print("\nTiempos Promedio:")
        print(f"Baseline:  {df['baseline_time'].mean():.3f}s")
        print(f"LLM-Judge: {df['llmj_time'].mean():.3f}s")
        print(f"HACE:      {df['HACE_time'].mean():.3f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    run_unified_batch_evaluation()
