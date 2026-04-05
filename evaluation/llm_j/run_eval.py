import sys
import json
import time
import pandas as pd
from langchain_core.messages import ToolMessage, AIMessage
from orchestrator.graph import build_graph
from evaluation.metrics_accumulator.logger import MetricsLogger
from evaluation.llm_j.judge import (
    evaluate_planner,
    evaluate_supervisor,
    evaluate_agent,
    evaluate_final_output,
    evaluate_comprehensive
)

DATASET_PATH = "evaluation/metrics_accumulator/dataset.json"
FILE_PATH = "evaluation/llm_j/dataset"

class TraceCollector:
    """Colector de trazas del sistema multi-agente"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.planner_tasks = []
        self.routing_trace = []
        self.agent_executions = []
        self.tool_outputs = {}
        self.final_answer = ""
        self.completed_tasks = []
        self.pending_tasks = []
    
    def capture_planner(self, state: dict):
        """Captura las tareas generadas por el Planner"""
        self.pending_tasks = state.get("pending_tasks", [])
        self.planner_tasks = list(self.pending_tasks)
    
    def capture_supervisor_decision(self, state: dict):
        """Captura cada decisión de routing del Supervisor"""
        current_task = state.get("current_task", "")
        next_agent = state.get("next", "")
        
        if next_agent != "FINISH" and current_task:
            self.routing_trace.append({
                "task": current_task,
                "agent_selected": next_agent,
                "pending_tasks_count": len(state.get("pending_tasks", []))
            })
    
    def capture_agent_execution(self, agent_name: str, state: dict):
        """Captura la ejecución de un agente"""
        current_task = state.get("current_task", "")
        messages = state.get("messages", [])
        
        # Extraer herramientas usadas y sus outputs
        tools_used = []
        tool_outputs_text = []
        agent_response = ""
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Capturar tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tools_used.append(tool_call.get('name', 'unknown'))
                
                # Capturar respuesta final del agente
                if msg.content and not msg.tool_calls:
                    agent_response = msg.content
            
            elif isinstance(msg, ToolMessage):
                tool_outputs_text.append(f"[{msg.name}]: {msg.content}")
        
        execution_record = {
            "agent": agent_name,
            "task": current_task,
            "tools_used": tools_used,
            "tool_outputs": "\n".join(tool_outputs_text),
            "agent_response": agent_response
        }
        
        self.agent_executions.append(execution_record)
        self.tool_outputs[agent_name] = tool_outputs_text
    
    def capture_final_output(self, state: dict):
        """Captura el mensaje final del sistema"""
        messages = state.get("messages", [])
        completed = state.get("completed_outputs", [])
        
        # Buscar el último mensaje del Supervisor
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                self.final_answer = msg.content
                break
        
        self.completed_tasks = completed


def run_comprehensive_evaluation():
    """
    Sistema de evaluación comprehensivo (escala 1-4)
    """
    app = build_graph()
    
    # Inicializar MetricsLogger
    metrics_logger = MetricsLogger()
    
    # Cargar Dataset
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"El archivo {DATASET_PATH} no existe.")
        return
    
    all_results = []
    
    print("=" * 80)
    print("INICIANDO EVALUACIÓN COMPREHENSIVA DEL SISTEMA MULTI-AGENTE")
    print("=" * 80)
    print()
    
    for idx, case in enumerate(dataset, 1):
        qid = case.get("id")
        category = case.get("category")
        difficulty = case.get("difficulty", "Medium")
        question = case.get("question")
        expected_behavior = case.get("expected_behavior", "")
        focus_area = case.get("focus_area", "General")
        
        print(f"\n{'='*80}")
        print(f"CASO {idx}/{len(dataset)}: {qid} | {category} | Dificultad: {difficulty}")
        print(f"   Pregunta: {question}")
        print(f"   Área de Enfoque: {focus_area}")
        print(f"{'='*80}\n")
        
        try:
            trace = TraceCollector()
            initial_state = {"messages": [("user", question)]}
            final_state = None
            
            # PASO 1: EJECUCIÓN Y CAPTURA DE TRAZAS
            print("Ejecutando el sistema...")
            
            for event in app.stream(initial_state):
                for node_name, node_output in event.items():
                    final_state = node_output
                    
                    # Capturar Planner
                    if node_name == "Planner":
                        trace.capture_planner(node_output)
                        print(f"  ✓ Planner generó {len(trace.planner_tasks)} tareas")
                    
                    # Capturar Supervisor
                    elif node_name == "Supervisor":
                        trace.capture_supervisor_decision(node_output)
                        next_step = node_output.get("next", "")
                        if next_step != "FINISH":
                            print(f"  ↳ Supervisor enrutó a: {next_step}")
                        else:
                            print(f"  ✓ Supervisor finalizó el proceso")
                    
                    # Capturar Agentes
                    elif node_name in ["Technical_Analyst", "Fundamental_Analyst", "Risk_Officer"]:
                        trace.capture_agent_execution(node_name, node_output)
                        print(f"  - {node_name} ejecutó tarea")

            
            # Capturar output final
            if final_state is not None:
                trace.capture_final_output(final_state)
            else:
                raise RuntimeError("El grafo no produjo ningún estado final.")
            
            print(f"\nEjecución completada")
            print(f"  • Tareas planificadas: {len(trace.planner_tasks)}")
            print(f"  • Decisiones de routing: {len(trace.routing_trace)}")
            print(f"  • Agentes ejecutados: {len(trace.agent_executions)}")
            
            # PASO 2: EVALUACIÓN MODULAR (escala 1-4)
            print(f"\nIniciando evaluación por módulos...\n")
            
            eval_start_time = time.perf_counter()
            
            # 2.1 Evaluar Planner
            print("  [1/4] Evaluando Planner...")
            planner_eval = evaluate_planner(
                user_message=question,
                generated_tasks=trace.planner_tasks,
                expected_behavior=expected_behavior
            )
            planner_score = (
                planner_eval.correctness +
                planner_eval.completeness +
                planner_eval.precision +
                planner_eval.task_decomposition
            ) / 4
            print(f"        Score: {planner_score:.1f}/4")
            
            # 2.2 Evaluar Supervisor
            print("  [2/4] Evaluando Supervisor...")
            supervisor_eval = evaluate_supervisor(
                pending_tasks=trace.planner_tasks,
                routing_trace=trace.routing_trace,
                expected_behavior=expected_behavior
            )
            supervisor_score = (
                supervisor_eval.routing_accuracy +
                supervisor_eval.task_completion
            ) / 2
            print(f"        Score: {supervisor_score:.1f}/4")
            
            # 2.3 Evaluar cada Agente
            print("  [3/4] Evaluando Agentes Especializados...")
            agents_eval = []
            
            # Mapeo de herramientas disponibles por agente
            tools_map = {
                "Technical_Analyst": ["crypto_history_tool", "crypto_prediction_tool", "crypto_chart_tool"],
                "Fundamental_Analyst": ["crypto_rag_tool", "crypto_news_tool"],
                "Risk_Officer": ["crypto_volatility_tool"]
            }
            
            for execution in trace.agent_executions:
                agent_name = execution["agent"]
                agent_eval = evaluate_agent(
                    agent_name=agent_name,
                    current_task=execution["task"],
                    available_tools=tools_map.get(agent_name, []),
                    tools_used=execution["tools_used"],
                    tool_outputs=execution["tool_outputs"],
                    agent_response=execution["agent_response"],
                    expected_behavior=expected_behavior
                )
                agents_eval.append(agent_eval)
                
                agent_score = (
                    agent_eval.tool_selection +
                    agent_eval.tool_execution +
                    agent_eval.output_fidelity +
                    agent_eval.output_completeness +
                    agent_eval.hallucination_check
                ) / 5
                print(f"        • {agent_name}: {agent_score:.1f}/4")
            
            # 2.4 Evaluar Informe Final
            print("  [4/4] Evaluando Informe Final...")
            final_eval = evaluate_final_output(
                original_tasks=trace.planner_tasks,
                agent_outputs=[e["agent_response"] for e in trace.agent_executions],
                final_report=trace.final_answer,
                expected_behavior=expected_behavior
            )
            final_score = (
                final_eval.completeness +
                final_eval.accuracy +
                final_eval.structure +
                final_eval.chart_attribution
            ) / 4
            print(f"        Score: {final_score:.1f}/4")
            
            # PASO 3: EVALUACIÓN COMPREHENSIVA
            print(f"\nGenerando evaluación comprehensiva...")
            comprehensive_eval = evaluate_comprehensive(
                planner_eval=planner_eval,
                supervisor_eval=supervisor_eval,
                agents_eval=agents_eval,
                final_eval=final_eval
            )
            
            eval_elapsed_time = time.perf_counter() - eval_start_time
            
            print(f"\n{'─'*80}")
            print(f"RESULTADO FINAL: {comprehensive_eval.overall_score}/4")
            print(f"   Categoría de Error: {comprehensive_eval.error_category}")
            if comprehensive_eval.critical_failures:
                print(f"Fallos Críticos:")
                for failure in comprehensive_eval.critical_failures:
                    print(f"   • {failure}")
            print(f"\nResumen Ejecutivo:")
            print(f"   {comprehensive_eval.executive_summary}")
            print(f"{'─'*80}\n")
            
            # GUARDAR EN METRICS ACCUMULATOR
            trace_data = {
                'user_question': question,
                'planner_tasks': trace.planner_tasks,
                'routing_trace': trace.routing_trace,
                'agent_executions': trace.agent_executions,
                'sql_queries': [],
                'final_answer': trace.final_answer
            }
            
            llm_judge_data = {
                'comprehensive_eval': comprehensive_eval,
                'planner_score': planner_score,
                'supervisor_score': supervisor_score,
                'agents_avg_score': sum(
                    (a.tool_selection + a.tool_execution + a.output_fidelity + 
                     a.output_completeness + a.hallucination_check) / 5
                    for a in agents_eval
                ) / len(agents_eval) if agents_eval else 0,
                'final_output_score': final_score,
                'elapsed_time': eval_elapsed_time
            }
            
            metrics_logger.log_offline_evaluation(
                test_case={
                    'id': qid,
                    'query': question,
                    'difficulty': difficulty,
                    'category': category,
                    'expected_tasks': case.get('expected_tasks', [])
                },
                trace_data=trace_data,
                baseline_eval=None,  # Solo LLM-Judge
                llm_judge_data=llm_judge_data
            )
            
            # PASO 4: ALMACENAR RESULTADOS (para CSV legacy)
            result = {
                # Metadata
                "id": qid,
                "category": category,
                "difficulty": difficulty,
                "focus_area": focus_area,
                "question": question,
                "expected_behavior": expected_behavior,
                
                # Trazas de Ejecución
                "planner_tasks": "; ".join(trace.planner_tasks),
                "routing_decisions": len(trace.routing_trace),
                "agents_invoked": "; ".join([e["agent"] for e in trace.agent_executions]),
                
                # Scores por Módulo (escala 1-4)
                "planner_score": planner_score,
                "planner_correctness": planner_eval.correctness,
                "planner_completeness": planner_eval.completeness,
                "planner_precision": planner_eval.precision,
                "planner_errors": "; ".join(planner_eval.errors),
                
                "supervisor_score": supervisor_score,
                "supervisor_routing_accuracy": supervisor_eval.routing_accuracy,
                "supervisor_task_completion": supervisor_eval.task_completion,
                "supervisor_errors": "; ".join(supervisor_eval.errors),
                
                "agents_count": len(agents_eval),
                "agents_avg_score": llm_judge_data['agents_avg_score'],
                
                "final_output_score": final_score,
                "final_completeness": final_eval.completeness,
                "final_accuracy": final_eval.accuracy,
                "final_errors": "; ".join(final_eval.errors),
                
                # Evaluación Global (escala 1-4)
                "overall_score": comprehensive_eval.overall_score,
                "error_category": comprehensive_eval.error_category,
                "critical_failures": "; ".join(comprehensive_eval.critical_failures),
                "executive_summary": comprehensive_eval.executive_summary,
                "evaluation_time": eval_elapsed_time,
                
                # Outputs
                "final_answer": trace.final_answer[:500],
            }
            
            # Añadir detalles de cada agente
            for i, agent_eval in enumerate(agents_eval):
                prefix = f"agent_{i+1}"
                result.update({
                    f"{prefix}_name": agent_eval.agent_name,
                    f"{prefix}_tool_selection": agent_eval.tool_selection,
                    f"{prefix}_tool_execution": agent_eval.tool_execution,
                    f"{prefix}_output_fidelity": agent_eval.output_fidelity,
                    f"{prefix}_hallucination_check": agent_eval.hallucination_check,
                    f"{prefix}_errors": "; ".join(agent_eval.errors[:3])
                })
            
            all_results.append(result)
            
        except Exception as e:
            print(f"\nERROR CRÍTICO en {qid}: {str(e)}\n")
            all_results.append({
                "id": qid,
                "category": category,
                "question": question,
                "overall_score": 0,
                "error_category": "System_Error",
                "critical_failures": str(e),
                "executive_summary": f"Fallo del sistema: {str(e)}"
            })

    # EXPORTAR RESULTADOS
    print(f"\n{'='*80}")
    print("Exportando resultados...")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(all_results)
    
    # Guardar CSV completo
    df.to_csv(f"{FILE_PATH}_results.csv", index=False)
    print(f"Resultados completos guardados en: {FILE_PATH}_results.csv")
    
    # Guardar resumen ejecutivo
    summary = df[[
        "id", "category", "difficulty", "focus_area",
        "overall_score", "error_category",
        "planner_score", "supervisor_score", "agents_avg_score", "final_output_score"
    ]].copy()
    summary.to_csv(f"{FILE_PATH}_summary.csv", index=False)
    print(f"Resumen ejecutivo guardado en: {FILE_PATH}_summary.csv")
    
    # Info del accumulator
    print(f"\nMétricas acumuladas guardadas en:")
    print(f"   evaluation/accumulated_data/offline_metrics.csv")
    
    # Estadísticas globales (escala 1-4)
    print(f"\n{'='*80}")
    print("ESTADÍSTICAS GLOBALES")
    print(f"{'='*80}\n")
    print(f"Total de casos evaluados: {len(df)}")
    print(f"Score promedio global: {df['overall_score'].mean():.2f}/4")
    print(f"\nScore promedio por módulo:")
    print(f"  • Planner: {df['planner_score'].mean():.2f}/4")
    print(f"  • Supervisor: {df['supervisor_score'].mean():.2f}/4")
    print(f"  • Agentes: {df['agents_avg_score'].mean():.2f}/4")
    print(f"  • Output Final: {df['final_output_score'].mean():.2f}/4")
    
    print(f"\nCategorías de error:")
    error_counts = df['error_category'].value_counts()
    for error, count in error_counts.items():
        print(f"  • {error}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nCasos por dificultad:")
    diff_scores = df.groupby('difficulty')['overall_score'].agg(['mean', 'count'])
    for diff, row in diff_scores.iterrows():
        print(f"  • {diff}: {row['mean']:.2f}/4 (n={int(row['count'])})")
    
    print(f"\n{'='*80}")
    print("EVALUACIÓN COMPLETADA")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    run_comprehensive_evaluation()