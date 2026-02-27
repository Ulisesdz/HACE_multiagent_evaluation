import sys
import json
import time
import pandas as pd
from langchain_core.messages import ToolMessage, AIMessage
from orchestrator.graph import build_graph
from evaluation.baseline.evaluator import evaluate_baseline

DATASET_PATH = "evaluation/llm_j/dataset.json"  # Mismo dataset que LLM-Judge
OUTPUT_PREFIX = "evaluation/baseline/dataset"


class TraceCollector:
    """Colector de trazas para Baseline (mismo que en app.py)"""
    
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
        """Captura las tareas del Planner"""
        self.planner_tasks = state.get("pending_tasks", [])
    
    def capture_supervisor_decision(self, state: dict):
        """Captura decisión de routing"""
        current_task = state.get("current_task", "")
        next_agent = state.get("next", "")
        
        if next_agent != "FINISH" and current_task:
            self.routing_trace.append({
                "task": current_task,
                "agent": next_agent
            })
    
    def capture_agent_execution(self, agent_name: str, messages: list, current_task: str = ""):
        """Captura ejecución del agente"""
        tools_used = []
        tool_outputs_text = []
        agent_response = ""
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tools_used.append(tool_call.get('name', 'unknown'))
                
                if msg.content and not msg.tool_calls:
                    agent_response = msg.content
            
            elif isinstance(msg, ToolMessage):
                tool_outputs_text.append(f"[{msg.name}]: {msg.content}")
        
        if tools_used or agent_response:
            self.agent_executions.append({
                "agent": agent_name,
                "task": current_task,
                "tools_used": tools_used,
                "tool_outputs": "\n".join(tool_outputs_text),
                "agent_response": agent_response
            })
    
    def capture_sql_query(self, task: str, tool_call_args: dict):
        """Captura queries SQL ejecutadas"""
        if 'query' in tool_call_args:
            self.sql_queries.append({
                'task': task,
                'sql': tool_call_args['query']
            })
    
    def capture_final_output(self, state: dict):
        """Captura el mensaje final del sistema"""
        messages = state.get("messages", [])
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                self.final_answer = msg.content
                break


def run_baseline_batch_evaluation():
    """
    Evaluación Batch con Baseline Metrics sobre el dataset completo.
    Genera CSV con métricas automáticas (routing, numeric, coverage, SQL).
    """
    app = build_graph()
    
    # Cargar Dataset
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ El archivo {DATASET_PATH} no existe.")
        return
    
    all_results = []
    
    print("=" * 80)
    print("🚀 INICIANDO EVALUACIÓN BASELINE (BATCH MODE)")
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
        print(f"📝 CASO {idx}/{len(dataset)}: {qid} | {category} | Dificultad: {difficulty}")
        print(f"   Pregunta: {question}")
        print(f"{'='*80}")
        
        try:
            trace = TraceCollector()
            trace.user_question = question
            
            initial_state = {"messages": [("user", question)]}
            final_state = None
            
            # EJECUCIÓN Y CAPTURA DE TRAZAS
            print("⚙️  Ejecutando sistema...")
            
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
                    
                    elif node_name in ["Technical_Analyst", "Fundamental_Analyst", "Risk_Officer"]:
                        current_task = node_output.get("current_task", "")
                        
                        # Fallback: usar última tarea del routing_trace
                        if not current_task and trace.routing_trace:
                            current_task = trace.routing_trace[-1].get("task", "")
                        
                        trace.capture_agent_execution(node_name, node_output.get("messages", []), current_task)
                        
                        # Capturar SQL si es Technical_Analyst
                        for msg in node_output.get("messages", []):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    if tool_call["name"] == "crypto_history_tool":
                                        if 'query' in tool_call["args"]:
                                            query_value = tool_call["args"]['query']
                                            
                                            # Solo capturar si es string (SQL válido)
                                            if isinstance(query_value, str) and query_value.strip():
                                                trace.capture_sql_query(current_task, tool_call["args"])
                        
                        print(f"   ✓ {node_name}: ejecutado")
            
            # Capturar output final
            if final_state is not None:
                trace.capture_final_output(final_state)
            else:
                raise RuntimeError("El grafo no produjo ningún estado final.")
            
            print(f"✅ Ejecución completada")
            
            # EVALUACIÓN BASELINE
            print(f"📐 Evaluando con Baseline Metrics...")
            
            trace_data = {
                'user_question': trace.user_question,
                'planner_tasks': trace.planner_tasks,
                'routing_trace': trace.routing_trace,
                'agent_executions': trace.agent_executions,
                'sql_queries': trace.sql_queries,
                'final_answer': trace.final_answer
            }
            
            baseline_eval = evaluate_baseline(trace_data)
            
            print(f"\n{'─'*60}")
            print(f"📊 RESULTADO:")
            print(f"   Baseline Score: {baseline_eval.baseline_score:.3f}")
            print(f"   • Routing F1:       {baseline_eval.routing_metrics.f1:.3f}")
            print(f"   • Numeric F1:       {baseline_eval.numeric_metrics.f1:.3f}")
            print(f"   • Hallucination:    {baseline_eval.numeric_metrics.hallucination_rate:.1%}")
            print(f"   • Task Coverage:    {baseline_eval.task_coverage_metrics.coverage:.1%}")
            print(f"   • SQL Correctness:  {baseline_eval.sql_metrics.correctness:.1%}")
            print(f"   • Evaluation Time:  {baseline_eval.evaluation_time:.3f}s")
            print(f"{'─'*60}\n")
            
            # ALMACENAR RESULTADOS
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
                "agents_invoked": "; ".join([e["agent"] for e in trace.agent_executions]),
                "sql_queries_count": len(trace.sql_queries),
                
                # Métricas Baseline
                "baseline_score": baseline_eval.baseline_score,
                "evaluation_time": baseline_eval.evaluation_time,
                
                # Routing Metrics
                "routing_accuracy": baseline_eval.routing_metrics.accuracy,
                "routing_precision": baseline_eval.routing_metrics.precision,
                "routing_recall": baseline_eval.routing_metrics.recall,
                "routing_f1": baseline_eval.routing_metrics.f1,
                
                # Numeric Metrics
                "numeric_precision": baseline_eval.numeric_metrics.precision,
                "numeric_recall": baseline_eval.numeric_metrics.recall,
                "numeric_f1": baseline_eval.numeric_metrics.f1,
                "hallucination_rate": baseline_eval.numeric_metrics.hallucination_rate,
                
                # Task Coverage
                "task_coverage": baseline_eval.task_coverage_metrics.coverage,
                "omission_rate": baseline_eval.task_coverage_metrics.omission_rate,
                "planned_tasks": baseline_eval.task_coverage_metrics.planned_tasks,
                "completed_tasks": baseline_eval.task_coverage_metrics.completed_tasks,
                
                # SQL Correctness
                "sql_correctness": baseline_eval.sql_metrics.correctness,
                "sql_total_queries": baseline_eval.sql_metrics.total_queries,
                "sql_correct_queries": baseline_eval.sql_metrics.correct_queries,
                "sql_violations": "; ".join(baseline_eval.sql_metrics.violations),
                
                # Per-Class Routing (opcional)
                "routing_per_class": str(baseline_eval.routing_metrics.per_class) if baseline_eval.routing_metrics.per_class else "",
                
                # Output
                "final_answer": trace.final_answer[:500],
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO en {qid}: {str(e)}\n")
            all_results.append({
                "id": qid,
                "category": category,
                "difficulty": difficulty,
                "question": question,
                "baseline_score": 0.0,
                "evaluation_time": 0.0,
                "routing_f1": 0.0,
                "numeric_f1": 0.0,
                "hallucination_rate": 1.0,
                "task_coverage": 0.0,
                "sql_correctness": 0.0,
                "error": str(e)
            })
    
    # EXPORTAR RESULTADOS
    print(f"\n{'='*80}")
    print("💾 Exportando resultados...")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(all_results)
    
    # Guardar CSV completo
    full_path = f"{OUTPUT_PREFIX}_baseline_results.csv"
    df.to_csv(full_path, index=False)
    print(f"✅ Resultados completos: {full_path}")
    
    # Guardar resumen ejecutivo
    summary_cols = [
        "id", "category", "difficulty", "focus_area",
        "baseline_score", "routing_f1", "numeric_f1", 
        "hallucination_rate", "task_coverage", "sql_correctness",
        "evaluation_time"
    ]
    summary = df[summary_cols].copy()
    summary_path = f"{OUTPUT_PREFIX}_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"✅ Resumen ejecutivo: {summary_path}")
    
    # ESTADÍSTICAS GLOBALES
    print(f"\n{'='*80}")
    print("📊 ESTADÍSTICAS GLOBALES (BASELINE)")
    print(f"{'='*80}\n")
    print(f"Total de casos evaluados: {len(df)}")
    print(f"Baseline Score promedio:  {df['baseline_score'].mean():.3f}")
    print(f"Tiempo promedio:          {df['evaluation_time'].mean():.3f}s")
    
    print(f"\n📈 Métricas Promedio:")
    print(f"  • Routing F1:       {df['routing_f1'].mean():.3f}")
    print(f"  • Numeric F1:       {df['numeric_f1'].mean():.3f}")
    print(f"  • Hallucination:    {df['hallucination_rate'].mean():.1%}")
    print(f"  • Task Coverage:    {df['task_coverage'].mean():.1%}")
    print(f"  • SQL Correctness:  {df['sql_correctness'].mean():.1%}")
    
    print(f"\n🎯 Por Dificultad:")
    diff_stats = df.groupby('difficulty').agg({
        'baseline_score': ['mean', 'std', 'count'],
        'routing_f1': 'mean',
        'numeric_f1': 'mean'
    }).round(3)
    print(diff_stats)
    
    print(f"\n📂 Por Categoría (Top 5):")
    cat_stats = df.groupby('category')['baseline_score'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(5)
    print(cat_stats)
    
    print(f"\n{'='*80}")
    print("✅ EVALUACIÓN BASELINE COMPLETADA")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    run_baseline_batch_evaluation()