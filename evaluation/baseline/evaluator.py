import time
from typing import Dict
from evaluation.baseline.metrics import BaselineMetricsCalculator
from evaluation.baseline.state import (
    BaselineEvaluation,
    RoutingMetrics,
    NumericMetrics,
    TaskCoverageMetrics,
    SQLMetrics,
)


def evaluate_baseline(trace_data: Dict) -> BaselineEvaluation:
    """
    Evaluación baseline completa del sistema.

    Args:
        trace_data: Diccionario con datos capturados del trace:
            {
                'user_question': str,
                'planner_tasks': List[str],
                'routing_trace': List[Dict],
                'agent_executions': List[Dict],
                'sql_queries': List[Dict],
                'final_answer': str
            }

    Returns:
        BaselineEvaluation con todas las métricas calculadas
    """
    start_time = time.perf_counter()

    calculator = BaselineMetricsCalculator()

    # 1. Routing Metrics
    routing_data = calculator.evaluate_routing_accuracy(
        routing_trace=trace_data.get("routing_trace", [])
    )

    routing_metrics = RoutingMetrics(
        accuracy=routing_data["accuracy"],
        precision=routing_data["precision"],
        recall=routing_data["recall"],
        f1=routing_data["f1"],
        per_class=routing_data["per_class"],
        confusion_matrix=routing_data["confusion_matrix"],
    )

    # 2. Numeric Metrics
    tool_outputs = []
    agent_responses = []

    for execution in trace_data.get("agent_executions", []):
        tool_outputs.append(execution.get("tool_outputs", ""))
        agent_responses.append(execution.get("agent_response", ""))

    numeric_data = calculator.evaluate_numeric_accuracy(
        tool_outputs=tool_outputs, agent_responses=agent_responses
    )

    numeric_metrics = NumericMetrics(
        precision=numeric_data["precision"],
        recall=numeric_data["recall"],
        f1=numeric_data["f1"],
        hallucination_rate=numeric_data["hallucination_rate"],
    )

    # 3. Task Coverage Metrics
    completed_tasks = [
        exec["task"]
        for exec in trace_data.get("agent_executions", [])
        if "task" in exec
    ]

    coverage_data = calculator.evaluate_task_coverage(
        planned_tasks=trace_data.get("planner_tasks", []),
        completed_tasks=completed_tasks,
    )

    task_coverage_metrics = TaskCoverageMetrics(
        coverage=coverage_data["coverage"],
        omission_rate=coverage_data["omission_rate"],
        planned_tasks=coverage_data["planned_tasks"],
        completed_tasks=coverage_data["completed_tasks"],
    )

    # 4. SQL Metrics
    sql_data = calculator.evaluate_sql_correctness(
        sql_queries=trace_data.get("sql_queries", [])
    )

    sql_metrics = SQLMetrics(
        correctness=sql_data["correctness"],
        violations=sql_data["violations"],
        total_queries=sql_data["total_queries"],
        correct_queries=sql_data["correct_queries"],
    )

    # 5. Baseline Score Agregado
    baseline_score = (
        routing_metrics.f1 * 0.30
        + numeric_metrics.f1 * 0.30
        + task_coverage_metrics.coverage * 0.25
        + sql_metrics.correctness * 0.15
    )

    evaluation_time = time.perf_counter() - start_time

    return BaselineEvaluation(
        routing_metrics=routing_metrics,
        numeric_metrics=numeric_metrics,
        task_coverage_metrics=task_coverage_metrics,
        sql_metrics=sql_metrics,
        baseline_score=baseline_score,
        total_tasks=len(trace_data.get("planner_tasks", [])),
        evaluation_time=evaluation_time,
    )
