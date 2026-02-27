from pydantic import BaseModel, Field
from typing import List, Dict

# ============================================================================
# MODELOS DE EVALUACIÓN BASELINE
# ============================================================================

class RoutingMetrics(BaseModel):
    """Métricas de precisión de routing"""
    accuracy: float = Field(description="Precisión global del routing (0-1)")
    precision: float = Field(description="Precisión promedio por clase (0-1)")
    recall: float = Field(description="Recall promedio por clase (0-1)")
    f1: float = Field(description="F1-Score promedio (0-1)")
    per_class: Dict[str, Dict[str, float]] = Field(
        description="Métricas por agente",
        default_factory=dict
    )
    confusion_matrix: Dict[str, Dict[str, int]] = Field(
        description="Matriz de confusión",
        default_factory=dict
    )


class NumericMetrics(BaseModel):
    """Métricas de exactitud numérica (detección de alucinaciones)"""
    precision: float = Field(description="% de números del agente que están en tool (0-1)")
    recall: float = Field(description="% de números del tool reportados por el agente (0-1)")
    f1: float = Field(description="F1-Score de fidelidad numérica (0-1)")
    hallucination_rate: float = Field(description="% de números inventados (0-1)")


class TaskCoverageMetrics(BaseModel):
    """Métricas de completitud de tareas"""
    coverage: float = Field(description="% de tareas completadas (0-1)")
    omission_rate: float = Field(description="% de tareas omitidas (0-1)")
    planned_tasks: int = Field(description="Número de tareas planificadas")
    completed_tasks: int = Field(description="Número de tareas completadas")


class SQLMetrics(BaseModel):
    """Métricas de corrección SQL"""
    correctness: float = Field(description="Score de corrección (0-1)")
    violations: List[str] = Field(
        description="Lista de patrones violados",
        default_factory=list
    )
    total_queries: int = Field(description="Número total de queries evaluadas", default=0)
    correct_queries: int = Field(description="Número de queries correctas", default=0)


class BaselineEvaluation(BaseModel):
    """Evaluación completa con métricas baseline"""
    # Métricas individuales
    routing_metrics: RoutingMetrics
    numeric_metrics: NumericMetrics
    task_coverage_metrics: TaskCoverageMetrics
    sql_metrics: SQLMetrics
    
    # Score agregado
    baseline_score: float = Field(description="Score baseline global (0-1)")
    
    # Metadata
    total_tasks: int = Field(description="Número total de tareas evaluadas")
    evaluation_time: float = Field(description="Tiempo de evaluación en segundos")