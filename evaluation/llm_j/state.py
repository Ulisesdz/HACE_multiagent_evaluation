from pydantic import BaseModel, Field
from pydantic.types import StringConstraints
from typing_extensions import Annotated
from typing import List


class PlannerEvaluation(BaseModel):
    """Evaluación del Planner (escala 1-4)"""

    correctness: int = Field(
        ge=1, le=4, description="¿Identificó correctamente las tareas? 1-4"
    )
    completeness: int = Field(
        ge=1, le=4, description="¿Capturó TODAS las solicitudes? 1-4"
    )
    precision: int = Field(ge=1, le=4, description="¿Mantuvo detalles importantes? 1-4")
    task_decomposition: int = Field(
        ge=1, le=4, description="¿Separó correctamente en tareas independientes? 1-4"
    )
    errors: List[str] = Field(
        default_factory=list, description="Lista de errores específicos"
    )
    analysis: str = Field(
        description="Análisis detallado de la planificación (2-3 oraciones)"
    )


class SupervisorEvaluation(BaseModel):
    """Evaluación del Supervisor (escala 1-4)"""

    routing_accuracy: int = Field(
        ge=1, le=4, description="¿Eligió al agente correcto? 1-4"
    )
    task_completion: int = Field(
        ge=1, le=4, description="¿Completó todas las tareas? 1-4"
    )
    routing_decisions: List[dict] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    analysis: str = Field(description="Análisis del comportamiento del Supervisor")


class AgentEvaluation(BaseModel):
    """Evaluación de un Agente (escala 1-4)"""

    agent_name: str
    tool_selection: int = Field(
        ge=1, le=4, description="¿Eligió las herramientas correctas? 1-4"
    )
    tool_execution: int = Field(ge=1, le=4, description="¿Las usó correctamente? 1-4")
    output_fidelity: int = Field(
        ge=1, le=4, description="¿La respuesta es fiel a los datos? 1-4"
    )
    output_completeness: int = Field(
        ge=1, le=4, description="¿Reportó TODOS los datos? 1-4"
    )
    hallucination_check: int = Field(
        ge=1, le=4, description="¿Inventó datos? 1=sí, 4=no"
    )
    errors: List[str] = Field(default_factory=list)
    analysis: str = Field(description="Análisis del desempeño del agente")


class FinalOutputEvaluation(BaseModel):
    """Evaluación del informe final (escala 1-4)"""

    completeness: int = Field(ge=1, le=4, description="¿Incluye TODAS las tareas? 1-4")
    accuracy: int = Field(ge=1, le=4, description="¿Los datos son correctos? 1-4")
    structure: int = Field(ge=1, le=4, description="¿Está bien organizado? 1-4")
    chart_attribution: int = Field(ge=1, le=4, description="¿Gráficos correctos? 1-4")
    errors: List[str] = Field(default_factory=list)
    analysis: str = Field(description="Análisis del informe final")


class ComprehensiveEvaluation(BaseModel):
    """Evaluación completa del sistema (escala 1-4)"""

    planner: PlannerEvaluation
    supervisor: SupervisorEvaluation
    agents: List[AgentEvaluation]
    final_output: FinalOutputEvaluation

    overall_score: int = Field(ge=1, le=4, description="Score global (1-4)")
    critical_failures: List[str] = Field(default_factory=list)
    error_category: str = Field(
        description="None/Planning_Error/Routing_Error/Tool_Error/Fabrication/Incompleteness"
    )

    # Análisis ejecutivo
    executive_summary: Annotated[str, StringConstraints(max_length=500)] = Field(
        description="Resumen ejecutivo de máximo 100 palabras"
    )
