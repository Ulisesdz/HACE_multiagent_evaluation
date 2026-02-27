from pydantic import BaseModel, Field
from pydantic.types import StringConstraints
from typing_extensions import Annotated
from typing import List

# MODELOS DE EVALUACIÓN GRANULAR
class PlannerEvaluation(BaseModel):
    """Evaluación específica del Planner"""
    correctness: int = Field(
        description="¿Identificó correctamente las tareas del mensaje? 0-10"
    )
    completeness: int = Field(
        description="¿Capturó TODAS las solicitudes sin omitir ninguna? 0-10"
    )
    precision: int = Field(
        description="¿Mantuvo adjetivos, cantidades y fechas exactas? 0-10"
    )
    task_decomposition: int = Field(
        description="¿Separó correctamente en tareas independientes? 0-10"
    )
    errors: List[str] = Field(
        description="Lista de errores específicos del Planner",
        default_factory=list
    )
    analysis: str = Field(
        description="Análisis detallado de la planificación"
    )


class SupervisorEvaluation(BaseModel):
    """Evaluación específica del Supervisor (Routing)"""
    routing_accuracy: int = Field(
        description="¿Eligió al agente correcto para cada tarea? 0-10"
    )
    task_completion: int = Field(
        description="¿Completó todas las tareas o terminó prematuramente? 0-10"
    )
    routing_decisions: List[dict] = Field(
        description="Lista de decisiones: [{'task': '...', 'agent': '...', 'correct': bool}]",
        default_factory=list
    )
    errors: List[str] = Field(
        description="Errores de enrutamiento específicos",
        default_factory=list
    )
    analysis: str = Field(
        description="Análisis del comportamiento del Supervisor"
    )


class AgentEvaluation(BaseModel):
    """Evaluación específica de un Agente"""
    agent_name: str = Field(
        description="Nombre del agente evaluado"
    )
    tool_selection: int = Field(
        description="¿Eligió las herramientas correctas? 0-10"
    )
    tool_execution: int = Field(
        description="¿Usó las herramientas correctamente (parámetros, SQL)? 0-10"
    )
    output_fidelity: int = Field(
        description="¿La respuesta es fiel a los datos de las herramientas? 0-10"
    )
    output_completeness: int = Field(
        description="¿Reportó TODOS los datos (Top 3 = 3 items, no resumen)? 0-10"
    )
    hallucination_check: int = Field(
        description="¿Inventó datos no presentes en las herramientas? 0=Inventó, 10=Fiel"
    )
    errors: List[str] = Field(
        description="Lista de errores específicos del agente",
        default_factory=list
    )
    analysis: str = Field(
        description="Análisis del desempeño del agente"
    )


class FinalOutputEvaluation(BaseModel):
    """Evaluación del informe final consolidado"""
    completeness: int = Field(
        description="¿Incluye TODAS las tareas solicitadas? 0-10"
    )
    accuracy: int = Field(
        description="¿Los datos son correctos y verificables? 0-10"
    )
    structure: int = Field(
        description="¿Está bien organizado por activos/tareas? 0-10"
    )
    chart_attribution: int = Field(
        description="¿Los gráficos están asociados al activo correcto? 0-10"
    )
    errors: List[str] = Field(
        description="Errores en el consolidado final",
        default_factory=list
    )
    analysis: str = Field(
        description="Análisis del informe final"
    )


class ComprehensiveEvaluation(BaseModel):
    """Evaluación completa del sistema end-to-end"""
    # Evaluaciones por etapa
    planner: PlannerEvaluation
    supervisor: SupervisorEvaluation
    agents: List[AgentEvaluation] = Field(
        description="Evaluación de cada agente que participó"
    )
    final_output: FinalOutputEvaluation
    
    # Métricas globales
    overall_score: int = Field(
        description="Score global del sistema (0-10), promedio ponderado"
    )
    critical_failures: List[str] = Field(
        description="Lista de fallos críticos que rompen el sistema",
        default_factory=list
    )
    error_category: str = Field(
        description=(
            "Categoría principal del error: "
            "'None', 'Planning_Error', 'Routing_Error', 'Tool_Error', "
            "'Fabrication', 'Incompleteness', 'Logic_Error', 'Loop_Error', "
            "'Risk_Negligence', 'Parametric_Leak'"
        )
    )
    
    # Análisis ejecutivo
    executive_summary: Annotated[str, StringConstraints(max_length=500)] = Field(
        description="Resumen ejecutivo de máximo 100 palabras"
    )