from langchain_core.prompts import ChatPromptTemplate
from typing import List
from orchestrator.config import get_llm
from evaluation.llm_j.prompts import (
    COMPREHENSIVE_JUDGE_PROMPT,
    FINAL_OUTPUT_JUDGE_PROMPT,
    AGENT_JUDGE_PROMPT,
    SUPERVISOR_JUDGE_PROMPT,
    PLANNER_JUDGE_PROMPT
)
from evaluation.llm_j.state import (
    PlannerEvaluation,
    SupervisorEvaluation,
    AgentEvaluation,
    FinalOutputEvaluation,
    ComprehensiveEvaluation
)

judge_llm = get_llm()

def evaluate_planner(user_message: str, generated_tasks: list, expected_behavior: str) -> PlannerEvaluation:
    """Evalúa el módulo Planner"""
    structured_llm = judge_llm.with_structured_output(PlannerEvaluation)
    prompt = ChatPromptTemplate.from_template(PLANNER_JUDGE_PROMPT)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "user_message": user_message,
            "generated_tasks": generated_tasks,
            "expected_behavior": expected_behavior
        })
        return result
    except Exception as e:
        return PlannerEvaluation(
            correctness=0,
            completeness=0,
            precision=0,
            task_decomposition=0,
            errors=[f"Error de evaluación: {str(e)}"],
            analysis=f"Fallo crítico en evaluación del Planner: {str(e)}"
        )


def evaluate_supervisor(pending_tasks: list, routing_trace: list, expected_behavior: str) -> SupervisorEvaluation:
    """Evalúa el módulo Supervisor"""
    structured_llm = judge_llm.with_structured_output(SupervisorEvaluation)
    prompt = ChatPromptTemplate.from_template(SUPERVISOR_JUDGE_PROMPT)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "pending_tasks": pending_tasks,
            "routing_trace": routing_trace,
            "expected_behavior": expected_behavior
        })
        return result
    except Exception as e:
        return SupervisorEvaluation(
            routing_accuracy=0,
            task_completion=0,
            routing_decisions=[],
            errors=[f"Error de evaluación: {str(e)}"],
            analysis=f"Fallo crítico en evaluación del Supervisor: {str(e)}"
        )


def evaluate_agent(
    agent_name: str,
    current_task: str,
    available_tools: list,
    tools_used: list,
    tool_outputs: str,
    agent_response: str,
    expected_behavior: str
) -> AgentEvaluation:
    """Evalúa un agente específico"""
    structured_llm = judge_llm.with_structured_output(AgentEvaluation)
    prompt = ChatPromptTemplate.from_template(AGENT_JUDGE_PROMPT)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "agent_name": agent_name,
            "current_task": current_task,
            "available_tools": available_tools,
            "tools_used": tools_used,
            "tool_outputs": tool_outputs[:2000],  # Truncar si es muy largo
            "agent_response": agent_response[:1000],
            "expected_behavior": expected_behavior
        })
        return result
    except Exception as e:
        return AgentEvaluation(
            agent_name=agent_name,
            tool_selection=0,
            tool_execution=0,
            output_fidelity=0,
            output_completeness=0,
            hallucination_check=0,
            errors=[f"Error de evaluación: {str(e)}"],
            analysis=f"Fallo crítico en evaluación del agente: {str(e)}"
        )


def evaluate_final_output(
    original_tasks: list,
    agent_outputs: list,
    final_report: str,
    expected_behavior: str
) -> FinalOutputEvaluation:
    """Evalúa el informe final consolidado"""
    structured_llm = judge_llm.with_structured_output(FinalOutputEvaluation)
    prompt = ChatPromptTemplate.from_template(FINAL_OUTPUT_JUDGE_PROMPT)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "original_tasks": original_tasks,
            "agent_outputs": agent_outputs,
            "final_report": final_report[:2000],
            "expected_behavior": expected_behavior
        })
        return result
    except Exception as e:
        return FinalOutputEvaluation(
            completeness=0,
            accuracy=0,
            structure=0,
            chart_attribution=0,
            errors=[f"Error de evaluación: {str(e)}"],
            analysis=f"Fallo crítico en evaluación del output final: {str(e)}"
        )


def evaluate_comprehensive(
    planner_eval: PlannerEvaluation,
    supervisor_eval: SupervisorEvaluation,
    agents_eval: List[AgentEvaluation],
    final_eval: FinalOutputEvaluation
) -> ComprehensiveEvaluation:
    """
    Evaluación comprehensiva del sistema completo sin dependencia de cálculo de LLM
    """
    
    # Calcular scores promedio de cada módulo
    planner_avg = (
        planner_eval.correctness + 
        planner_eval.completeness + 
        planner_eval.precision + 
        planner_eval.task_decomposition
    ) / 4
    
    supervisor_avg = (
        supervisor_eval.routing_accuracy + 
        supervisor_eval.task_completion
    ) / 2
    
    agents_avg = sum(
        (a.tool_selection + a.tool_execution + a.output_fidelity + 
         a.output_completeness + a.hallucination_check) / 5
        for a in agents_eval
    ) / len(agents_eval) if agents_eval else 0
    
    final_avg = (
        final_eval.completeness + 
        final_eval.accuracy + 
        final_eval.structure + 
        final_eval.chart_attribution
    ) / 4
    
    # Calcular overall score con ponderación
    overall_score_calculated = int(
        planner_avg * 0.20 +
        supervisor_avg * 0.25 +
        agents_avg * 0.40 +
        final_avg * 0.15
    )
    
    # Determinar error category basado en scores
    if overall_score_calculated >= 8:
        error_category = "None"
    elif planner_avg < 5:
        error_category = "Planning_Error"
    elif supervisor_avg < 5:
        error_category = "Routing_Error"
    elif any(a.tool_execution < 5 for a in agents_eval):
        error_category = "Tool_Error"
    elif any(a.hallucination_check < 3 for a in agents_eval):
        error_category = "Fabrication"
    elif planner_eval.completeness < 5 or final_eval.completeness < 5:
        error_category = "Incompleteness"
    else:
        error_category = "None"
    
    # Identificar critical failures (solo los realmente críticos)
    critical_failures = []
    
    if any(a.hallucination_check < 3 for a in agents_eval):
        critical_failures.append("Fabricación de datos detectada")
    
    if supervisor_avg < 5 and len(agents_eval) > 1:
        critical_failures.append("Routing incorrecto en múltiples tareas")
    
    # Buscar omisión de advertencias de riesgo ALTO
    for agent in agents_eval:
        if agent.agent_name == "Risk_Officer":
            if "ALTO" in str(agent.errors) or "HIGH" in str(agent.errors):
                critical_failures.append("Omisión de advertencia de riesgo alto")
    
    # Generar executive summary
    if overall_score_calculated >= 9:
        estado = "EXCELENTE"
        descripcion = "Todos los módulos funcionan óptimamente."
    elif overall_score_calculated >= 7:
        estado = "BUENO"
        descripcion = "Funcionamiento correcto con pequeñas áreas de mejora."
    elif overall_score_calculated >= 5:
        estado = "MEJORABLE"
        descripcion = "Errores moderados detectados que requieren atención."
    else:
        estado = "CRÍTICO"
        descripcion = "Errores graves que afectan funcionalidad."
    
    # Identificar módulo con peor desempeño
    scores_modulos = [
        ("Planner", planner_avg),
        ("Supervisor", supervisor_avg),
        ("Agentes", agents_avg),
        ("Output Final", final_avg)
    ]
    peor_modulo, peor_score = min(scores_modulos, key=lambda x: x[1])
    
    if peor_score < 8:
        summary = (
            f"Sistema en estado {estado} (score: {overall_score_calculated}/10). "
            f"{descripcion} Módulo con menor desempeño: {peor_modulo} ({peor_score:.1f}/10)."
        )
    else:
        summary = f"Sistema en estado {estado} (score: {overall_score_calculated}/10). {descripcion}"
    
    # Intentar obtener un resumen narrativo del LLM
    try:
        structured_llm = judge_llm.with_structured_output(ComprehensiveEvaluation)
        prompt = ChatPromptTemplate.from_template(COMPREHENSIVE_JUDGE_PROMPT)
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "planner_eval": planner_eval.model_dump_json(),
            "supervisor_eval": supervisor_eval.model_dump_json(),
            "agents_eval": [a.model_dump_json() for a in agents_eval],
            "final_eval": final_eval.model_dump_json()
        })
        
        # Forzar los valores calculados manualmente
        result.planner = planner_eval
        result.supervisor = supervisor_eval
        result.agents = agents_eval
        result.final_output = final_eval
        result.overall_score = overall_score_calculated
        result.error_category = error_category
        
        # Si el LLM generó un mejor resumen, usarlo
        if result.executive_summary and len(result.executive_summary) > 10:
            summary = result.executive_summary
        
        result.executive_summary = summary
        result.critical_failures = critical_failures if critical_failures else []
        
        return result
        
    except Exception as e:

        # FALLBACK COMPLETO: Crear el objeto manualmente

        print(f"[JUDGE WARNING] LLM falló, usando fallback manual: {str(e)}")
        
        return ComprehensiveEvaluation(
            planner=planner_eval,
            supervisor=supervisor_eval,
            agents=agents_eval,
            final_output=final_eval,
            overall_score=overall_score_calculated,
            critical_failures=critical_failures,
            error_category=error_category,
            executive_summary=summary
        )
