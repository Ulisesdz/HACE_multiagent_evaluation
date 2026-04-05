"""
Layer 3: LLM-Judge Selectivo - Análisis profundo solo cuando necesario
Propósito: Evaluación profunda de módulos problemáticos
"""

from typing import Dict, List, Any
from evaluation.llm_j.judge import (
    evaluate_planner,
    evaluate_supervisor,
    evaluate_agent,
    evaluate_final_output,
)


class SelectiveLLMJudge:
    """
    Capa 3: LLM-Judge selectivo
    Solo evalúa módulos que necesitan análisis profundo
    """

    def __init__(self):
        self.tools_map = {
            "Technical_Analyst": [
                "crypto_history_tool",
                "crypto_prediction_tool",
                "crypto_chart_tool",
            ],
            "Fundamental_Analyst": ["crypto_rag_tool", "crypto_news_tool"],
            "Risk_Officer": ["crypto_volatility_tool"],
        }

    def evaluate_selectively(
        self,
        trace_data: Dict[str, Any],
        layer1_results: Dict[str, Dict],
        layer2_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluación selectiva de módulos problemáticos con LLM-Judge

        Args:
            trace_data: Trazas del sistema
            layer1_results: Resultados de guardrails
            layer2_results: Resultados semánticos

        Returns:
            Dict con evaluaciones por módulo y overall_score
        """
        evaluations_needed = self._determine_modules_to_evaluate(
            layer1_results, layer2_results
        )

        results = {
            "modules_evaluated": evaluations_needed,
            "evaluation_complete": False,
        }

        # Evaluar solo módulos necesarios
        if "planner" in evaluations_needed:
            results["planner"] = self._evaluate_planner_module(trace_data)
        else:
            results["planner"] = {"score": 4.0, "skipped": True}

        if "supervisor" in evaluations_needed:
            results["supervisor"] = self._evaluate_supervisor_module(trace_data)
        else:
            results["supervisor"] = {"score": 4.0, "skipped": True}

        if "agents" in evaluations_needed:
            results["agents"] = self._evaluate_agents_module(trace_data)
        else:
            results["agents"] = {"score": 4.0, "skipped": True}

        if "final_output" in evaluations_needed:
            results["final_output"] = self._evaluate_final_output_module(trace_data)
        else:
            results["final_output"] = {"score": 4.0, "skipped": True}

        # Calcular overall score
        results["overall_score"] = self._calculate_overall_score(results)
        results["evaluation_complete"] = True

        return results

    def _determine_modules_to_evaluate(
        self, layer1_results: Dict, layer2_results: Dict
    ) -> List[str]:
        """
        Decidir qué módulos necesitan evaluación profunda

        Args:
            layer1_results: Resultados de Layer 1
            layer2_results: Resultados de Layer 2

        Returns:
            Lista de nombres de módulos ['planner', 'supervisor', 'agents', 'final_output']
        """
        modules = []

        # Planner: Si task fidelity es baja
        if not layer2_results.get("task_fidelity", {}).get("pass", True):
            modules.append("planner")

        # Planner: Si completitud estructural falló
        if not layer1_results.get("validate_completeness", {}).get("pass", True):
            coverage = layer1_results["validate_completeness"].get("coverage_rate", 1.0)
            if coverage < 0.8:
                modules.append("planner")

        # Supervisor: Si routing quality es bajo
        if not layer2_results.get("routing_quality", {}).get("pass", True):
            modules.append("supervisor")

        # Supervisor: Si routing syntax falló
        if not layer1_results.get("validate_routing_syntax", {}).get("pass", True):
            modules.append("supervisor")

        # Agents: Si algún agent fidelity es bajo
        agent_fidelities = layer2_results.get("agent_fidelities", [])
        if any(not af.get("pass", True) for af in agent_fidelities):
            modules.append("agents")

        # Agents: Si anomalías numéricas
        if not layer1_results.get("validate_numeric_ranges", {}).get("pass", True):
            modules.append("agents")

        # Final Output: Si report completeness es bajo
        if not layer2_results.get("report_completeness", {}).get("pass", True):
            modules.append("final_output")

        # Final Output: Si gráficos fantasma
        if not layer1_results.get("validate_chart_mentions", {}).get("pass", True):
            modules.append("final_output")

        return list(set(modules))  # Eliminar duplicados

    def _evaluate_planner_module(self, trace_data: Dict) -> Dict:
        """
        Evaluar Planner con LLM-Judge

        Args:
            trace_data: Dict con 'user_question' y 'planner_tasks'

        Returns:
            Dict con 'score' (1-4), 'evaluation', 'skipped'
        """
        try:
            planner_eval = evaluate_planner(
                user_message=trace_data.get("user_question", ""),
                generated_tasks=trace_data.get("planner_tasks", []),
                expected_behavior="El Planner debe identificar todas las tareas del mensaje y mantener precisión literal.",
            )

            score = (
                planner_eval.correctness
                + planner_eval.completeness
                + planner_eval.precision
                + planner_eval.task_decomposition
            ) / 4

            return {"score": score, "evaluation": planner_eval, "skipped": False}

        except Exception as e:
            return {"score": 2.0, "error": str(e), "skipped": False}

    def _evaluate_supervisor_module(self, trace_data: Dict) -> Dict:
        """
        Evaluar Supervisor con LLM-Judge

        Args:
            trace_data: Dict con 'planner_tasks' y 'routing_trace'

        Returns:
            Dict con 'score' (1-4), 'evaluation', 'skipped'
        """
        try:
            supervisor_eval = evaluate_supervisor(
                pending_tasks=trace_data.get("planner_tasks", []),
                routing_trace=trace_data.get("routing_trace", []),
                expected_behavior="El Supervisor debe enrutar correctamente cada tarea al especialista apropiado.",
            )

            score = (
                supervisor_eval.routing_accuracy + supervisor_eval.task_completion
            ) / 2

            return {"score": score, "evaluation": supervisor_eval, "skipped": False}

        except Exception as e:
            return {"score": 2.0, "error": str(e), "skipped": False}

    def _evaluate_agents_module(self, trace_data: Dict) -> Dict:
        """
        Evaluar Agentes con LLM-Judge

        Args:
            trace_data: Dict con 'agent_executions'

        Returns:
            Dict con 'score' (1-4), 'evaluations', 'skipped'
        """
        try:
            agents_eval = []

            for execution in trace_data.get("agent_executions", []):
                agent_name = execution["agent"]
                agent_eval = evaluate_agent(
                    agent_name=agent_name,
                    current_task=execution.get("task", "No especificada"),
                    available_tools=self.tools_map.get(agent_name, []),
                    tools_used=execution["tools_used"],
                    tool_outputs=execution["tool_outputs"],
                    agent_response=execution["agent_response"],
                    expected_behavior=f"{agent_name} debe usar las herramientas correctamente y reportar datos fielmente.",
                )
                agents_eval.append(agent_eval)

            if agents_eval:
                avg_score = sum(
                    (
                        a.tool_selection
                        + a.tool_execution
                        + a.output_fidelity
                        + a.output_completeness
                        + a.hallucination_check
                    )
                    / 5
                    for a in agents_eval
                ) / len(agents_eval)
            else:
                avg_score = 4.0

            return {"score": avg_score, "evaluations": agents_eval, "skipped": False}

        except Exception as e:
            return {"score": 2.0, "error": str(e), "skipped": False}

    def _evaluate_final_output_module(self, trace_data: Dict) -> Dict:
        """
        Evaluar Output Final con LLM-Judge

        Args:
            trace_data: Dict con 'planner_tasks', 'agent_executions', 'final_answer'

        Returns:
            Dict con 'score' (1-4), 'evaluation', 'skipped'
        """
        try:
            final_eval = evaluate_final_output(
                original_tasks=trace_data.get("planner_tasks", []),
                agent_outputs=[
                    e["agent_response"] for e in trace_data.get("agent_executions", [])
                ],
                final_report=trace_data.get("final_answer", ""),
                expected_behavior="El informe final debe consolidar todos los outputs de forma completa y precisa.",
            )

            score = (
                final_eval.completeness
                + final_eval.accuracy
                + final_eval.structure
                + final_eval.chart_attribution
            ) / 4

            return {"score": score, "evaluation": final_eval, "skipped": False}

        except Exception as e:
            return {"score": 2.0, "error": str(e), "skipped": False}

    def _calculate_overall_score(self, results: Dict) -> float:
        """
        Calcular overall score con ponderación estándar
        Pesos: Planner 20%, Supervisor 25%, Agents 40%, Final 15%

        Args:
            results: Dict con scores por módulo

        Returns:
            Score global (1-4)
        """
        weights = {
            "planner": 0.20,
            "supervisor": 0.25,
            "agents": 0.40,
            "final_output": 0.15,
        }

        overall_score = 0.0

        for module, weight in weights.items():
            module_data = results.get(module, {})
            score = module_data.get("score", 4.0)
            overall_score += score * weight

        return overall_score
