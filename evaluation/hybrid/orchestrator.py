import time
from typing import Dict, Any
from evaluation.hybrid.layer1_guardrails import GuardrailsValidator
from evaluation.hybrid.layer2_semantic import SemanticEvaluator
from evaluation.hybrid.layer3_llm import SelectiveLLMJudge
from evaluation.hybrid.scorer import HybridScorer


class HybridEvaluator:
    """
    HACE - Hybrid Agent Comprehensive Evaluator

    Arquitectura de 3 capas:
    - Layer 1: Guardrails (Deterministic)
    - Layer 2: Semantic Evaluators (ML)
    - Layer 3: LLM-Judge Selectivo
    """

    def __init__(self):
        self.layer1 = GuardrailsValidator()
        self.layer2 = SemanticEvaluator()
        self.layer3 = SelectiveLLMJudge()
        self.scorer = HybridScorer()

    def evaluate(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pipeline completo de evaluación híbrida

        Args:
            trace_data: Dict con trazas del sistema (user_question, planner_tasks,
                       routing_trace, agent_executions, final_answer, etc.)

        Returns:
            Dict con evaluación completa y scores por capa
        """
        start_time = time.perf_counter()

        # CAPA 1: GUARDRAILS (Deterministic Validators)
        layer1_start = time.perf_counter()
        layer1_results = self.layer1.validate_all(trace_data)
        layer1_time = time.perf_counter() - layer1_start

        # Identificar fallos críticos
        critical_failures = self.layer1.get_critical_failures(layer1_results)

        # CAPA 2: SEMANTIC EVALUATORS
        layer2_start = time.perf_counter()
        layer2_results = self.layer2.evaluate_all(trace_data)
        layer2_time = time.perf_counter() - layer2_start

        # DECISIÓN: ¿Necesitamos Capa 3?
        need_layer3, escalation_reason = self._should_escalate_to_layer3(
            layer1_results, layer2_results, critical_failures
        )

        # CAPA 3: LLM-JUDGE SELECTIVO (Conditional)
        layer3_results = None
        layer3_time = 0.0

        if need_layer3:
            layer3_start = time.perf_counter()
            layer3_results = self.layer3.evaluate_selectively(
                trace_data, layer1_results, layer2_results
            )
            layer3_time = time.perf_counter() - layer3_start

        # SCORING FINAL (Multi-Layer Fusion)
        final_evaluation = self.scorer.calculate_final_score(
            layer1_results, layer2_results, layer3_results
        )

        # Tiempo total
        total_time = time.perf_counter() - start_time

        return {
            # Scores
            "final_score": final_evaluation["final_score"],
            "quality_label": self.scorer.get_quality_label(
                final_evaluation["final_score"]
            ),
            "confidence": final_evaluation["confidence"],
            # Desglose por capa
            "layer1_score": final_evaluation["layer1_score"],
            "layer2_score": final_evaluation["layer2_score"],
            "layer3_score": final_evaluation["layer3_score"],
            # Metadata
            "layer3_used": final_evaluation["layer3_used"],
            "escalation_reason": escalation_reason if need_layer3 else None,
            "critical_failures": critical_failures,
            # Detalles completos
            "layer1_details": layer1_results,
            "layer2_details": layer2_results,
            "layer3_details": layer3_results,
            # Tiempos
            "evaluation_time": total_time,
            "layer1_time": layer1_time,
            "layer2_time": layer2_time,
            "layer3_time": layer3_time,
            # Weights usados
            "weights_used": final_evaluation["weights_used"],
        }

    def _should_escalate_to_layer3(
        self,
        layer1_results: Dict[str, Dict],
        layer2_results: Dict[str, Any],
        critical_failures: list,
    ) -> tuple:
        """
        Decidir si se necesita Capa 3 (LLM-Judge)

        Args:
            layer1_results: Resultados de guardrails
            layer2_results: Resultados semánticos
            critical_failures: Lista de fallos críticos de Layer 1

        Returns:
            (bool, str): (need_escalation, reason)
        """
        # Razón 1: Fallos críticos en Capa 1
        if critical_failures:
            return True, f"Critical failures detected: {', '.join(critical_failures)}"

        # Razón 2: Score ambiguo en Capa 2
        layer2_score = layer2_results.get("avg_score", 0.5)
        if 0.5 <= layer2_score <= 0.75:
            return True, f"Ambiguous Layer 2 score: {layer2_score:.2f}"

        # Razón 3: Discrepancia entre Capa 1 y Capa 2
        layer1_score = self.scorer._compute_layer1_score(layer1_results)
        if abs(layer1_score - layer2_score) > 0.3:
            return (
                True,
                f"Layer 1-2 discrepancy: {abs(layer1_score - layer2_score):.2f}",
            )

        # Razón 4: Fallos específicos en agentes
        agent_fidelities = layer2_results.get("agent_fidelities", [])
        low_fidelity_agents = [
            af.get("agent", "unknown")
            for af in agent_fidelities
            if not af.get("pass", True)
        ]
        if low_fidelity_agents:
            return True, f"Low fidelity agents: {', '.join(low_fidelity_agents)}"

        # No se necesita Capa 3
        return False, None

    def get_summary(self, evaluation_result: Dict[str, Any]) -> str:
        """
        Generar resumen ejecutivo de la evaluación

        Args:
            evaluation_result: Resultado de evaluate()

        Returns:
            String con resumen ejecutivo
        """
        final_score = evaluation_result["final_score"]
        quality = evaluation_result["quality_label"]
        confidence = evaluation_result["confidence"]
        layer3_used = evaluation_result["layer3_used"]
        total_time = evaluation_result["evaluation_time"]

        summary = f"Evaluación {quality} (score: {final_score:.2f}/1.00, confianza: {confidence})\n"
        summary += f"Tiempo de evaluación: {total_time:.3f}s\n"

        if layer3_used:
            reason = evaluation_result.get("escalation_reason", "unknown")
            summary += f"Escalado a Capa 3: {reason}\n"
        else:
            summary += "Evaluación completada en Capas 1-2 (rápido)\n"

        # Fallos críticos
        critical = evaluation_result.get("critical_failures", [])
        if critical:
            summary += f" [WARNING] Fallos críticos detectados: {', '.join(critical)}\n"

        return summary
