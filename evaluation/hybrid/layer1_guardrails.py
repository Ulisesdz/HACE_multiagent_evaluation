"""
Layer 1: Guardrails - Validadores Deterministas
Propósito: Detectar fallos obvios sin procesamiento semántico
"""

import os
import re
from typing import Dict, List, Any


class GuardrailsValidator:
    """
    Capa 1: Validaciones deterministas rápidas

    Ejecuta validadores estructurales, numéricos y de formato
    para detectar fallos obvios sin procesamiento semántico.
    """

    def __init__(self):
        """Inicializa lista de validadores disponibles"""
        self.validators = [
            self.validate_completeness,
            self.validate_routing_syntax,
            self.validate_numeric_ranges,
            self.validate_chart_mentions,
            self.validate_task_agent_mapping,
        ]

    def validate_all(self, trace_data: Dict[str, Any]) -> Dict[str, Dict]:
        """
        Ejecutar todos los validadores

        Args:
            trace_data: Dict con trazas del sistema (user_question, planner_tasks,
                       routing_trace, agent_executions, final_answer)

        Returns:
            Dict {validator_name: result_dict} con resultados de cada validador
        """
        results = {}

        for validator in self.validators:
            validator_name = validator.__name__
            try:
                results[validator_name] = validator(trace_data)
            except Exception as e:
                results[validator_name] = {"pass": False, "error": str(e)}

        return results

    # STRUCTURAL VALIDATORS
    def validate_completeness(self, trace_data: Dict) -> Dict:
        """
        Validación: ¿Todas las tareas planificadas tienen ejecución?

        Args:
            trace_data: Trazas del sistema

        Returns:
            Dict con 'pass' (bool), 'missing_tasks' (list), 'coverage_rate' (float)
        """
        planned_tasks = trace_data.get("planner_tasks", [])
        agent_executions = trace_data.get("agent_executions", [])

        if not planned_tasks:
            return {"pass": True, "reason": "No tasks planned"}

        # Tareas ejecutadas (aproximación fuzzy)
        executed_tasks = set()
        for execution in agent_executions:
            task = execution.get("task", "").lower()
            if task:
                executed_tasks.add(task)

        # Verificar que cada tarea planificada tenga match
        missing_tasks = []
        for planned in planned_tasks:
            planned_lower = planned.lower()

            # Fuzzy match: si alguna tarea ejecutada contiene palabras clave
            words_planned = set(planned_lower.split())

            matched = False
            for executed in executed_tasks:
                words_executed = set(executed.split())
                # Si comparten >50% de palabras, considerar match
                overlap = len(words_planned & words_executed)
                if overlap / len(words_planned) > 0.5:
                    matched = True
                    break

            if not matched:
                missing_tasks.append(planned)

        coverage_rate = 1.0 - (len(missing_tasks) / len(planned_tasks))

        return {
            "pass": len(missing_tasks) == 0,
            "missing_tasks": missing_tasks,
            "coverage_rate": coverage_rate,
            "planned_count": len(planned_tasks),
            "executed_count": len(agent_executions),
        }

    def validate_routing_syntax(self, trace_data: Dict) -> Dict:
        """
        Validación: ¿Todas las decisiones de routing son válidas?

        Args:
            trace_data: Trazas del sistema

        Returns:
            Dict con 'pass' (bool), 'invalid_routings' (list)
        """
        routing_trace = trace_data.get("routing_trace", [])
        valid_agents = {
            "Technical_Analyst",
            "Fundamental_Analyst",
            "Risk_Officer",
            "FINISH",
        }

        invalid_routings = []

        for i, decision in enumerate(routing_trace):
            agent = decision.get("agent", "")
            if agent not in valid_agents:
                invalid_routings.append(
                    {
                        "index": i,
                        "task": decision.get("task", ""),
                        "invalid_agent": agent,
                    }
                )

        return {
            "pass": len(invalid_routings) == 0,
            "invalid_routings": invalid_routings,
            "total_decisions": len(routing_trace),
        }

    # NUMERIC VALIDATORS
    def validate_numeric_ranges(self, trace_data: Dict) -> Dict:
        """
        Validación: ¿Los valores numéricos en las respuestas de los agentes son plausibles?

        Args:
            trace_data: Trazas del sistema

        Returns:
            Dict con 'pass' (bool), 'anomalies' (list)
        """
        agent_executions = trace_data.get("agent_executions", [])

        anomalies = []

        for execution in agent_executions:
            agent = execution.get("agent", "")
            response = execution.get("agent_response", "")

            # Extraer números
            numbers = self._extract_numbers(response)

            if agent == "Technical_Analyst":
                for num in numbers:
                    # Ignorar ceros y valores < 0.01
                    # Son coordenadas, timestamps, etc.
                    if num == 0.0 or abs(num) < 0.01:
                        continue
                    if num > 200_000:
                        anomalies.append(
                            {
                                "agent": agent,
                                "value": num,
                                "reason": "Price exceeds maximum plausible value ($200k)",
                            }
                        )
                    elif num < 0:
                        # Solo marcar negativos, no los cercanos a cero
                        anomalies.append(
                            {
                                "agent": agent,
                                "value": num,
                                "reason": "Price cannot be negative",
                            }
                        )

            elif agent == "Risk_Officer":
                for num in numbers:
                    if num == 0.0:
                        continue  # 0% de volatilidad es raro pero no imposible como output intermedio
                    if num > 100:
                        anomalies.append(
                            {
                                "agent": agent,
                                "value": num,
                                "reason": "Volatility exceeds 100%",
                            }
                        )
                    elif num < 0:
                        anomalies.append(
                            {
                                "agent": agent,
                                "value": num,
                                "reason": "Volatility cannot be negative",
                            }
                        )

        return {
            "pass": len(anomalies) == 0,
            "anomalies": anomalies,
            "total_numbers_checked": sum(
                len(self._extract_numbers(ex.get("agent_response", "")))
                for ex in agent_executions
            ),
        }

    # FORMAT VALIDATORS
    def validate_chart_mentions(self, trace_data: Dict) -> Dict:
        """
        Validación: Si se menciona un gráfico, ¿existe el archivo?

        Args:
            trace_data: Trazas del sistema

        Returns:
            Dict con 'pass' (bool), 'phantom_charts' (list de rutas no existentes)
        """
        final_answer = trace_data.get("final_answer", "")

        # Buscar menciones de archivos .png
        mentioned_charts = re.findall(r"plots_temp/[\w\-\.]+\.png", final_answer)

        phantom_charts = []
        for chart_path in mentioned_charts:
            if not os.path.exists(chart_path):
                phantom_charts.append(chart_path)

        return {
            "pass": len(phantom_charts) == 0,
            "phantom_charts": phantom_charts,
            "mentioned_charts": mentioned_charts,
        }

    def validate_task_agent_mapping(self, trace_data: Dict) -> Dict:
        """
        Validación: ¿Hay tareas sin agente asignado? (indicador de routing roto)

        Args:
            trace_data: Trazas del sistema

        Returns:
            Dict con 'pass' (bool), 'unrouted_tasks' (list), 'routing_coverage' (float)
        """
        planner_tasks = trace_data.get("planner_tasks", [])
        routing_trace = trace_data.get("routing_trace", [])

        if not planner_tasks:
            return {"pass": True, "reason": "No tasks to route"}

        # Tareas que fueron enrutadas
        routed_tasks = set(r.get("task", "").lower() for r in routing_trace)

        # Tareas sin routing
        unrouted = []
        for task in planner_tasks:
            task_lower = task.lower()

            # Fuzzy match con tareas enrutadas
            matched = False
            for routed in routed_tasks:
                # Si comparten >60% de palabras, considerar match
                words_task = set(task_lower.split())
                words_routed = set(routed.split())

                if not words_task:
                    continue

                overlap = len(words_task & words_routed)
                if overlap / len(words_task) > 0.6:
                    matched = True
                    break

            if not matched:
                unrouted.append(task)

        return {
            "pass": len(unrouted) == 0,
            "unrouted_tasks": unrouted,
            "routing_coverage": (
                1.0 - (len(unrouted) / len(planner_tasks)) if planner_tasks else 1.0
            ),
        }

    # HELPERS
    def _extract_numbers(self, text: str) -> List[float]:
        """
        Extraer números de texto usando regex

        Args:
            text: String de texto

        Returns:
            Lista de números (float)
        """
        pattern = r"-?\d+(?:[.,]\d+)?"
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                # Normalizar (coma a punto)
                num_str = match.replace(",", ".")
                numbers.append(float(num_str))
            except ValueError:
                continue

        return numbers

    def get_critical_failures(self, results: Dict[str, Dict]) -> List[str]:
        """
        Identificar fallos críticos en los resultados

        Args:
            results: Dict con resultados de validate_all()

        Returns:
            Lista de strings describiendo fallos críticos
        """
        critical = []

        # Completitud <50% es crítico
        if not results.get("validate_completeness", {}).get("pass", True):
            coverage = results["validate_completeness"].get("coverage_rate", 0)
            if coverage < 0.5:
                critical.append("Critical incompleteness (coverage < 50%)")

        # Routing syntax inválido es crítico
        if not results.get("validate_routing_syntax", {}).get("pass", True):
            critical.append("Invalid routing syntax detected")

        # Anomalías numéricas son críticas
        if not results.get("validate_numeric_ranges", {}).get("pass", True):
            anomalies = results["validate_numeric_ranges"].get("anomalies", [])
            if anomalies:
                critical.append(f"{len(anomalies)} numeric anomalies detected")

        # Gráficos fantasma son críticos
        if not results.get("validate_chart_mentions", {}).get("pass", True):
            phantom = results["validate_chart_mentions"].get("phantom_charts", [])
            if phantom:
                critical.append(f"{len(phantom)} phantom charts mentioned")

        return critical
