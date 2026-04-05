"""
Layer 2: Semantic Evaluators - Evaluación semántica rápida
Propósito: Evaluación semántica sin LLM completo
"""

import numpy as np
from typing import Dict, List, Any
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticEvaluator:
    """
    Capa 2: Evaluación semántica con embeddings y heurísticas
    """

    def __init__(self):
        """
        Inicializa el evaluador semántico

        Intenta cargar modelo de embeddings. Si falla, usa solo heurísticas.
        """
        try:
            # Modelo ligero
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.cosine_similarity = cosine_similarity
            self.available = True
        except ImportError:
            print(
                "   [WARNING] sentence-transformers no instalado. Capa 2 usará solo heurísticas."
            )
            print("   Instalar con: pip install sentence-transformers")
            self.available = False

    def evaluate_all(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecutar todas las evaluaciones semánticas

        Args:
            trace_data: Dict con trazas del sistema

        Returns:
            Dict con resultados de cada evaluador y avg_score global
        """
        results = {}

        # 1. Task Fidelity (Planner)
        results["task_fidelity"] = self.evaluate_task_fidelity(
            trace_data.get("user_question", ""), trace_data.get("planner_tasks", [])
        )

        # 2. Agent Fidelity (Tool Output vs Agent Response)
        agent_fidelities = []
        for execution in trace_data.get("agent_executions", []):
            fidelity = self.evaluate_agent_fidelity(
                execution.get("tool_outputs", ""), execution.get("agent_response", "")
            )
            fidelity["agent"] = execution.get("agent", "unknown")
            agent_fidelities.append(fidelity)

        results["agent_fidelities"] = agent_fidelities

        # 3. Routing Quality (Heuristic)
        results["routing_quality"] = self.evaluate_routing_keywords(
            trace_data.get("routing_trace", [])
        )

        # 4. Final Report Completeness (Heuristic)
        results["report_completeness"] = self.evaluate_report_completeness(
            trace_data.get("planner_tasks", []), trace_data.get("final_answer", "")
        )

        # Calcular score promedio
        scores = []

        if results["task_fidelity"]["pass"]:
            scores.append(1.0)
        else:
            scores.append(results["task_fidelity"].get("avg_similarity", 0.5))

        for af in agent_fidelities:
            scores.append(af.get("score", 0.5))

        if results["routing_quality"]["pass"]:
            scores.append(results["routing_quality"]["confidence"])
        else:
            scores.append(0.5)

        if results["report_completeness"]["pass"]:
            scores.append(1.0)
        else:
            scores.append(results["report_completeness"]["coverage"])

        results["avg_score"] = np.mean(scores) if scores else 0.5
        results["evaluation_method"] = (
            "embeddings" if self.available else "heuristics_only"
        )

        return results

    # EMBEDDING-BASED EVALUATORS
    def evaluate_task_fidelity(self, user_query: str, planner_tasks: List[str]) -> Dict:
        """
        ¿Las tareas del Planner son semánticamente fieles a la query?
        Basado en BERTScore (Zhang et al., 2019)

        Args:
            user_query: Pregunta del usuario
            planner_tasks: Lista de tareas generadas

        Returns:
            Dict con 'pass' (bool), 'avg_similarity', 'min_similarity', 'low_similarity_tasks'
        """
        if not user_query or not planner_tasks:
            return {"pass": True, "reason": "No tasks or query"}

        if not self.available:
            # Fallback: heurística básica
            return self._task_fidelity_heuristic(user_query, planner_tasks)

        try:
            # Encode user query
            query_embedding = self.encoder.encode([user_query])

            # Encode cada tarea
            task_embeddings = self.encoder.encode(planner_tasks)

            # Similitud coseno
            similarities = self.cosine_similarity(query_embedding, task_embeddings)[0]

            # Métricas
            avg_similarity = float(np.mean(similarities))
            min_similarity = float(np.min(similarities))

            # Threshold empírico: 0.5
            low_similarity_tasks = [
                planner_tasks[i] for i, sim in enumerate(similarities) if sim < 0.5
            ]

            return {
                "pass": min_similarity > 0.5,
                "avg_similarity": avg_similarity,
                "min_similarity": min_similarity,
                "low_similarity_tasks": low_similarity_tasks,
                "method": "embedding",
            }

        except Exception as e:
            return {"pass": True, "error": str(e), "method": "fallback"}

    def evaluate_agent_fidelity(self, tool_output: str, agent_response: str) -> Dict:
        """
        ¿La respuesta del agente es fiel al tool output?
        Similar a BERTScore pero adaptado

        Args:
            tool_output: Salida de la herramienta
            agent_response: Respuesta del agente

        Returns:
            Dict con 'pass' (bool), 'precision', 'score'
        """
        if not tool_output or not agent_response:
            return {"pass": True, "score": 1.0, "reason": "N/A"}

        if not self.available:
            # Fallback: verificar que números del agente estén en tool
            return self._agent_fidelity_heuristic(tool_output, agent_response)

        try:
            # Tokenizar en oraciones
            tool_sentences = self._split_sentences(tool_output)
            agent_sentences = self._split_sentences(agent_response)

            if not tool_sentences or not agent_sentences:
                return {"pass": True, "score": 1.0, "reason": "No sentences"}

            # Encode
            tool_embs = self.encoder.encode(tool_sentences)
            agent_embs = self.encoder.encode(agent_sentences)

            # Para cada oración del agente, buscar la más similar en tool
            max_similarities = []
            for agent_emb in agent_embs:
                sims = self.cosine_similarity([agent_emb], tool_embs)[0]
                max_similarities.append(float(np.max(sims)))

            # Precision: ¿Qué % de lo que dijo el agente está en el tool?
            precision = float(np.mean(max_similarities))

            return {
                "pass": precision > 0.7,
                "precision": precision,
                "score": precision,
                "method": "embedding",
            }

        except Exception as e:
            return {"pass": True, "score": 0.8, "error": str(e), "method": "fallback"}

    # HEURISTIC EVALUATORS (Fallback cuando no hay embeddings)
    def _task_fidelity_heuristic(
        self, user_query: str, planner_tasks: List[str]
    ) -> Dict:
        """
        Heurística de fidelidad usando Jaccard similarity
        Fallback cuando embeddings no disponibles

        Args:
            user_query: Pregunta del usuario
            planner_tasks: Lista de tareas

        Returns:
            Dict con 'pass', 'avg_similarity', 'method'
        """
        query_words = set(user_query.lower().split())

        similarities = []
        for task in planner_tasks:
            task_words = set(task.lower().split())

            if not task_words:
                similarities.append(0.0)
                continue

            # Jaccard similarity
            intersection = len(query_words & task_words)
            union = len(query_words | task_words)

            sim = intersection / union if union > 0 else 0.0
            similarities.append(sim)

        avg_sim = np.mean(similarities) if similarities else 0.0

        return {
            "pass": avg_sim > 0.3,
            "avg_similarity": float(avg_sim),
            "method": "heuristic",
        }

    def _agent_fidelity_heuristic(self, tool_output: str, agent_response: str) -> Dict:
        """
        Heurística de fidelidad verificando números en común
        Fallback cuando embeddings no disponibles

        Args:
            tool_output: Salida de herramienta
            agent_response: Respuesta del agente

        Returns:
            Dict con 'pass', 'precision', 'score', 'method'
        """
        tool_numbers = set(self._extract_numbers(tool_output))
        agent_numbers = self._extract_numbers(agent_response)

        if not agent_numbers:
            return {"pass": True, "score": 1.0, "reason": "No numbers in response"}

        # Contar cuántos números del agente están en tool (con tolerancia)
        matches = 0
        for agent_num in agent_numbers:
            if self._number_in_set(agent_num, tool_numbers, tolerance=0.01):
                matches += 1

        precision = matches / len(agent_numbers) if agent_numbers else 1.0

        return {
            "pass": precision > 0.7,
            "precision": precision,
            "score": precision,
            "method": "heuristic",
        }

    def evaluate_routing_keywords(self, routing_trace: List[Dict]) -> Dict:
        """
        Validar routing usando keywords ponderados

        Args:
            routing_trace: Lista de decisiones de routing

        Returns:
            Dict con pass, accuracy, confidence, routing_details
        """
        keyword_map = {
            "Fundamental_Analyst": {
                "high": [
                    "qué es",
                    "explicar",
                    "concepto",
                    "definición",
                    "noticias",
                    "contexto",
                ],
                "medium": ["investigar", "buscar", "información"],
            },
            "Risk_Officer": {
                "high": ["riesgo", "volatilidad", "seguro", "peligro"],
                "medium": ["estable", "seguridad"],
            },
            "Technical_Analyst": {
                "high": ["precio", "gráfico", "predicción", "top", "histórico"],
                "medium": ["cierre", "datos", "número"],
            },
        }

        correct_routings = 0
        total_routings = 0
        routing_details = []

        for decision in routing_trace:
            task = decision.get("task", "").lower()
            assigned_agent = decision.get("agent", "")

            if not task or assigned_agent == "FINISH":
                continue

            total_routings += 1

            # Buscar keywords de alta prioridad
            expected_agent = None
            confidence = 0.5

            for agent, keywords in keyword_map.items():
                if any(kw in task for kw in keywords["high"]):
                    expected_agent = agent
                    confidence = 0.9
                    break
                elif any(kw in task for kw in keywords["medium"]):
                    expected_agent = agent
                    confidence = 0.6
                    break

            # Si no match claro, asumir correcto
            if expected_agent is None:
                correct_routings += 1
                routing_details.append(
                    {
                        "task": task,
                        "assigned": assigned_agent,
                        "correct": True,
                        "confidence": 0.5,
                        "reason": "ambiguous",
                    }
                )
            elif expected_agent == assigned_agent:
                correct_routings += 1
                routing_details.append(
                    {
                        "task": task,
                        "assigned": assigned_agent,
                        "correct": True,
                        "confidence": confidence,
                    }
                )
            else:
                routing_details.append(
                    {
                        "task": task,
                        "assigned": assigned_agent,
                        "expected": expected_agent,
                        "correct": False,
                        "confidence": confidence,
                    }
                )

        accuracy = correct_routings / total_routings if total_routings > 0 else 1.0
        avg_confidence = (
            np.mean([r["confidence"] for r in routing_details])
            if routing_details
            else 0.5
        )

        return {
            "pass": accuracy > 0.8,
            "accuracy": accuracy,
            "confidence": float(avg_confidence),
            "routing_details": routing_details,
            "correct_count": correct_routings,
            "total_count": total_routings,
        }

    def evaluate_report_completeness(
        self, planner_tasks: List[str], final_report: str
    ) -> Dict:
        """
        ¿El informe final menciona todas las tareas?
        """
        if not planner_tasks or not final_report:
            return {"pass": True, "reason": "No tasks or report"}

        report_lower = final_report.lower()

        # Extraer activos de las tareas
        crypto_keywords = [
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "solana",
            "sol",
            "cardano",
            "ada",
            "dogecoin",
            "doge",
            "xrp",
            "ripple",
            "binance",
            "bnb",
        ]

        mentioned_assets = set()
        for task in planner_tasks:
            task_lower = task.lower()
            for crypto in crypto_keywords:
                if crypto in task_lower:
                    mentioned_assets.add(crypto)

        # Verificar que cada activo esté en el reporte
        missing_assets = []
        for asset in mentioned_assets:
            if asset not in report_lower:
                missing_assets.append(asset)

        coverage = (
            1.0 - (len(missing_assets) / len(mentioned_assets))
            if mentioned_assets
            else 1.0
        )

        return {
            "pass": len(missing_assets) == 0,
            "coverage": coverage,
            "missing_assets": missing_assets,
            "mentioned_assets": list(mentioned_assets),
        }

    # HELPERS
    def _split_sentences(self, text: str) -> List[str]:
        """
        Dividir texto en oraciones usando regex simple

        Args:
            text: Texto a dividir

        Returns:
            Lista de oraciones
        """
        # Simple split por puntos
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _extract_numbers(self, text: str) -> List[float]:
        """
        Extraer números de texto (enteros y decimales)

        Args:
            text: Texto a analizar

        Returns:
            Lista de números encontrados
        """
        pattern = r"-?\d+(?:[.,]\d+)?"
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            try:
                num_str = match.replace(",", ".")
                numbers.append(float(num_str))
            except ValueError:
                continue

        return numbers

    def _number_in_set(
        self, number: float, number_set: set, tolerance: float = 0.01
    ) -> bool:
        """
        Verificar si número está en set con tolerancia relativa

        Args:
            number: Número a buscar
            number_set: Set de números
            tolerance: Tolerancia relativa (default 1%)

        Returns:
            True si encontrado dentro de tolerancia
        """
        for n in number_set:
            if abs(number - n) / max(abs(n), 1) < tolerance:
                return True
        return False
