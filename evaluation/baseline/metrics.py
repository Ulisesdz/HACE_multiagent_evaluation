import re
from typing import List, Dict

class BaselineMetricsCalculator:
    """
    Calculador de métricas automáticas deterministas.
    
    Basado en:
    - Precision/Recall para clasificación (Routing)
    - SelfCheckGPT (2023) para detección de alucinaciones
    - Pattern Matching para validación SQL
    """
    
    # MÉTRICA 1: Routing Accuracy
    def evaluate_routing_accuracy(
        self, 
        routing_trace: List[Dict]
    ) -> Dict[str, any]:
        """
        Evalúa precisión del routing.
        Solo calcula métricas para agentes que participaron.
        
        Args:
            routing_trace: [
                {'task': 'Obtener precio BTC', 'agent': 'Technical_Analyst'},
                ...
            ]
        
        Returns:
            Dict con accuracy, precision, recall, f1, confusion_matrix
        """
        # Determinar agente esperado para cada tarea
        for trace in routing_trace:
            trace['expected'] = self._infer_expected_agent(trace['task'])
        
        correct = 0
        total = len(routing_trace)
        
        # Confusion matrix SOLO para agentes que participaron
        agents_involved = set()
        for trace in routing_trace:
            agents_involved.add(trace['agent'])
            agents_involved.add(trace['expected'])
        
        confusion = {agent: {'TP': 0, 'FP': 0, 'FN': 0} for agent in agents_involved}
        
        for trace in routing_trace:
            actual = trace['agent']
            expected = trace['expected']
            
            if actual == expected:
                correct += 1
                confusion[actual]['TP'] += 1
            else:
                confusion[actual]['FP'] += 1
                confusion[expected]['FN'] += 1
        
        # Calcular métricas SOLO para agentes involucrados
        metrics_per_class = {}
        for agent in confusion:
            tp = confusion[agent]['TP']
            fp = confusion[agent]['FP']
            fn = confusion[agent]['FN']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_per_class[agent] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Métricas globales (macro-average SOLO de agentes involucrados)
        accuracy = correct / total if total > 0 else 0
        
        num_agents = len(metrics_per_class)
        avg_precision = sum(m['precision'] for m in metrics_per_class.values()) / num_agents if num_agents > 0 else 0
        avg_recall = sum(m['recall'] for m in metrics_per_class.values()) / num_agents if num_agents > 0 else 0
        avg_f1 = sum(m['f1'] for m in metrics_per_class.values()) / num_agents if num_agents > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'per_class': metrics_per_class,
            'confusion_matrix': confusion
        }
    
    def _infer_expected_agent(self, task: str) -> str:
        """Infiere qué agente debería ejecutar la tarea"""
        task_lower = task.lower()
        
        # Reglas de routing (mismas que en SUPERVISOR_ROUTER_PROMPT)
        technical_keywords = [
            'precio', 'gráfico', 'gráfica', 'top', 'histórico', 
            'predicción', 'chart', 'cierre', 'máximo', 'mínimo', 'obtener'
        ]
        
        fundamental_keywords = [
            'noticia', 'explicar', 'qué es', 'concepto', 'definición',
            'por qué', 'contexto', 'halving', 'blockchain', 'buscar', 'investigar'
        ]
        
        risk_keywords = [
            'volatilidad', 'riesgo', 'seguro', 'peligro', 'calcular'
        ]
        
        # Prioridad: Conceptos > Noticias > Riesgo > Técnico
        if any(kw in task_lower for kw in fundamental_keywords):
            return 'Fundamental_Analyst'
        elif any(kw in task_lower for kw in risk_keywords):
            return 'Risk_Officer'
        elif any(kw in task_lower for kw in technical_keywords):
            return 'Technical_Analyst'
        
        return 'Technical_Analyst'  # Default
    
    def _empty_routing_metrics(self) -> Dict:
        """Devuelve métricas vacías si no hay datos"""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'per_class': {},
            'confusion_matrix': {}
        }
    
    # MÉTRICA 2: Numeric Accuracy (Hallucination Detection)
    def evaluate_numeric_accuracy(
        self,
        tool_outputs: List[str],
        agent_responses: List[str]
    ) -> Dict[str, float]:
        """
        Detecta alucinaciones numéricas.
        
        Args:
            tool_outputs: Lista de outputs de herramientas
            agent_responses: Lista de respuestas de agentes
        
        Returns:
            Dict con precision, recall, f1, hallucination_rate
        """
        if not tool_outputs or not agent_responses:
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'hallucination_rate': 0.0
            }
        
        total_precision = []
        total_recall = []
        hallucinations = []
        
        for tool_out, agent_resp in zip(tool_outputs, agent_responses):
            tool_numbers = self._extract_numbers(tool_out)
            agent_numbers = self._extract_numbers(agent_resp)
            
            if not tool_numbers:
                continue  # Skip si no hay números en tool
            
            # True Positives: números del agente que están en tool
            tp = len([n for n in agent_numbers if self._is_in_tolerance(n, tool_numbers)])
            
            # False Positives: números del agente que NO están en tool
            fp = len([n for n in agent_numbers if not self._is_in_tolerance(n, tool_numbers)])
            
            # False Negatives: números del tool que NO reportó el agente
            fn = len([n for n in tool_numbers if not self._is_in_tolerance(n, agent_numbers)])
            
            # Métricas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            hallucination_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            total_precision.append(precision)
            total_recall.append(recall)
            hallucinations.append(hallucination_rate)
        
        avg_precision = sum(total_precision) / len(total_precision) if total_precision else 1.0
        avg_recall = sum(total_recall) / len(total_recall) if total_recall else 1.0
        avg_hallucination = sum(hallucinations) / len(hallucinations) if hallucinations else 0.0
        
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': f1,
            'hallucination_rate': avg_hallucination
        }
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extrae todos los números de un texto"""
        if not text:
            return []
        
        pattern = r'-?\d+(?:[.,]\d+)?'
        matches = re.findall(pattern, text)
        numbers = []
        
        for match in matches:
            try:
                normalized = match.replace(',', '.')
                numbers.append(float(normalized))
            except ValueError:
                continue
        
        return numbers
    
    def _is_in_tolerance(self, number: float, number_list: List[float], tolerance=0.01) -> bool:
        """Verifica si un número está en una lista con tolerancia del 1%"""
        for n in number_list:
            if abs(n) < 0.001:  # Evitar división por cero
                if abs(number - n) < 0.001:
                    return True
            elif abs(number - n) / max(abs(n), 1) < tolerance:
                return True
        return False
    
    # MÉTRICA 3: Task Coverage
    def evaluate_task_coverage(
        self,
        planned_tasks: List[str],
        completed_tasks: List[str]
    ) -> Dict[str, any]:
        """
        Evalúa completitud de tareas. Mejor fuzzy matching y detección de tareas vacías.
        
        Returns:
            Dict con coverage, omission_rate, planned_tasks, completed_tasks
        """
        if not planned_tasks:
            return {
                'coverage': 1.0,
                'omission_rate': 0.0,
                'planned_tasks': 0,
                'completed_tasks': 0
            }
        
        total_tasks = len(planned_tasks)
        
        # Filtrar completed_tasks vacíos o None
        completed_tasks_clean = [t for t in completed_tasks if t and isinstance(t, str) and t.strip()]
        
        # Si no hay completed_tasks pero hubo agent_executions, asumir que se completaron
        if not completed_tasks_clean and total_tasks > 0:
            # Si hay planned_tasks pero no completed_tasks, es probable un bug de captura
            # En este caso, se asume coverage basado en si routing_trace tiene entries
            matched = 0  # Lo dejo en 0 para forzar a que se capture correctamente
        else:
            # Match fuzzy normal
            matched = 0
            for planned in planned_tasks:
                for completed in completed_tasks_clean:
                    if self._fuzzy_match(planned, completed):
                        matched += 1
                        break
        
        coverage = matched / total_tasks if total_tasks > 0 else 0.0
        omission_rate = 1.0 - coverage
        
        return {
            'coverage': coverage,
            'omission_rate': omission_rate,
            'planned_tasks': total_tasks,
            'completed_tasks': matched
        }
    
    def _fuzzy_match(self, text1: str, text2: str, threshold=0.4) -> bool:
        """Match fuzzy entre dos strings usando Jaccard similarity"""
        if not text1 or not text2:
            return False
        
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    # MÉTRICA 4: SQL Correctness
    def evaluate_sql_correctness(
        self,
        sql_queries: List[Dict[str, str]]
    ) -> Dict[str, any]:
        """
        Valida corrección de queries SQL.
        
        Args:
            sql_queries: [
                {'task': 'Obtener Top 3 precios BTC', 'sql': 'SELECT ...'},
                ...
            ]
        
        Returns:
            Dict con correctness, violations, total_queries, correct_queries
        Maneja casos donde sql es dict, None o vacío.
        """
        if not sql_queries:
            return {
                'correctness': 1.0,
                'total_queries': 0,
                'correct_queries': 0,
                'violations': []
            }
        
        total_queries = len(sql_queries)
        violations = []
        
        for query_entry in sql_queries:
            task = query_entry.get('task', '')
            sql = query_entry.get('sql', '')
            
            # Validación de tipo de SQL
            # Caso 1: sql es None
            if sql is None:
                violations.append(f"[{task}] SQL is None (tool error or not captured)")
                continue
            
            # Caso 2: sql es un dict (error de invocación de tool)
            if isinstance(sql, dict):
                violations.append(f"[{task}] SQL is dict instead of string: {sql}")
                continue
            
            # Caso 3: sql es string vacío
            if not isinstance(sql, str) or not sql.strip():
                violations.append(f"[{task}] SQL is empty string")
                continue
            
            # Validación normal (solo si sql es string válido)
            query_violations = self._validate_sql_patterns(task, sql)
            
            if query_violations:
                for violation in query_violations:
                    violations.append(f"[{task}] {violation}")
        
        # Calcular correctness
        correct_queries = total_queries - len(violations)
        correctness = correct_queries / total_queries if total_queries > 0 else 1.0
        
        return {
            'correctness': correctness,
            'total_queries': total_queries,
            'correct_queries': correct_queries,
            'violations': violations
        }
    
    def _validate_sql_patterns(self, task: str, sql: str) -> List[str]:
        """Valida patrones SQL según el tipo de tarea"""
        violations = []
        sql_upper = sql.upper()
        task_lower = task.lower()
        
        # Regla 1: Top X debe usar ORDER BY + LIMIT, NO WHERE con fecha
        if any(kw in task_lower for kw in ['top', 'más alto', 'más bajo', 'máximo', 'mínimo']):
            if 'ORDER BY' not in sql_upper:
                violations.append("Missing ORDER BY for Top X query")
            if 'LIMIT' not in sql_upper:
                violations.append("Missing LIMIT for Top X query")
            if 'WHERE' in sql_upper and 'DATE' in sql_upper:
                violations.append("Top X query should not filter by specific date")
        
        # Regla 2: "Últimos N" debe usar ORDER BY Date DESC
        if any(kw in task_lower for kw in ['último', 'últimos', 'reciente', 'recientes']):
            if 'ORDER BY' not in sql_upper:
                violations.append("Missing ORDER BY for recent data query")
            if 'DESC' not in sql_upper:
                violations.append("Missing DESC for recent data query")
        
        return violations