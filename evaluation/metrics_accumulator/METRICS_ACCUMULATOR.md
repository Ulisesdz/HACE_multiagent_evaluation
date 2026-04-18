# Accumulated Metrics - Sistema de Acumulación de Evaluaciones

## Descripción

Este directorio contiene las evaluaciones acumuladas del sistema multi-agente, tanto online (Streamlit) como offline (batch).

**Versión:** 2.0  
**Fecha de creación:** Marzo 2026  
**Propósito:** Acumular métricas para análisis histórico, detección de drift, y fine-tuning futuro.

---

## Archivos

### `online_metrics.csv`
Evaluaciones generadas durante sesiones de usuario en Streamlit.

**Fuente:** `app.py` → `MetricsLogger.log_online_evaluation()`

**Generación:** Automática cada vez que un usuario evalúa una consulta en la interfaz web.

### `offline_metrics.csv`
Evaluaciones batch sobre el dataset de pruebas (45 casos).

**Fuente:** 
- `evaluation/baseline/run_eval.py`
- `evaluation/llm_j/run_eval.py`
- `evaluation/hybrid/run_eval.py`

**Generación:** Manual ejecutando los scripts de batch evaluation.

---

## Estructura de Datos

### Campos Comunes (Metadata)

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `timestamp` | ISO-8601 | Fecha/hora de la evaluación | `2026-03-01T14:30:00` |
| `source` | `online` \| `offline` | Origen de la evaluación | `online` |
| `query_id` | string | Identificador único | `online_20260301_143000` |
| `user_query` | string | Pregunta del usuario | `Precio de Bitcoin` |
| `num_tasks` | int | Número de tareas planificadas | `2` |
| `difficulty` | string | Easy, Medium, Hard, Very Hard | `Medium` (solo offline) |
| `category` | string | Categoría del caso de prueba | `Price_Query` (solo offline) |

### Baseline Metrics (Escala 0-1)

| Campo | Descripción | Rango |
|-------|-------------|-------|
| `baseline_score` | Score global ponderado | 0-1 |
| `baseline_routing_f1` | Precisión del routing (F1-Score) | 0-1 |
| `baseline_numeric_f1` | Fidelidad numérica (F1-Score) | 0-1 |
| `baseline_hallucination_rate` | Tasa de alucinaciones numéricas | 0-1 |
| `baseline_task_coverage` | Cobertura de tareas completadas | 0-1 |
| `baseline_sql_correctness` | Correctness de queries SQL | 0-1 |
| `baseline_time` | Tiempo de evaluación (segundos) | >0 |

### LLM-Judge Metrics (Escala 1-4)

**NOTA:** A partir de la versión 2.0, LLM-Judge usa escala **1-4** (antes 0-10).

| Campo | Descripción | Rango |
|-------|-------------|-------|
| `llm_judge_overall` | Score global ponderado | 1-4 |
| `llm_judge_planner` | Score del Planner | 1-4 |
| `llm_judge_supervisor` | Score del Supervisor | 1-4 |
| `llm_judge_agents` | Score promedio de agentes | 1-4 |
| `llm_judge_final` | Score del output final | 1-4 |
| `llm_judge_error_category` | Categoría de error detectada | `None`, `Planning_Error`, etc. |
| `llm_judge_time` | Tiempo de evaluación (segundos) | >0 |

**Interpretación de scores:**
- **4**: Perfecto
- **3**: Bueno
- **2**: Mejorable
- **1**: Crítico

### HACE Metrics (Hybrid - Escala 0-1)

**HACE** (Hybrid Agent Consensus Evaluator) combina validación determinista, evaluación semántica con embeddings, y LLM-Judge selectivo en una arquitectura de 3 capas.

| Campo | Descripción | Rango |
|-------|-------------|-------|
| `HACE_score` | Score global híbrido final | 0-1 |
| `HACE_quality` | Label cualitativo | `Excelente`, `Bueno`, `Mejorable`, `Crítico` |
| `HACE_confidence` | Nivel de confianza en evaluación | `high`, `medium`, `low` |
| `HACE_layer1` | Score Layer 1 (Guardrails) | 0-1 |
| `HACE_layer2` | Score Layer 2 (Semantic) | 0-1 |
| `HACE_layer3` | Score Layer 3 (LLM-Judge) | 0-1 (null si no se usó) |
| `HACE_layer3_used` | ¿Se escaló a Layer 3? | 0 (No) o 1 (Sí) |
| `HACE_time` | Tiempo total de evaluación | >0 |

**Componentes de HACE:**

1. **Layer 1 (Guardrails - ~0.05s):** Validadores deterministas
   - Completitud estructural
   - Sintaxis de routing
   - Rangos numéricos plausibles
   - Existencia de archivos mencionados

2. **Layer 2 (Semantic - ~0.5s):** Evaluación semántica con embeddings
   - Task Fidelity (BERTScore-inspired)
   - Agent Fidelity (similitud tool output vs respuesta)
   - Routing Quality (keywords ponderados)
   - Report Completeness

3. **Layer 3 (LLM-Judge Selectivo - ~3.5s):** Solo se ejecuta en ~40% de casos
   - Evaluación profunda de módulos problemáticos detectados en Layer 1-2
   - Usa el mismo LLM-Judge de evaluación cualitativa
   - Escalación basada en fallos críticos, scores ambiguos o discrepancias

**Ventajas vs sistemas individuales:**
- Cobertura semántica completa (vs Baseline que solo valida numérico/estructural)
- Menor costo (menos llamadas a LLM que LLM-Judge puro)

**Papers base:**
- BERTScore (Zhang et al., 2019) - Similitud semántica
- ARES (Saad-Falcon et al., 2023) - Clasificadores binarios para RAG
- Cascading LLMs (Chen et al., 2023) - Routing adaptativo
- Prometheus (Kim et al., 2023) - Rubric-guided evaluation

---

### Metadata Adicional

| Campo | Descripción | Formato |
|-------|-------------|---------|
| `critical_failures` | JSON con fallos críticos detectados | `["Hallucination detected", ...]` |
| `raw_trace` | JSON con traza completa del sistema | `{"planner_tasks": [...], ...}` |

---

### Datos AlHACEnados

✅ **SE ALHACENA:**
- Pregunta del usuario (`user_query`)
- Respuestas del sistema
- Métricas de evaluación
- Trazas de ejecución (herramientas usadas, routing)

❌ **NO SE ALHACENA:**
- Información personal identificable (PII)
- Tokens de sesión
- Credenciales
- Datos sensibles del usuario