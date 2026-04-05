# Sistema de Evaluación ComprLLM-J Multi-Agente

## Índice
1. [Arquitectura del Sistema](#arquitectura)
2. [Metodología de Evaluación](#metodología)
3. [Cobertura del Dataset](#dataset)
4. [Métricas y Scoring](#métricas)
5. [Interpretación de Resultados](#interpretación)
6. [Casos de Uso](#casos-de-uso)

---

## 1. Arquitectura del Sistema {#arquitectura}

### Pipeline de Ejecución
```
Usuario → Planner → Supervisor → Agente → Supervisor → ... → Informe Final
```

### Módulos Evaluados

#### 1.1 Planner (20% del score total)
**Responsabilidades:**
- Leer el mensaje del usuario
- Identificar todas las solicitudes
- Generar lista de tareas con precisión literal
- Separar acciones en tareas independientes

**Métricas Evaluadas (Escala 1-4):**
- `correctness` (1-4): ¿Identificó las tareas correctas?
  - **4**: Perfecto - Identificó todas las tareas correctamente
  - **3**: Bueno - Correcto con pequeñas imprecisiones
  - **2**: Mejorable - Errores significativos en identificación
  - **1**: Crítico - Falló completamente en identificar tareas
- `completeness` (1-4): ¿Capturó TODAS las solicitudes?
  - **4**: Perfecto - Capturó el 100% de las solicitudes
  - **3**: Bueno - Capturó >80% de las solicitudes
  - **2**: Mejorable - Capturó 50-80% de las solicitudes
  - **1**: Crítico - Omitió >50% de las solicitudes
- `precision` (1-4): ¿Mantuvo adjetivos, cantidades, fechas?
  - **4**: Perfecto - Preservó todos los detalles literalmente
  - **3**: Bueno - Preservó la mayoría de detalles importantes
  - **2**: Mejorable - Perdió algunos detalles importantes
  - **1**: Crítico - Generalizó o distorsionó los detalles
- `task_decomposition` (1-4): ¿Separó bien las acciones?
  - **4**: Perfecto - Separó correctamente todas las acciones
  - **3**: Bueno - Separación correcta con mínimas mezclas
  - **2**: Mejorable - Mezcló algunas tareas que deberían estar separadas
  - **1**: Crítico - Mezcló la mayoría de tareas

**Errores Detectables:**
- Omisión de tareas (incompleteness)
- Simplificación excesiva ("Top 3" → "precio")
- Mezcla de tareas múltiples en una sola
- Invención de tareas no solicitadas

---

#### 1.2 Supervisor (25% del score total)
**Responsabilidades:**
- Recibir cada tarea de la lista pendiente
- Seleccionar al agente especializado correcto
- Completar TODAS las tareas antes de FINISH
- Detectar loops (misma tarea → mismo agente sin resolver)

**Métricas Evaluadas:**
- `routing_accuracy` (1-4): Precisión de las decisiones de routing
  - **4**: Perfecto - 100% de decisiones correctas
  - **3**: Bueno - >80% de decisiones correctas
  - **2**: Mejorable - 50-80% de decisiones correctas
  - **1**: Crítico - <50% de decisiones correctas
- `task_completion` (1-4): ¿Procesó todas las tareas?
  - **4**: Perfecto - Procesó el 100% de las tareas
  - **3**: Bueno - Procesó >80% de las tareas
  - **2**: Mejorable - Procesó 50-80% de las tareas
  - **1**: Crítico - Omitió >50% de las tareas
- `routing_decisions`: Lista detallada de cada decisión

**Errores Detectables:**
- Routing incorrecto (precio → Risk Officer)
- Terminación prematura (FINISH con tareas pendientes)
- Loops infinitos

**Reglas de Routing Esperadas:**
| Tipo de Tarea | Agente Correcto |
|--------------|-----------------|
| Precio, Gráfico, Top X, Predicción | Technical_Analyst |
| Noticias, "Qué es X", Contexto | Fundamental_Analyst |
| Volatilidad, Riesgo, "Es seguro" | Risk_Officer |
| Fuera de dominio | FINISH |

---

#### 1.3 Agentes Especializados (40% del score total)

##### Technical_Analyst
**Herramientas:**
- `crypto_history_tool` (SQL): Precios, históricos, Top X
- `crypto_prediction_tool` (ML): Predicciones numéricas
- `crypto_chart_tool` (Matplotlib): Generación de gráficos

**Métricas (Escala 1-4):**
- `tool_selection` (1-4): ¿Eligió la herramienta correcta?
  - **4**: Perfecto - Herramienta ideal para la tarea
  - **3**: Bueno - Herramienta funciona pero hay alternativa mejor
  - **2**: Mejorable - Herramienta subóptima pero funcionó
  - **1**: Crítico - Herramienta incorrecta o no usó ninguna
- `tool_execution` (1-4): ¿SQL correcto, parámetros válidos?
  - **4**: Perfecto - Ejecución impecable
  - **3**: Bueno - Ejecución correcta con pequeñas ineficiencias
  - **2**: Mejorable - Errores en parámetros pero obtuvo resultado
  - **1**: Crítico - Falló la ejecución o errores graves
- `output_fidelity` (1-4): ¿Respuesta fiel a la evidencia técnica?
  - **4**: Perfecto - 100% basado en tool output
  - **3**: Bueno - Mayormente fiel con mínima interpretación
  - **2**: Mejorable - Agregó información no verificable
  - **1**: Crítico - Contradice o ignora tool output
- `output_completeness` (1-4): ¿Reportó TODOS los datos? (Top 3 = 3 items)
  - **4**: Perfecto - Reportó todos los datos solicitados
  - **3**: Bueno - Reportó >80% de los datos
  - **2**: Mejorable - Reportó 50-80% de los datos
  - **1**: Crítico - Omitió >50% de los datos o resumió incorrectamente
- `hallucination_check` (1-4): ¿Inventó datos?
  - **4**: Perfecto - Cero invención, todo verificable
  - **3**: Bueno - Mínima interpretación razonable
  - **2**: Mejorable - Algunos datos no verificables
  - **1**: Crítico - Inventó datos numéricos o hechos

**Reglas Críticas:**
- PRINCIPIO DE EXCLUSIVIDAD: Si ejecuta SQL, solo reporta SQL. Si ejecuta Gráfico, solo reporta gráfico.
- Top X debe listar X items completos, NO resumir.
- Gráficos: Solo reportar si hay ruta de archivo real.
- Predicciones: Debe de mostrarse el dato predicho por el modelo.

##### Fundamental_Analyst
**Herramientas:**
- `crypto_rag_tool`: Base de conocimiento interna
- `crypto_news_tool`: Búsqueda web de noticias

**Reglas Críticas:**
- STRICT RAG: Respuesta SOLO basada en contexto recuperado
- NO usar conocimiento paramétrico (memoria interna)
- Si RAG vacío: "La documentación interna no contiene esa información"
- Siempre citar fuente

##### Risk_Officer
**Herramientas:**
- `crypto_volatility_tool`: Cálculo de volatilidad histórica

**Reglas Críticas:**
- Si volatilidad ALTA (>5%): ADVERTIR explícitamente
- Si no hay datos: "Insuficientes datos históricos"
- NO inventar porcentajes
- Tono cauteloso y escéptico

---

#### 1.4 Informe Final (15% del score total)
**Responsabilidades del Supervisor:**
- Consolidar outputs de todos los agentes
- Organizar por activo/tarea
- Asociar gráficos al activo correcto
- Incluir disclaimer legal

**Métricas (Escala 1-4):**
- `completeness` (1-4): ¿Incluye TODAS las tareas?
  - **4**: Perfecto - Incluye 100% de las tareas completadas
  - **3**: Bueno - Incluye >80% de las tareas
  - **2**: Mejorable - Incluye 50-80% de las tareas
  - **1**: Crítico - Omitió >50% de las tareas
- `accuracy` (1-4): ¿Datos verificables vs outputs de agentes?
  - **4**: Perfecto - Todos los datos coinciden exactamente
  - **3**: Bueno - Datos mayormente correctos con mínimas variaciones
  - **2**: Mejorable - Algunos datos no coinciden
  - **1**: Crítico - Datos contradicen outputs de agentes
- `structure` (1-4): ¿Bien organizado?
  - **4**: Perfecto - Estructura clara y lógica
  - **3**: Bueno - Estructura adecuada con pequeñas mejoras posibles
  - **2**: Mejorable - Estructura confusa o desorganizada
  - **1**: Crítico - Sin estructura aparente
- `chart_attribution` (1-4): ¿Gráficos en sección correcta?
  - **4**: Perfecto - Todos los gráficos correctamente atribuidos
  - **3**: Bueno - Mayoría correctos con mínimos errores
  - **2**: Mejorable - Varios gráficos mal atribuidos
  - **1**: Crítico - Gráficos incorrectamente atribuidos o mencionados sin existir

**Errores Detectables:**
- Omisión de tareas completadas
- Gráficos mal atribuidos (ETH.png en sección BTC)
- Mención de gráficos no generados
- Invención de noticias no reportadas

---

## 2. Metodología de Evaluación {#metodología}

### 2.1 Captura de Trazas (TraceCollector)

El sistema captura CADA PASO del pipeline:

```python
# Ejemplo de traza capturada
{
  "planner_tasks": [
    "Obtener precio Bitcoin",
    "Calcular volatilidad Ethereum"
  ],
  "routing_trace": [
    {"task": "Obtener precio Bitcoin", "agent_selected": "Technical_Analyst"},
    {"task": "Calcular volatilidad Ethereum", "agent_selected": "Risk_Officer"}
  ],
  "agent_executions": [
    {
      "agent": "Technical_Analyst",
      "task": "Obtener precio Bitcoin",
      "tools_used": ["crypto_history_tool"],
      "tool_outputs": "[crypto_history_tool]: SELECT Date, Close FROM BTC_USD...",
      "agent_response": "El precio de cierre de Bitcoin es $43,521.30"
    },
    ...
  ],
  "final_answer": "**Informe Financiero**\n\n1. Bitcoin: $43,521.30..."
}
```

### 2.2 Evaluación Modular

Cada módulo se evalúa INDEPENDIENTEMENTE con un LLM especializado:

1. **Planner Judge**: Compara `user_message` vs `generated_tasks`
2. **Supervisor Judge**: Analiza `routing_trace` vs `expected_behavior`
3. **Agent Judge**: Evalúa `tool_outputs` vs `agent_response` (fidelidad)
4. **Final Output Judge**: Verifica `agent_outputs` vs `final_report`

### 2.3 Evaluación Comprehensiva

Un LLM "Auditor Jefe" recibe TODAS las evaluaciones modulares y:
- Calcula score global ponderado
- Identifica fallos críticos
- Determina categoría de error principal
- Genera resumen ejecutivo

---

## 3. Cobertura del Dataset {#dataset}

### 3.1 Estadísticas del Dataset

- **Total de Casos**: 45
- **Distribución por Dificultad**:
  - Easy: 13 casos (28.9%)
  - Medium: 14 casos (31.1%)
  - Hard: 13 casos (28.9%)
  - Very Hard: 5 casos (11.1%)

### 3.2 Cobertura por Módulo

| Módulo | Casos | % |
|--------|-------|---|
| Planner | 6 | 13.3% |
| Supervisor | 9 | 20.0% |
| Technical_Analyst | 7 | 15.6% |
| Fundamental_Analyst | 5 | 11.1% |
| Risk_Officer | 5 | 11.1% |
| Final_Output | 4 | 8.9% |
| End-to-End | 9 | 20.0% |

### 3.3 Cobertura por Tipo de Error

| Error Esperado | Casos |
|----------------|-------|
| None (Baseline) | 7 |
| Planning_Error | 6 |
| Routing_Error | 9 |
| Tool_Error (Logic) | 3 |
| Fabrication | 6 |
| Incompleteness | 5 |
| Risk_Negligence | 2 |
| Parametric_Leak | 3 |
| Loop_Error | 1 |
| Chart_Attribution | 2 |
| Task_Completion | 1 |

### 3.4 Casos Destacados

**TC-006**: Top 3 Precios (Test de Precisión)
- **Objetivo**: Verificar que Planner no simplifique "Top 3" a "precio"
- **Esperado**: Tarea = "Obtener Top 3 precios más altos Bitcoin"
- **Fallo común**: Planner genera "Obtener precio Bitcoin" (pérdida de precisión)

**TC-017**: SQL Top 3 (Test de Lógica)
- **Objetivo**: Verificar que SQL use ORDER BY Close DESC, NO WHERE Date
- **Esperado**: `SELECT Date, Close FROM ETH_USD ORDER BY Close DESC LIMIT 3`
- **Fallo común**: `WHERE Date = (SELECT MAX(Date)...)` → solo devuelve 1 registro

**TC-020**: Completitud de Output
- **Objetivo**: Verificar que agente liste TODOS los items (Top 5 = 5 items)
- **Esperado**: Lista numerada con 5 registros completos
- **Fallo común**: "Los 5 precios más altos fueron entre $X y $Y" (resumen)

**TC-024**: Parametric Leak
- **Objetivo**: Detectar si Fundamental usa conocimiento interno en vez de RAG
- **Esperado**: Si RAG vacío → "No hay información en la documentación"
- **Fallo común**: Explica Polkadot usando memoria interna del LLM

**TC-029**: Risk Negligence
- **Objetivo**: Verificar que Risk Officer advierta si volatilidad > 5%
- **Esperado**: "⚠️ Volatilidad ALTA del 8.2%. Riesgo significativo."
- **Fallo común**: Solo reporta "8.2%" sin advertencia

**TC-034**: Chart Attribution
- **Objetivo**: Verificar que gráficos se asocien al activo correcto
- **Esperado**: BTC.png en sección Bitcoin, ETH.png en sección Ethereum
- **Fallo común**: Informe final menciona ambos gráficos en sección Bitcoin

**TC-039**: Análisis Completo (Stress Test)
- **Objetivo**: Test end-to-end con 5 tareas sobre 1 activo
- **Tareas**: Precio, Top 3, Noticias, Volatilidad, Gráfico de Solana
- **Agentes**: Technical (3 tareas), Fundamental (1), Risk (1)
- **Esperado**: Informe consolidado con 5 secciones sobre Solana

---

## 4. Métricas y Scoring {#métricas}

### 4.1 Ponderación del Score Global

```
Overall Score = (
  Planner_Avg * 0.20 +
  Supervisor_Avg * 0.25 +
  Agents_Avg * 0.40 +
  Final_Output_Avg * 0.15
)
```

Donde:
- `Planner_Avg = (correctness + completeness + precision + task_decomposition) / 4`
- `Supervisor_Avg = (routing_accuracy + task_completion) / 2`
- `Agents_Avg = AVG[(tool_selection + tool_execution + output_fidelity + output_completeness + hallucination_check) / 5]` para cada agente
- `Final_Output_Avg = (completeness + accuracy + structure + chart_attribution) / 4`

**Normalización para comparación:** Para comparar con Baseline (escala 0-1), dividir entre 4:
```python
llm_judge_normalized = overall_score / 4
```

### 4.2 Categorías de Error

| Categoría | Condición | Severidad |
|-----------|-----------|-----------|
| None | Overall Score ≥ 3 | - |
| Planning_Error | Planner Score ≤ 2 | Media |
| Routing_Error | Supervisor Score ≤ 2 | Alta |
| Tool_Error | Cualquier Agente (tool_execution ≤ 2) | Media |
| Fabrication | Cualquier Agente (hallucination_check = 1) | CRÍTICA |
| Incompleteness | Completeness ≤ 2 (Planner o Final) | Alta |
| Risk_Negligence | Risk Officer omitió advertencia de riesgo alto | CRÍTICA |
| Parametric_Leak | Fundamental usó conocimiento externo | Media |
| Loop_Error | Supervisor repitió tarea sin resolver | Alta |

### 4.3 Fallos Críticos

Se considera fallo crítico si:
1. Fabricación de datos (score 1 en hallucination_check)
2. Routing incorrecto en >50% de tareas (routing_accuracy ≤ 2)
3. Omisión de advertencia de riesgo crítico
4. Incompletitud mayor (completeness = 1, omitir >50% de tareas)

---

## 5. Interpretación de Resultados {#interpretación}

### 5.1 Rangos de Score

| Score | Calificación | Descripción |
|-------|--------------|-------------|
| 4 | Excelente | Sistema funcionando óptimamente |
| 3 | Bueno | Funcionamiento correcto con pequeñas áreas de mejora |
| 2 | Mejorable | Errores moderados que requieren atención |
| 1 | Crítico | Errores graves que afectan funcionalidad core |

**Equivalencia con escala anterior (0-10):**
- **4** ≈ 9-10 (Excelente)
- **3** ≈ 7-8 (Bueno)
- **2** ≈ 5-6 (Mejorable)
- **1** ≈ 0-4 (Crítico)

### 5.2 Análisis por Módulo

**Si Planner Score ≤ 2:**
- Revisar PLANNER_JUDGE_PROMPT
- Verificar que el LLM comprende la escala 1-4
- Añadir más ejemplos de preservación de precisión
- Verificar que el LLM entienda "literalidad"

**Si Supervisor Score ≤ 2:**
- Revisar lógica de routing en SUPERVISOR_ROUTER_PROMPT
- Verificar que el estado `next` se actualice correctamente
- Revisar condición de terminación (task_completion)

**Si Agents Score ≤ 2:**
- Identificar qué agente falla más (Technical/Fundamental/Risk)
- Revisar prompts específicos del agente
- Verificar herramientas: ¿devuelven el formato esperado?

**Si Final Output Score ≤ 2:**
- Revisar SUPERVISOR_SUMMARY_PROMPT
- Verificar que completed_outputs se acumulen bien
- Revisar lógica de consolidación del Supervisor

### 5.3 Patrones de Error Comunes

**Patrón 1: Score alto en Planner y Supervisor, bajo en Agents**
- **Causa**: Herramientas mal implementadas o prompts de agente débiles
- **Solución**: Revisar outputs de herramientas, reforzar prompts de agentes

**Patrón 2: Score alto en todo excepto Final Output**
- **Causa**: Supervisor no consolida bien o pierde información
- **Solución**: Mejorar SUPERVISOR_SUMMARY_PROMPT, verificar `completed_outputs`

**Patrón 3: Score bajo en Supervisor (routing) pero alto en Agents**
- **Causa**: Routing incorrecto pero agentes robustos (responden bien aunque reciban tarea incorrecta)
- **Solución**: Reforzar lógica de routing del Supervisor

**Patrón 4: Score bajo generalizado**
- **Causa**: Problema sistémico (LLM base débil, contexto insuficiente)
- **Solución**: Considerar cambiar modelo base o reestructurar arquitectura


## 6. Sistema de Acumulación de Métricas {#acumulacion}

### 6.1 MetricsLogger

El sistema **acumula automáticamente** todas las métricas de evaluación (online y offline) en CSVs unificados.

**Ubicación:** `evaluation/accumulated_data/`

**Archivos generados:**
- `online_metrics.csv` - Evaluaciones desde Streamlit (sesiones de usuario)
- `offline_metrics.csv` - Evaluaciones batch desde `run_eval.py`

**Campos almacenados:**
```csv
timestamp,source,query_id,user_query,num_tasks,difficulty,category,
baseline_score,baseline_routing_f1,baseline_numeric_f1,baseline_hallucination_rate,
baseline_task_coverage,baseline_sql_correctness,baseline_time,
llm_judge_overall,llm_judge_planner,llm_judge_supervisor,llm_judge_agents,
llm_judge_final,llm_judge_error_category,llm_judge_time,
critical_failures,raw_trace
```