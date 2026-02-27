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

**Métricas Evaluadas:**
- `correctness` (0-10): ¿Identificó las tareas correctas?
- `completeness` (0-10): ¿Capturó TODAS las solicitudes?
- `precision` (0-10): ¿Mantuvo adjetivos, cantidades, fechas?
- `task_decomposition` (0-10): ¿Separó bien las acciones?

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
- `routing_accuracy` (0-10): % de decisiones correctas
- `task_completion` (0-10): ¿Procesó todas las tareas?
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

**Métricas:**
- `tool_selection` (0-10): ¿Eligió la herramienta correcta?
- `tool_execution` (0-10): ¿SQL correcto, parámetros válidos?
- `output_fidelity` (0-10): ¿Respuesta fiel a la evidencia técnica?
- `output_completeness` (0-10): ¿Reportó TODOS los datos? (Top 3 = 3 items)
- `hallucination_check` (0-10): ¿Inventó datos?

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

**Métricas:**
- `completeness` (0-10): ¿Incluye TODAS las tareas?
- `accuracy` (0-10): ¿Datos verificables vs outputs de agentes?
- `structure` (0-10): ¿Bien organizado?
- `chart_attribution` (0-10): ¿Gráficos en sección correcta?

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

### 4.2 Categorías de Error

| Categoría | Condición | Severidad |
|-----------|-----------|-----------|
| None | Overall Score ≥ 9 | - |
| Planning_Error | Planner Score < 5 | Media |
| Routing_Error | Supervisor Score < 5 | Alta |
| Tool_Error | Cualquier Agente (tool_execution < 5) | Media |
| Fabrication | Cualquier Agente (hallucination_check < 3) | CRÍTICA |
| Incompleteness | Completeness < 5 (Planner o Final) | Alta |
| Risk_Negligence | Risk Officer omitió advertencia de riesgo alto | CRÍTICA |
| Parametric_Leak | Fundamental usó conocimiento externo | Media |
| Loop_Error | Supervisor repitió tarea sin resolver | Alta |

### 4.3 Fallos Críticos

Se considera fallo crítico si:
1. Fabricación de datos (score 0 en hallucination_check)
2. Routing incorrecto en >50% de tareas
3. Omisión de advertencia de riesgo crítico
4. Incompletitud mayor (omitir >30% de tareas solicitadas)

---

## 5. Interpretación de Resultados {#interpretación}

### 5.1 Rangos de Score

| Score | Calificación | Descripción |
|-------|--------------|-------------|
| 9-10 | Excelente | Sistema funcionando óptimamente |
| 7-8.9 | Bueno | Errores menores, no críticos |
| 5-6.9 | Mejorable | Errores procedimentales, requiere atención |
| 3-4.9 | Deficiente | Múltiples fallos, sistema inestable |
| 0-2.9 | Crítico | Fallos graves, sistema no funcional |

### 5.2 Análisis por Módulo

**Si Planner Score < 6:**
- Revisar PLANNER_SYSTEM_PROMPT
- Añadir más ejemplos de preservación de precisión
- Verificar que el LLM entienda "literalidad"

**Si Supervisor Score < 6:**
- Revisar lógica de routing en SUPERVISOR_ROUTER_PROMPT
- Verificar que el estado `next` se actualice correctamente
- Revisar condición de terminación (task_completion)

**Si Agents Score < 6:**
- Identificar qué agente falla más (Technical/Fundamental/Risk)
- Revisar prompts específicos del agente
- Verificar herramientas: ¿devuelven el formato esperado?

**Si Final Output Score < 6:**
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