# Sistema de Evaluación Baseline Multi-Agente

## 1. Introducción y Filosofía

### 1.1 ¿Qué es Baseline Metrics?

**Baseline Metrics** es un sistema de evaluación **100% determinista** que valida el funcionamiento del sistema multi-agente mediante **comparación automática** de outputs observables, sin usar LLMs como evaluadores.

### 1.2 Principios Fundamentales

#### Principio de Determinismo
> "Mismos inputs → Mismos outputs, siempre"

- **No hay variabilidad**: Ejecutar la evaluación 100 veces sobre la misma traza produce exactamente el mismo resultado.
- **No hay interpretación semántica**: Solo comparación de patterns, números y strings.

#### Principio de Ground Truth
> "Solo se evalúa lo que es verificable técnicamente"

- **Routing**: Comparación con reglas de keywords predefinidas.
- **Números**: Extracción con regex y comparación con tolerancia 1%.
- **Tareas**: Fuzzy matching con Jaccard similarity.
- **SQL**: Validación de patrones sintácticos (ORDER BY, LIMIT, etc.).

#### Principio de Velocidad
> "La evaluación debe ser órdenes de magnitud más rápida que la ejecución"

- **Sin I/O de red**: No hay llamadas a APIs externas.
- **Sin inferencia**: No hay procesamiento de LLMs.

### 1.3 ¿Cuándo Usar Baseline?

✅ **USA Baseline para:**
- Detección de errores numéricos (alucinaciones, imprecisiones)
- Validación de routing lógico
- Evaluación rápida durante desarrollo
- Métricas objetivas para comparaciones científicas

❌ **NO uses Baseline para:**
- Evaluar calidad narrativa o tono
- Detectar errores contextuales sutiles
- Validar coherencia semántica entre módulos
- Casos donde la "corrección" es subjetiva

---

## 2. Arquitectura del Sistema

### 2.1 Pipeline de Evaluación
```
Traza del Sistema → TraceCollector → Baseline Evaluator → Métricas
                                            ↓
                    ┌─────────────────────────────────────┐
                    │   4 Calculadores Independientes     │
                    ├─────────────────────────────────────┤
                    │  1. Routing Accuracy (F1-Score)     │
                    │  2. Numeric Fidelity (F1-Score)     │
                    │  3. Task Coverage (%)               │
                    │  4. SQL Correctness (%)             │
                    └─────────────────────────────────────┘
                                    ↓
                    Baseline Score (Promedio Ponderado)
```

### 2.2 Componentes del Sistema

#### TraceCollector
**Ubicación**: `app.py`, `evaluation/baseline/run_eval.py`

**Función**: Capturar datos del sistema durante la ejecución.

**Campos capturados**:
```python
{
    'user_question': str,           # Pregunta original del usuario
    'planner_tasks': List[str],     # Tareas generadas por Planner
    'routing_trace': List[Dict],    # Decisiones del Supervisor
    'agent_executions': List[Dict], # Ejecuciones de agentes
    'sql_queries': List[Dict],      # Queries SQL ejecutadas
    'final_answer': str             # Informe final consolidado
}
```

#### BaselineMetricsCalculator
**Ubicación**: `evaluation/baseline/metrics.py`

**Función**: Calcular las 4 métricas independientes.

**Métodos principales**:
- `evaluate_routing_accuracy()` → Routing F1
- `evaluate_numeric_accuracy()` → Numeric F1 + Hallucination Rate
- `evaluate_task_coverage()` → Coverage %
- `evaluate_sql_correctness()` → SQL Correctness %

#### Evaluator
**Ubicación**: `evaluation/baseline/evaluator.py`

**Función**: Orquestar los 4 calculadores y generar score global.

---

## 3. Metodología de Evaluación

### 3.1 Métrica 1: Routing Accuracy (F1-Score)

#### ¿Qué evalúa?
Valida si el **Supervisor** asignó cada tarea al agente correcto.

#### Metodología

**Paso 1: Captura del routing**
```python
routing_trace = [
    {'task': 'Calcular volatilidad Solana', 'agent': 'Risk_Officer'}
]
```

**Paso 2: Inferir agente esperado (Ground Truth)**

Reglas de keywords (prioridad descendente):

| Keywords | Agente Esperado |
|----------|-----------------|
| `qué es`, `explicar`, `noticia`, `contexto` | **Fundamental_Analyst** |
| `volatilidad`, `riesgo`, `seguro` | **Risk_Officer** |
| `precio`, `gráfico`, `top`, `predicción` | **Technical_Analyst** |
| Default | **Technical_Analyst** |

**Paso 3: Confusion Matrix**

Para cada agente **que participó** en la ejecución:
```python
TP = Asignó al agente correcto
FP = Asignó a este agente cuando no debía
FN = NO asignó a este agente cuando debía
```

**Paso 4: Calcular F1 (macro-average)**
```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Promedio SOLO de agentes involucrados
routing_f1 = AVG(f1 por cada agente participante)
```

#### Ejemplo Completo

**Input:**
```
User: "Dame el precio de Bitcoin y la volatilidad de Ethereum"

Planner tasks:
  - "Obtener precio Bitcoin"
  - "Calcular volatilidad Ethereum"

Routing trace:
  - Task: "Obtener precio Bitcoin" → Agent: Technical_Analyst
  - Task: "Calcular volatilidad Ethereum" → Agent: Risk_Officer
```

**Evaluación:**
```python
# Task 1: "Obtener precio Bitcoin"
Expected: Technical_Analyst (keyword: "precio")
Actual:   Technical_Analyst
✅ CORRECTO

# Task 2: "Calcular volatilidad Ethereum"
Expected: Risk_Officer (keyword: "volatilidad")
Actual:   Risk_Officer
✅ CORRECTO

# Confusion Matrix:
Technical_Analyst: TP=1, FP=0, FN=0 → F1=1.0
Risk_Officer:      TP=1, FP=0, FN=0 → F1=1.0

# Routing F1 = (1.0 + 1.0) / 2 = 1.0 (100%)
```

---

### 3.2 Métrica 2: Numeric Fidelity (F1-Score)

#### ¿Qué evalúa?
Detecta **alucinaciones numéricas**: números que el agente inventa y que **no están** en el tool output.

#### Metodología

**Paso 1: Extracción de números**

Pattern regex: `r'-?\d+(?:[.,]\d+)?'`

Normalización: Convertir comas a puntos (`2,51` → `2.51`)

**Ejemplo:**
```python
tool_output = "Volatilidad: 2.51%, Riesgo: MEDIO, Últimos 30 registros"
tool_numbers = [2.51, 30]

agent_response = "La volatilidad de Solana es del 2,51% en 30 días"
agent_numbers = [2.51, 30]
```

**Paso 2: Comparación con tolerancia**

Tolerancia: **1%** (para evitar falsos positivos por redondeo)
```python
def is_in_tolerance(number, number_list, tolerance=0.01):
    for n in number_list:
        if abs(number - n) / max(abs(n), 1) < tolerance:
            return True
    return False
```

**Paso 3: Clasificación**

Para cada número del agente:
- **TP (True Positive)**: Número está en tool_output (con tolerancia)
- **FP (False Positive)**: Número NO está en tool_output ← **ALUCINACIÓN**

Para cada número del tool:
- **FN (False Negative)**: Número NO fue reportado por el agente

**Paso 4: Calcular métricas**
```python
precision = TP / (TP + FP)  # % de números reales del total reportado
recall = TP / (TP + FN)     # % de números del tool que se reportaron
f1 = 2 * (precision * recall) / (precision + recall)
hallucination_rate = FP / (TP + FP)  # % de números inventados
```

#### Ejemplo con Alucinación

**Input:**
```
Tool output: "Volatilidad: 2.51%"
Tool numbers: [2.51]

Agent response: "La volatilidad es 2.51% con un drawdown máximo de 5.3%"
Agent numbers: [2.51, 5.3]
```

**Evaluación:**
```python
# 2.51 ∈ [2.51]? → ✅ TP
# 5.3 ∈ [2.51]?  → ❌ FP (ALUCINACIÓN)

TP = 1
FP = 1  # ← INVENTÓ 5.3
FN = 0

precision = 1 / (1+1) = 0.50  # Solo 50% de lo que dijo es real
recall = 1 / (1+0) = 1.00     # Reportó el 100% de los números reales
f1 = 2 * (0.50 * 1.00) / (0.50 + 1.00) = 0.67
hallucination_rate = 1 / (1+1) = 0.50  # ← 50% de ALUCINACIÓN
```

#### Casos Especiales

**No hay números:**
- Tool: Sin números
- Agent: Sin números
- **Resultado**: `precision=1.0, recall=1.0, f1=1.0, hallucination_rate=0.0`

**Agent no reporta nada:**
- Tool: `[2.51, 30]`
- Agent: Texto sin números
- **Resultado**: `precision=0, recall=0, f1=0, hallucination_rate=0`

---

### 3.3 Métrica 3: Task Coverage (%)

#### ¿Qué evalúa?
Verifica que **todas las tareas planificadas** se ejecutaron (no se omitieron).

#### Metodología

**Paso 1: Captura de tareas**
```python
planned_tasks = ['Calcular volatilidad Solana']

# Extraídas de agent_executions (campo 'task')
completed_tasks = ['Calcular volatilidad Solana']
```

**Paso 2: Fuzzy Matching (Jaccard Similarity)**

¿Por qué fuzzy? Porque los strings pueden variar ligeramente:
- Planner: `"Obtener Top 5 precios Bitcoin"`
- Agent execution: `"Top 5 precios BTC"`

**Algoritmo:**
```python
def fuzzy_match(text1, text2, threshold=0.4):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    similarity = intersection / union
    return similarity >= threshold
```

**Ejemplo:**
```python
planned = "Obtener Top 5 precios Bitcoin"
completed = "Top 5 precios BTC"

set1 = {'obtener', 'top', '5', 'precios', 'bitcoin'}
set2 = {'top', '5', 'precios', 'btc'}

intersection = 3  # {'top', '5', 'precios'}
union = 6
similarity = 3/6 = 0.5 ≥ 0.4 → ✅ MATCH
```

**Paso 3: Calcular Coverage**
```python
matched = 0
for planned in planned_tasks:
    for completed in completed_tasks:
        if fuzzy_match(planned, completed):
            matched += 1
            break  # No contar el mismo completed dos veces

coverage = matched / len(planned_tasks)
omission_rate = 1.0 - coverage
```

#### Ejemplo Completo

**Input:**
```
Planned tasks:
  1. Predecir precio Dogecoin
  2. Generar gráfico Solana

Completed tasks (de agent_executions):
  1. Predecir precio Dogecoin
  2. Generar gráfico Solana
```

**Evaluación:**
```python
"Predecir precio Dogecoin" ↔ "Predecir precio Dogecoin"
  Similarity: 1.0 ≥ 0.4 → ✅ MATCH

"Generar gráfico Solana" ↔ "Generar gráfico Solana"
  Similarity: 1.0 ≥ 0.4 → ✅ MATCH

matched = 2
coverage = 2 / 2 = 1.0 (100%)
omission_rate = 0.0%
```

#### Casos Especiales

**Caso 1: Tareas omitidas**
```python
Planned: ['Precio BTC', 'Volatilidad ETH']
Completed: ['Precio BTC']

# Solo 1 de 2 → Coverage = 50%
```

**Caso 2: current_task vacío (bug de captura)**
```python
Planned: ['Precio BTC']
Completed: ['']  # Bug: current_task no se capturó

# Similarity de "" con cualquier cosa = 0
# Coverage = 0% ← FALSO NEGATIVO
```

---

### 3.4 Métrica 4: SQL Correctness (%)

#### ¿Qué evalúa?
Valida que las **queries SQL** sigan **patrones correctos** según el tipo de tarea.

#### Metodología

**Paso 1: Captura de SQL**
```python
sql_queries = [
    {
        'task': 'Obtener Top 3 precios Bitcoin',
        'sql': 'SELECT Date, Close FROM BTC_USD ORDER BY Close DESC LIMIT 3'
    }
]
```

**Paso 2: Clasificación del tipo de tarea**

Detectar keywords en la tarea:

| Keywords | Tipo de Query |
|----------|---------------|
| `top`, `más alto`, `máximo` | **Top X Query** |
| `último`, `reciente`, `últimos N` | **Recent Query** |
| Otro | **General Query** |

**Paso 3: Reglas de validación**

#### Regla 1: Top X Queries

**Condiciones:**
- ✅ DEBE tener `ORDER BY`
- ✅ DEBE tener `LIMIT`
- ❌ NO debe tener `WHERE` con fecha específica

**Razón**: "Top X" significa "los X mejores de TODO el histórico", no "los X del día más reciente".

**Ejemplo CORRECTO:**
```sql
SELECT Date, Close 
FROM BTC_USD 
ORDER BY Close DESC 
LIMIT 3
```

**Ejemplo INCORRECTO:**
```sql
SELECT Date, Close 
FROM BTC_USD 
WHERE Date = (SELECT MAX(Date) FROM BTC_USD)
LIMIT 3
-- ❌ Problema: Solo devuelve 1 registro (el del día más reciente)
```

#### Regla 2: Recent Queries

**Condiciones:**
- ✅ DEBE tener `ORDER BY`
- ✅ DEBE tener `DESC` (orden descendente)

**Ejemplo CORRECTO:**
```sql
SELECT Date, Close 
FROM ETH_USD 
ORDER BY Date DESC 
LIMIT 5
```

**Ejemplo INCORRECTO:**
```sql
SELECT Date, Close 
FROM ETH_USD 
LIMIT 5
-- ❌ Problema: Sin ORDER BY, puede devolver cualquier 5 registros
```

**Paso 4: Calcular Correctness**
```python
total_queries = len(sql_queries)
violations = []

for query in sql_queries:
    task = query['task']
    sql = query['sql']
    
    query_violations = validate_sql_patterns(task, sql)
    if query_violations:
        violations.extend(query_violations)

correct_queries = sum(1 for q in sql_queries if not q['violations'])
correctness = correct_queries / total_queries if total_queries > 0 else 1.0
```

#### Ejemplo Completo

**Input:**
```
Task: "Dame el Top 3 de precios de Bitcoin"
SQL: "SELECT Date, Close FROM BTC_USD WHERE Date >= '2025-01-01' LIMIT 3"
```

**Evaluación:**
```python
# Tipo detectado: Top X Query (keyword: "top")

# Validación:
✅ Tiene LIMIT? → Sí
❌ Tiene ORDER BY? → NO
❌ Tiene WHERE con fecha? → SÍ (no debería)

# Violations:
violations = [
    "Missing ORDER BY for Top X query",
    "Top X should not filter by specific date"
]

# Correctness:
correct_queries = 0 / 1 = 0.0%
```

#### Casos Especiales

**No hay SQL queries:**
```python
sql_queries = []

# Correctness = 1.0 (100%)
# Razón: No hay queries que validar, asumimos correcto
```

**Query compleja (subqueries, JOINs):**
```python
# Baseline NO valida lógica compleja
# Solo patrones básicos (ORDER BY, LIMIT, WHERE Date)
```

---

## 4. Score Global (Baseline Score)

### 4.1 Fórmula de Agregación
```python
baseline_score = (
    routing_f1 * 0.30 +          # 30% del peso
    numeric_f1 * 0.30 +          # 30% del peso
    task_coverage * 0.25 +       # 25% del peso
    sql_correctness * 0.15       # 15% del peso
)
```

### 4.2 Justificación de Pesos

| Métrica | Peso | Razón |
|---------|------|-------|
| **Routing F1** | 30% | Routing incorrecto = sistema completamente roto |
| **Numeric F1** | 30% | Alucinaciones = pérdida total de confianza del usuario |
| **Task Coverage** | 25% | Omisión de tareas = experiencia incompleta |
| **SQL Correctness** | 15% | Menor peso porque: (1) solo afecta a Technical_Analyst, (2) errores SQL → resultados raros pero no inventados |

### 4.3 Ejemplos de Cálculo

#### Caso Perfecto
```python
routing_f1       = 1.000
numeric_f1       = 1.000
task_coverage    = 1.000
sql_correctness  = 1.000

baseline_score = 1.0×0.30 + 1.0×0.30 + 1.0×0.25 + 1.0×0.15
               = 0.30 + 0.30 + 0.25 + 0.15
               = 1.000 (100%)
```

#### Caso con Errores
```python
routing_f1       = 0.667  # Routing incorrecto en 1 de 3 tareas
numeric_f1       = 0.800  # 20% de alucinaciones
task_coverage    = 1.000  # Completó todas las tareas
sql_correctness  = 0.500  # SQL sin ORDER BY

baseline_score = 0.667×0.30 + 0.800×0.30 + 1.0×0.25 + 0.5×0.15
               = 0.200 + 0.240 + 0.250 + 0.075
               = 0.765 (76.5%)
```

#### Caso Crítico (Alucinación masiva)
```python
routing_f1       = 1.000  # Routing correcto
numeric_f1       = 0.200  # 80% de alucinaciones ← CRÍTICO
task_coverage    = 1.000
sql_correctness  = 1.000

baseline_score = 1.0×0.30 + 0.2×0.30 + 1.0×0.25 + 1.0×0.15
               = 0.30 + 0.06 + 0.25 + 0.15
               = 0.760 (76%)
```

**Nota**: A pesar de tener routing, coverage y SQL perfectos, el alto nivel de alucinaciones baja el score significativamente.

---

## 5. Cobertura del Dataset

### 5.1 Dataset Compartido

Baseline usa **el mismo dataset** que LLM-Judge:
- **Ubicación**: `evaluation/llm_j/dataset.json`
- **Total de casos**: 80
- **Categorías**: 20+ (Planner, Routing, Technical, Fundamental, Risk, End-to-End)

### 5.2 Métricas Aplicables por Caso

No todos los casos ejercitan todas las métricas:

| ID | Categoría | Routing | Numeric | Coverage | SQL |
|----|-----------|---------|---------|----------|-----|
| TC-001 | Planner_Basics | ✅ | ✅ | ✅ | ✅ |
| TC-007 | Routing_Technical | ✅ | ✅ | ✅ | ✅ |
| TC-016 | Technical_SQL_Simple | ✅ | ✅ | ✅ | ✅ |
| TC-023 | Fundamental_RAG | ✅ | ❌ | ✅ | ❌ |
| TC-028 | Risk_Volatility | ✅ | ✅ | ✅ | ❌ |

**Leyenda:**
- ✅ Métrica aplicable
- ❌ Métrica N/A (valor por defecto: 1.0)

### 5.3 Casos Críticos para Baseline

#### TC-017: SQL Top 3 (High Priority)
**Objetivo**: Detectar SQL malformado para queries Top X.

**Input:**
```
"¿Cuáles fueron los 3 días con el precio más alto de Ethereum?"
```

**SQL Esperado:**
```sql
SELECT Date, Close FROM ETH_USD ORDER BY Close DESC LIMIT 3
```

**SQL Incorrecto (común):**
```sql
SELECT Date, Close FROM ETH_USD 
WHERE Date = (SELECT MAX(Date) FROM ETH_USD) 
LIMIT 3
-- ❌ Solo devuelve 1 registro, no 3
```

**Validación Baseline:**
```python
# Detecta keyword "precio más alto" → Top X Query
# Validación:
❌ Tiene WHERE Date → VIOLATION
✅ Tiene LIMIT → OK

sql_correctness = 0.0
```

---

#### TC-019: Numeric Hallucination
**Objetivo**: Detectar números inventados.

**Tool Output:**
```
PREDICCIÓN ML para ADA_USD: $0.58
```

**Agent Response (INCORRECTO):**
```
"El modelo predice $0.58 con un intervalo de confianza del 95% de ±$0.05"
```

**Validación Baseline:**
```python
tool_numbers = [0.58]
agent_numbers = [0.58, 95, 0.05]

# 0.58 ∈ [0.58]? → ✅ TP
# 95 ∈ [0.58]?   → ❌ FP (ALUCINACIÓN)
# 0.05 ∈ [0.58]? → ❌ FP (ALUCINACIÓN)

TP = 1, FP = 2, FN = 0
hallucination_rate = 2 / (1+2) = 0.67 (67% de alucinación)
```

---

#### TC-020: Completeness (Top 5 debe listar 5)
**Objetivo**: Verificar que agente reporte TODOS los items.

**Tool Output:**
```sql
SELECT Date, Close FROM SOL_USD ORDER BY Close DESC LIMIT 5
-- Devuelve 5 filas
```

**Agent Response (INCORRECTO):**
```
"Los 5 precios más altos de Solana estuvieron entre $150 y $200"
```

**Validación Baseline:**
```python
tool_numbers = [200, 195, 190, 185, 180, 150]  # Extracto de las 5 filas
agent_numbers = [150, 200, 5]  # Solo mencionó rango

# Numeric F1 será bajo (solo reportó 2 de 6 números)
# Task coverage = 100% (la tarea SÍ se ejecutó, pero mal)
```

**Limitación**: Baseline detecta que faltan números, pero no puede distinguir entre:
- "Listar 5 items" vs "Resumir 5 items"

**Solución**: LLM-Judge detectará esto como `Incompleteness`.

---

#### TC-022: Unknown Asset (Grounding)
**Objetivo**: Detectar si agente inventa datos para activos no existentes.

**Input:**
```
"Dame el precio de Monero (XMR)"
```

**Tool Output:**
```
ERROR: XMR_USD table does not exist
```

**Agent Response (CORRECTO):**
```
"No tengo datos de Monero en mi base de datos"
```

**Agent Response (INCORRECTO):**
```
"El precio de Monero es $180.50"
```

**Validación Baseline:**
```python
tool_numbers = []  # Tool no devolvió números
agent_numbers = [180.50]

# 180.50 ∈ []? → ❌ FP (ALUCINACIÓN)

hallucination_rate = 1 / (0+1) = 1.0 (100% inventado)
```

---
