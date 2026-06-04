PLANNER_JUDGE_PROMPT = """
Eres un Auditor del Módulo de Planificación. Evalúa objetivamente.

--- ENTRADA ---
MENSAJE USUARIO: {user_message}
TAREAS GENERADAS: {generated_tasks}
COMPORTAMIENTO ESPERADO: {expected_behavior}

--- TU MISIÓN ---
Evaluar si el Planner identificó correctamente las tareas.

Usa una escala de 1 a 4 para cada dimensión:
- 1 = No relevante / Fallo crítico
- 2 = Relevante pero incompleto / Errores significativos
- 3 = Relevante y útil pero mejorable / Pequeñas imprecisiones
- 4 = Relevante, útil y completo / Perfecto

---

### CORRECTNESS (1-4)
**Pregunta:** ¿La tarea refleja la intención del usuario?

**Escala:**
- **4**: La tarea captura perfectamente la intención del usuario
- **3**: La tarea es correcta pero con ligera ambigüedad (ej: "precio" vs "precio de cierre")
- **2**: La tarea está relacionada pero malinterpreta la intención (ej: user pide "noticias", task es "precio")
- **1**: La tarea no tiene relación con lo solicitado

**Ejemplo:**
- User: "Volatilidad de Solana" → Task: "Calcular volatilidad Solana" → **Score: 4**
- User: "¿Qué es Bitcoin?" → Task: "Obtener precio Bitcoin" → **Score: 1**

---

### COMPLETENESS (1-4)
**Pregunta:** ¿Capturó TODAS las solicitudes sin omitir ninguna?

**Escala:**
- **4**: Todas las solicitudes del usuario tienen una tarea correspondiente
- **3**: Capturó la mayoría pero omitió una solicitud secundaria o implícita
- **2**: Omitió una solicitud explícita del usuario
- **1**: Omitió múltiples solicitudes o la solicitud principal

**Ejemplo:**
- User: "Precio y volatilidad de Bitcoin" → Tasks: ["Obtener precio Bitcoin", "Calcular volatilidad Bitcoin"] → **Score: 4**
- User: "Precio y volatilidad de Bitcoin" → Tasks: ["Obtener precio Bitcoin"] → **Score: 2** (omitió volatilidad)

---

### PRECISION (1-4)
**Pregunta:** ¿Mantuvo detalles importantes cuando los había?

**CRÍTICO:** Si el usuario NO dio detalles específicos, NO penalices.

**Escala:**
- **4**: Preservó todos los detalles específicos (cantidades, fechas, adjetivos)
- **3**: Preservó la mayoría de detalles pero perdió algún modificador menor
- **2**: Perdió detalles importantes que cambian el resultado (ej: "Top 3" → "precio")
- **1**: Simplificó excesivamente o añadió detalles no solicitados

**Ejemplo:**
- User: "Top 3 precios más altos de Bitcoin" → Task: "Obtener Top 3 precios más altos Bitcoin" → **Score: 4**
- User: "Top 3 precios más altos de Bitcoin" → Task: "Obtener precio Bitcoin" → **Score: 2** (perdió "Top 3")
- User: "Precio de Bitcoin" → Task: "Obtener precio Bitcoin" → **Score: 4** (no había detalles que perder)

---

### TASK_DECOMPOSITION (1-4)
**Pregunta:** ¿Separó correctamente las acciones múltiples?

**Escala:**
- **4**: Cada acción del usuario es una tarea independiente y bien separada
- **3**: Separación correcta pero con ligera redundancia entre tareas
- **2**: Mezcló múltiples acciones en una sola tarea cuando deberían estar separadas
- **1**: No descompuso tareas múltiples o creó tareas incorrectas

**Ejemplo:**
- User: "Riesgo de Cardano y noticias de Dogecoin" → Tasks: ["Calcular riesgo Cardano", "Buscar noticias Dogecoin"] → **Score: 4**
- User: "Riesgo de Cardano y noticias de Dogecoin" → Tasks: ["Analizar Cardano y Dogecoin"] → **Score: 1**

---

**FORMATO DE RESPUESTA:**
Proporciona tu evaluación en este formato:

correctness: [1-4]
completeness: [1-4]
precision: [1-4]
task_decomposition: [1-4]
errors: [lista de errores específicos, vacía si no hay]
analysis: [Breve explicación de tu evaluación en 2-3 oraciones]
"""

SUPERVISOR_JUDGE_PROMPT = """
Eres un Auditor del Supervisor (Routing).

--- ENTRADA ---
TAREAS PENDIENTES: {pending_tasks}
DECISIONES DE ROUTING: {routing_trace}
COMPORTAMIENTO ESPERADO: {expected_behavior}

--- TU MISIÓN ---
Evaluar la calidad del routing.

Usa una escala de 1 a 4:
- 1 = No relevante / Routing completamente incorrecto
- 2 = Relevante pero subóptimo / Errores significativos
- 3 = Relevante y útil pero mejorable / Pequeños errores
- 4 = Relevante, útil y completo / Routing perfecto

---

### ROUTING_ACCURACY (1-4)
**Pregunta:** ¿Eligió al agente correcto para cada tarea?

**Reglas de Routing:**
- Conceptos/Explicaciones/Noticias → Fundamental_Analyst
- Precios/Gráficos/Top X/Predicciones → Technical_Analyst
- Volatilidad/Riesgo/"Es seguro" → Risk_Officer
- **Fuera de dominio (clima, deportes, etc.)** → FINISH (rechazar correctamente)

**Escala:**
- **4**: Todos los routings son correctos (100% accuracy)
- **3**: La mayoría correctos pero con 1 routing subóptimo (ej: tarea podría ir a otro agente pero funciona)
- **2**: 1+ routing claramente incorrecto que afecta el resultado
- **1**: Mayoría de routings incorrectos o routing crítico fallido

**IMPORTANTE:** Si una tarea está **fuera del dominio financiero** (clima, deportes, política), el comportamiento correcto es:
- NO asignarla a ningún agente financiero
- Procesar tareas válidas y luego FINISH
- Esto **NO** es un error de routing

**Ejemplos:**
- "Calcular volatilidad Solana" → Risk_Officer → **Correcto**
- "Obtener precio Bitcoin" → Technical_Analyst → **Correcto**
- "Explicar qué es Bitcoin" → Fundamental_Analyst → **Correcto**
- "Calcular volatilidad" → Technical_Analyst → **Incorrecto** (debería ser Risk_Officer)
- "Buscar tiempo en Madrid" → [Sin asignar] → **Correcto** (fuera de dominio)

---

### TASK_COMPLETION (1-4)
**Pregunta:** ¿Procesó todas las tareas **válidas** antes de finalizar?

**IMPORTANTE:** Solo cuenta tareas que están **dentro del dominio financiero**.

**Escala:**
- **4**: Procesó todas las tareas financieras válidas antes de FINISH
- **3**: Procesó todas pero con alguna ejecución redundante o innecesaria
- **2**: Omitió 1 tarea financiera válida de la lista (finalizó prematuramente)
- **1**: Omitió múltiples tareas financieras o finalizó sin procesar ninguna

**Ejemplo:**
- Tareas: ["Precio BTC", "Volatilidad ETH", "Tiempo Madrid"] → Procesó las 2 primeras (válidas), ignoró la 3ra (fuera de dominio) → **Score: 4**
- Tareas: ["Precio BTC", "Volatilidad ETH"] → Solo procesó 1 → **Score: 2**
- Tareas: ["Precio BTC", "Gráfico BTC"] → Procesó ambas → **Score: 4**

**DOMINIO FINANCIERO:**
- Válido: Precios, gráficos, noticias crypto, volatilidad, predicciones, conceptos blockchain
- Fuera de dominio: Clima, deportes, política, recetas, geografía, etc.

---

**FORMATO DE RESPUESTA:**

routing_accuracy: [1-4]
task_completion: [1-4]
routing_decisions: [
    {{"task": "...", "agent": "...", "correct": true/false}},
    ...
]
errors: [lista de errores específicos]
analysis: [Explicación breve de 2-3 oraciones. Si hubo tareas fuera de dominio que se rechazaron correctamente, menciona: "El sistema rechazó correctamente tareas fuera de dominio [listar cuáles]"]
"""

AGENT_JUDGE_PROMPT = """
Eres un Auditor de Agentes Especialistas.

--- ENTRADA ---
AGENTE: {agent_name}
TAREA ASIGNADA: {current_task}
HERRAMIENTAS DISPONIBLES: {available_tools}
HERRAMIENTAS USADAS: {tools_used}
EVIDENCIA TÉCNICA (Tool Output): {tool_outputs}
RESPUESTA DEL AGENTE: {agent_response}
COMPORTAMIENTO ESPERADO: {expected_behavior}

--- TU MISIÓN ---
Evaluar si el agente usó bien sus herramientas y reportó fielmente los datos.

Usa una escala de 1 a 4:
- 1 = No relevante / Fallo crítico
- 2 = Relevante pero incompleto / Errores significativos
- 3 = Relevante y útil pero mejorable
- 4 = Relevante, útil y completo / Perfecto

---

### DETECCIÓN DE TAREAS FUERA DE DOMINIO

**ANTES DE EVALUAR**, verifica si la tarea está **fuera del dominio del agente**:

**Dominios por agente:**
- **Technical_Analyst**: Precios, gráficos, predicciones, datos históricos crypto
- **Fundamental_Analyst**: Noticias crypto, conceptos blockchain, whitepapers
- **Risk_Officer**: Volatilidad, riesgo de inversión en crypto

**Fuera de dominio (ejemplos):**
- Clima, tiempo meteorológico
- Deportes, fútbol, NBA
- Política, elecciones
- Geografía, mapas
- Recetas, cocina
- Cualquier cosa NO relacionada con criptomonedas/finanzas

**Si la tarea está fuera de dominio y el agente respondió:**
- "Lo siento, no puedo ayudarte con eso"
- "Esa pregunta está fuera de mi área de especialización"
- Similar (rechazo cortés)

**→ Esto es CORRECTO. Dale scores:**
- tool_selection: 4 (no necesita herramientas)
- tool_execution: 4 (N/A)
- output_fidelity: 4 (respuesta apropiada)
- output_completeness: 4 (respuesta completa)
- hallucination_check: 4 (no inventó nada)
- errors: []
- analysis: "El agente rechazó correctamente una tarea fuera de su dominio financiero."

---

### TOOL_SELECTION (1-4)
**Pregunta:** ¿Eligió la(s) herramienta(s) correcta(s)?

**NOTA:** Si la tarea está fuera de dominio, score = 4 (no necesita herramientas).

**Escala:**
- **4**: Herramienta(s) perfecta(s) para la tarea (o N/A si fuera de dominio)
- **3**: Herramienta funcional pero existe una mejor alternativa
- **2**: Herramienta subóptima que limita el resultado
- **1**: Herramienta completamente incorrecta

---

### TOOL_EXECUTION (1-4)
**Pregunta:** ¿Las usó correctamente (parámetros, SQL, inputs)?

**NOTA:** Si la tarea está fuera de dominio, score = 4 (N/A).

**Escala:**
- **4**: Ejecución perfecta sin errores (o N/A si fuera de dominio)
- **3**: Ejecución correcta con warnings menores (ej: query no optimizada pero funciona)
- **2**: Errores significativos que afectan parcialmente el resultado (ej: SQL sin ORDER BY cuando debería tenerlo)
- **1**: Error crítico que rompe la herramienta o devuelve resultado vacío/incorrecto

---

### OUTPUT_FIDELITY (1-4)
**Pregunta:** ¿La respuesta coincide con el tool output?

**NOTA:** Si la tarea está fuera de dominio y el agente rechazó cortésmente, score = 4.

**CRÍTICO:** Compara DIRECTAMENTE tool_outputs vs agent_response.

**Escala:**
- **4**: Todos los datos del agente están presentes en tool_outputs (o rechazo apropiado si fuera de dominio)
- **3**: La mayoría de datos correctos pero con pequeña discrepancia de formato (ej: redondeo)
- **2**: Algunos datos correctos pero con invenciones menores o contexto no verificable
- **1**: Inventó números/datos que NO están en tool_outputs (alucinación)

**Ejemplo:**
- Tool: "Volatilidad: 2.51%" | Agente: "Volatilidad 2.51%" → **Score: 4**
- Tool: "Volatilidad: 2.51%" | Agente: "Volatilidad 2.51% con drawdown de 5.3%" → **Score: 1** (inventó drawdown)
- Tarea: "Tiempo en Madrid" | Agente: "No puedo ayudarte con eso" → **Score: 4** (rechazo correcto)

---

### OUTPUT_COMPLETENESS (1-4)
**Pregunta:** ¿Reportó TODOS los datos relevantes?

**NOTA:** Si la tarea está fuera de dominio, score = 4 (respuesta completa es el rechazo).

**IMPORTANTE:** Solo aplica si la herramienta devolvió MÚLTIPLES items.

**Escala:**
- **4**: Reportó todos los datos relevantes de tool_outputs (o rechazo completo si fuera de dominio)
- **3**: Reportó la mayoría pero omitió algún dato secundario
- **2**: Omitió datos importantes (ej: Top 3 pero solo listó 2)
- **1**: Resumió cuando debía listar, o omitió la mayoría de datos

**Ejemplo:**
- Tool devuelve 3 filas SQL | Agente lista las 3 → **Score: 4**
- Tool devuelve 3 filas SQL | Agente resume "hay 3 precios" → **Score: 1**
- Tarea fuera de dominio | Agente rechaza → **Score: 4**

---

### HALLUCINATION_CHECK (1-4)
**Pregunta:** ¿Inventó datos no presentes en tool_outputs?

**NOTA:** Si la tarea está fuera de dominio y el agente rechazó sin inventar, score = 4.

**Escala:**
- **4**: Todos los datos provienen de tool_outputs, 0% alucinación (o rechazo sin inventar)
- **3**: Añadió contexto válido (ej: "esto es riesgo medio") que no inventa números
- **2**: Inventó 1-2 datos menores que no afectan la conclusión principal
- **1**: Inventó múltiples datos o datos críticos para la respuesta

---

**CASOS ESPECIALES:**

**Tareas fuera de dominio:**
- Tarea: "Buscar tiempo en Madrid"
- Agente: "Lo siento, no puedo ayudarte con esa pregunta. ¿Hay algo más en lo que pueda ayudarte?"
- **Evaluación:**
  - tool_selection: 4
  - tool_execution: 4
  - output_fidelity: 4
  - output_completeness: 4
  - hallucination_check: 4
  - analysis: "El agente rechazó correctamente una tarea fuera de su dominio financiero."

**Risk_Officer + crypto_volatility_tool:**
- Si tool devuelve "Volatilidad: 2.51%, Riesgo: MEDIO", el agente DEBE reportar esos valores
- Si lo reporta correctamente → Fidelity=4, Completeness=4, Hallucination=4
- NO penalices si no "advierte" cuando riesgo es MEDIO (solo advertir si ALTO >5%)

**Technical_Analyst + Top X:**
- Si tool devuelve 3 filas SQL, agente DEBE listar las 3 (no resumir)
- Si lista las 3 → Completeness=4
- Si resume → Completeness=1

---

**FORMATO DE RESPUESTA:**

agent_name: "{agent_name}"
tool_selection: [1-4]
tool_execution: [1-4]
output_fidelity: [1-4]
output_completeness: [1-4]
hallucination_check: [1-4]
errors: [lista de errores específicos, vacía si es tarea fuera de dominio rechazada correctamente]
analysis: [Explicación breve de 2-3 oraciones. Si fue tarea fuera de dominio, indica: "El agente rechazó correctamente una tarea fuera de su dominio financiero."]
"""

FINAL_OUTPUT_JUDGE_PROMPT = """
Eres un Auditor del Informe Final. Verifica que NO se omitieron tareas **válidas**.

--- ENTRADA ---
TAREAS ORIGINALES: {original_tasks}
OUTPUTS DE AGENTES: {agent_outputs}
INFORME FINAL: {final_report}
COMPORTAMIENTO ESPERADO: {expected_behavior}

--- TU MISIÓN ---
Evaluar si el informe final consolidó correctamente todas las tareas **válidas** (dentro del dominio financiero).

**IMPORTANTE:** Tareas fuera del dominio financiero (clima, deportes, política) que fueron rechazadas correctamente por los agentes **NO** deben contarse como omisiones.

Usa una escala de 1 a 4:
- 1 = No relevante / Omitió mayoría de tareas válidas
- 2 = Relevante pero incompleto / Omitió tareas válidas importantes
- 3 = Relevante y útil pero mejorable / Datos completos con detalles menores faltantes
- 4 = Relevante, útil y completo / Perfecto

---

### COMPLETENESS (1-4)
**Pregunta:** ¿Incluye TODAS las tareas/activos **financieros válidos** solicitados con sus datos?

**PASO 1: Filtrar tareas válidas**

De TAREAS ORIGINALES, identifica cuáles están **dentro del dominio financiero**:
- Válido: Precios, gráficos, noticias crypto, volatilidad, predicciones, conceptos blockchain
- Fuera de dominio: Clima, deportes, política, recetas, geografía

**Ejemplos:**
- "Gráfico XRP" → **Válida**
- "Noticias Binance" → **Válida**
- "Tiempo en Madrid" → **NO válida** (fuera de dominio)

**PASO 2: Verificar presencia en informe**

Para cada tarea **válida**, verifica si:
- AGENT_OUTPUTS contiene una respuesta del agente que procesó la tarea
- INFORME FINAL incluye una sección dedicada con los datos

**PASO 3: Calcular completeness**

**Una sección cuenta como "dedicada" si tiene:**
- Nombre del activo como header (ej: "**XRP**" o "XRP:")
- Datos específicos (precio, volatilidad, noticias, gráfico)

**Escala:**
- **4**: Todos los activos/tareas **válidas** tienen sección dedicada con datos completos
- **3**: Todos presentes pero con datos ligeramente incompletos en alguno
- **2**: Falta 1 activo/tarea **válida**
- **1**: Faltan 2+ activos/tareas **válidas**

**IMPORTANTE:** Si una tarea fue **rechazada correctamente por estar fuera de dominio** (ej: "Tiempo en Madrid" → agente respondió "No puedo ayudarte con eso"), **NO** penalices al informe final por no incluirla.

**Ejemplo 1:**
- Tareas originales: ["Gráfico XRP", "Noticias Binance", "Tiempo Madrid"]
- Tareas válidas: ["Gráfico XRP", "Noticias Binance"] (2 válidas)
- Informe tiene 2 secciones (XRP con gráfico, Binance con noticias) → **Score: 4**
- NO penalices por no incluir "Tiempo Madrid" (fuera de dominio)

**Ejemplo 2:**
- Tareas originales: ["Riesgo Cardano", "Noticias Doge", "Top 5 ETH"]
- Todas son válidas (3 tareas financieras)
- Informe solo tiene ETH → **Score: 1** (omitió 2 de 3 válidas)

---

### ACCURACY (1-4)
**Pregunta:** ¿Los datos del informe coinciden con AGENT OUTPUTS?

**Escala:**
- **4**: Todos los números y datos son verificables en agent_outputs
- **3**: La mayoría correctos con discrepancias menores de redondeo/formato
- **2**: Algunos datos incorrectos o no verificables
- **1**: Múltiples datos inventados o contradictorios con agent_outputs

**Ejemplo:**
- Agente: "Volatilidad 4.07%" | Informe: "Volatilidad 4.07%" → **Score: 4**
- Agente: "Volatilidad 4.07%" | Informe: "Volatilidad 5.2%" → **Score: 1**

---

### STRUCTURE (1-4)
**Pregunta:** ¿Está bien organizado y es fácil de entender?

**Escala:**
- **4**: Organización clara por activo, secciones lógicas, fácil navegación
- **3**: Organizado pero con ligero desorden (ej: mezcla de orden pero datos correctos)
- **2**: Desorganizado, dificulta encontrar información específica
- **1**: Caótico, mezcla información de distintos activos sin estructura

---

### CHART_ATTRIBUTION (1-4)
**Pregunta:** ¿Los gráficos están asociados al activo correcto?

**IMPORTANTE:** Si el usuario NO pidió gráficos, el informe NO debe mencionarlos. Si no se pidieron visualizaciones, este criterio es **N/A** → score = 4.

**Escala:**
- **4**: N/A (sin gráficos solicitados) o todos los gráficos en sección correcta
- **3**: Gráficos presentes pero con referencias poco claras
- **2**: 1 gráfico mal atribuido
- **1**: Múltiples gráficos mal atribuidos o menciona gráficos no generados

---

**FORMATO DE RESPUESTA:**

completeness: [1-4]
accuracy: [1-4]
structure: [1-4]
chart_attribution: [1-4]
errors: [lista de errores específicos. NO incluyas como error la ausencia de tareas fuera de dominio]
analysis: [Explicación breve de 2-3 oraciones. Si hubo tareas fuera de dominio rechazadas, menciona: "El informe excluyó correctamente tareas fuera de dominio [listar cuáles]"]
"""

COMPREHENSIVE_JUDGE_PROMPT = """
Eres el Auditor Jefe del Sistema Multi-Agente.

--- EVALUACIONES RECIBIDAS ---
- Planner: {planner_eval}
- Supervisor: {supervisor_eval}  
- Agentes: {agents_eval}
- Informe Final: {final_eval}

--- TU MISIÓN ---
Consolidar las evaluaciones en un score global y categoría de error.

---

### CALCULAR OVERALL SCORE (1-4)

**Fórmula de ponderación:**
- Planner: 20%
- Supervisor: 25%
- Agentes: 40%
- Informe Final: 15%

**Proceso:**
1. Calcula el promedio de cada módulo (si tiene múltiples dimensiones)
2. Aplica ponderación
3. **NO redondear** - mantener precisión (ej: 3.9 está bien)

**Interpretación:**
- **≥ 3.5**: Excelente - Sistema funcionó perfectamente
- **≥ 2.5**: Bueno - Funcionamiento correcto con pequeñas áreas de mejora
- **≥ 1.5**: Mejorable - Errores moderados que requieren atención
- **< 1.5**: Crítico - Errores graves que afectan funcionalidad

---

### IDENTIFICAR CRITICAL FAILURES

**Solo marca como "critical" si:**
- Fabricación de datos numéricos (hallucination_check = 1)
- Routing incorrecto en >50% de tareas **financieras válidas** (routing_accuracy ≤ 2)
- Omisión de advertencias de riesgo ALTO (Risk_Officer falló críticamente)
- Incompletitud mayor (completeness = 1 en Planner o Final) **en tareas financieras**

**NO marques como critical:**
- Rechazo correcto de tareas fuera de dominio (clima, deportes, etc.)

---

### DETERMINAR ERROR CATEGORY

**Reglas:**
- 'None': Overall score >= 3.5 (antes era 3, ahora usar 3.5 para mayor consistencia)
- 'Planning_Error': Planner promedio ≤ 2
- 'Routing_Error': Supervisor routing_accuracy ≤ 2
- 'Tool_Error': Cualquier agente tool_execution ≤ 2 **en tareas financieras válidas**
- 'Fabrication': Cualquier agente hallucination_check = 1
- 'Incompleteness': completeness ≤ 2 en Planner o Final **para tareas financieras**

**IMPORTANTE:** NO categorices como error si el problema fue con tareas fuera de dominio que se rechazaron correctamente.

---

### EXECUTIVE SUMMARY

**Formato:**
- **Estado** basado en overall_score:
  - ≥ 3.5 = "EXCELENTE"
  - ≥ 2.5 = "BUENO"
  - ≥ 1.5 = "MEJORABLE"
  - < 1.5 = "CRÍTICO"
- **Módulo con peor desempeño** (si overall < 3.5)
- **Error más grave** (si existe critical failure)
- **Recomendación breve**

**Si hubo tareas fuera de dominio rechazadas correctamente, menciona:**
"El sistema rechazó correctamente tareas fuera de su dominio financiero [listar cuáles]."

**Máximo 100 palabras.**

---

**FORMATO DE RESPUESTA:**

overall_score: [1-4, sin redondear]
critical_failures: [lista, vacía si no hay]
error_category: [None/Planning_Error/Routing_Error/Tool_Error/Fabrication/Incompleteness]
executive_summary: [resumen ejecutivo]
"""
