from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic.types import StringConstraints
from typing_extensions import Annotated
from orchestrator.config import get_llm

judge_llm = get_llm()


class EvaluationResult(BaseModel):
    step_by_step_analysis: Annotated[
        str,
        StringConstraints(max_length=1000)
    ] = Field(
        description=(
            "Análisis crítico paso a paso (Enrutamiento -> Herramientas -> Fidelidad). "
            "Máximo 150 palabras."
        )
    )
    score: int = Field(
        description="Puntuación 0-10. 0=Error Crítico, 5=Error Procedimiento/Routing, 10=Perfecto."
    )
    error_type: str = Field(
        description=(
            "Categoría del error: "
            "'None' (Correcto), "
            "'Routing_Error' (Supervisor eligió al agente incorrecto), "
            "'Fabrication' (Dato inventado), "
            "'Logic_Error' (SQL mal ordenado/Tool mal usada), "
            "'Risk_Negligence' (Ignoró advertencia de riesgo), "
            "'Parametric_Leak' (Usó conocimiento externo en vez de RAG), "
            "'Loop_Error' (Repetición sin respuesta)."
        )
    )

JUDGE_PROMPT = """
Eres un Auditor de Calidad Financiera y Técnica para un sistema de IA de Inversión.
Tu trabajo es evaluar la interacción completa: desde la elección del Supervisor hasta la respuesta del Agente.

--- INPUTS DE LA AUDITORÍA ---
1. [PREGUNTA DEL INVERSOR]: {question}
2. [AGENTE ELEGIDO POR SUPERVISOR]: {agent_selected}
3. [COMPORTAMIENTO ESPERADO]: {expected_behavior}
4. [EVIDENCIA TÉCNICA] (Output de Tools/SQL/RAG): 
{context}
5. [RESPUESTA FINAL]: {answer}
-------------------------

--- PROCEDIMIENTO DE AUDITORÍA (CHECKLIST JERÁRQUICO) ---

PASO 1: EVALUACIÓN DEL ENRUTAMIENTO (SUPERVISOR) 
Verifica si el [AGENTE ELEGIDO] es el especialista correcto para la [PREGUNTA]:
- **Technical_Analyst:** Solo para PRECIOS, GRÁFICOS, PREDICCIONES numéricas o SQL histórico.
- **Risk_Officer:** Solo para VOLATILIDAD, SEGURIDAD, RIESGO o "Es seguro invertir".
- **Fundamental_Analyst:** Solo para NOTICIAS, CONTEXTO, CONCEPTOS ("Qué es") o tecnología.
- **FINISH:** Para saludos o temas fuera de dominio.
-> *Si el enrutamiento es incorrecto (ej: Risk Officer para pedir un precio), marca error: **Routing_Error**.*

PASO 2: VERIFICAR COHERENCIA FINANCIERA Y RIESGO
- **Risk Officer:** Si la herramienta `crypto_volatility_tool` indica Riesgo "ALTO" o "EXTREMO", ¿la respuesta final advierte al usuario? Si lo omite, es **Risk_Negligence**.
- **Technical Analyst:** Si la herramienta devuelve una predicción numérica (ej: $3000), ¿la respuesta coincide? Si dice $3500, es **Fabrication**.
- **Fundamental Analyst:** Si usa RAG, ¿la información está en el texto recuperado? Si responde correctamente pero con datos que NO están en la [EVIDENCIA TÉCNICA], es **Parametric_Leak**.

PASO 3: VERIFICAR LÓGICA DE HERRAMIENTAS
- **Gráficos:** Si se pidió un gráfico, ¿se generó el archivo .png?
- **Orden SQL:** Si pidieron "máximos", ¿el SQL usó DESC? (Logic_Error).
- **Bucles:** ¿La evidencia muestra la misma herramienta ejecutándose múltiples veces? (Loop_Error).

PASO 4: DETECCIÓN DE ALUCINACIONES
- Si la [EVIDENCIA TÉCNICA] está vacía o dice "No data found", el agente DEBE decir "No lo sé". Si inventa un dato, es **Fabrication** (Score 0).

--- GUÍA DE PUNTUACIÓN ---

* **SCORE 10 (Impecable):** * Supervisor eligió al agente correcto.
    * Agente usó la herramienta correcta.
    * Datos fieles a la evidencia.

* **SCORE 5 (Error de Procedimiento / Routing):**
    * **Routing_Error:** El Supervisor se equivocó de agente, aunque la respuesta final sea razonable.
    * **Logic_Error:** SQL/Tool mal formulada.
    * **Parametric_Leak:** Respuesta correcta pero no basada en RAG (memoria interna).

* **SCORE 0 (Error Crítico / Mentira):**
    * **Fabrication:** Inventó un precio o dato.
    * **Risk_Negligence:** Omitió una alerta de riesgo alta.
    * **Loop_Error:** El sistema entró en bucle.

Tu veredicto debe ser estricto. La precisión financiera y el enrutamiento correcto son vitales.
"""

def evaluate_response(question, agent_selected, context, answer, expected_behavior="Sin especificar"):
    structured_llm = judge_llm.with_structured_output(EvaluationResult)
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

    chain = prompt | structured_llm

    try:
        # Limpieza básica del contexto
        if len(str(context)) > 3000:
            context = str(context)[:3000] + "... [TRUNCADO]"

        result = chain.invoke(
            {
                "question": question,
                "agent_selected": agent_selected,
                "context": context,
                "answer": answer,
                "expected_behavior": expected_behavior,
            }
        )
        return result
    except Exception as e:
        return EvaluationResult(
            step_by_step_analysis=f"Error Juez: {str(e)}",
            score=0,
            error_type="System_Error",
        )