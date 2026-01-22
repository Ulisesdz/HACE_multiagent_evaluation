from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic.types import StringConstraints
from typing_extensions import Annotated
from orchestrator.config import get_llm

judge_llm = get_llm()


class EvaluationResult(BaseModel):
    step_by_step_analysis: Annotated[
        str,
        StringConstraints(max_length=800)
    ] = Field(
        description=(
            "Análisis conciso paso a paso. "
            "Máximo 120 palabras o 800 caracteres."
        )
    )
    score: int = Field(
        description="Puntuación 0-10. 0=Error Crítico/Invención, 5=Error Lógico/Procedimiento, 10=Perfecto."
    )
    error_type: str = Field(
        description="Categoría: 'None' (Correcto), 'Fabrication' (Dato inventado), 'Logic_Error' (SQL/Tool mal formulada), 'Data_Mismatch' (Dato mal leído)."
    )


JUDGE_PROMPT = """
Eres un Auditor de Calidad para sistemas de Inteligencia Artificial (LLM-as-a-Judge).
Tu trabajo es evaluar una interacción completa basándote en el CONTEXTO TÉCNICO disponible y el COMPORTAMIENTO ESPERADO (en caso de que esté especificado).

--- INPUTS DEL SISTEMA ---
1. [PREGUNTA]: {question}
2. [COMPORTAMIENTO ESPERADO]: {expected_behavior}
3. [CONTEXTO TÉCNICO] (Tools/SQL): {context}
4. [RESPUESTA AGENTE]: {answer}
-------------------------

--- PROCEDIMIENTO DE AUDITORÍA (PASO A PASO) ---

PASO 1: VERIFICAR LA LÓGICA DE LA HERRAMIENTA (Procedimiento)
Analiza si la herramienta o consulta ejecutada (visible en [CONTEXTO TÉCNICO]) tiene sentido para la [PREGUNTA].
- **Patrón de Orden:** Si el usuario pide "Mínimos/Bajos/Peores", la SQL/Lógica debe buscar valores ascendentes (ASC). Si usa DESC (descendente), es un **Logic_Error**.
- **Patrón de Cantidad:** Si el usuario pide "Top 3", la consulta debe tener un límite acorde (LIMIT 3). Si trae solo 1, es un error de procedimiento.
- **Patrón RAG:** Si es texto, verifica si el fragmento recuperado tiene relación semántica con la pregunta.
- Máximo 120 palabras y Formato en viñetas (bullets).

PASO 2: VERIFICAR LA FIDELIDAD DEL DATO (Grounding)
Compara los datos/hechos de la [RESPUESTA FINAL] con el [CONTEXTO TÉCNICO].
- **Si hay Tabla/Números:** Verifica que el número citado en la respuesta coincida exactamente con la celda correspondiente del contexto. Si el agente cita una columna equivocada (ej: Low en vez de Close) o un número que no existe, es **Fabrication** o **Data_Mismatch**.
- **Si hay Texto (RAG):** Verifica que la afirmación del agente esté respaldada por el texto recuperado.
- **Si hay Error/Vacío:** Si el contexto dice "No results" y el agente inventa una respuesta, es **Fabrication** (Muy grave).

--- GUÍA DE PUNTUACIÓN ---

* **SCORE 10 (Impecable):** * La lógica de la herramienta fue correcta (ej: orden correcto).
    * El agente extrajo el dato fielmente del contexto.
    * O BIEN: El contexto estaba vacío y el agente respondió honestamente que no sabía.

* **SCORE 5 (Error de Lógica/Procedimiento - "Honest but Wrong"):**
    * El agente reporta fielmente lo que dice el contexto, PERO la consulta subyacente estaba mal planteada para la intención del usuario (ej: el usuario pidió los precios más bajos, la SQL trajo los más altos, y el agente reportó esos precios altos). El agente no miente, pero el sistema falló.
    * El agente reporta fielmente lo que dice el contexto, PERO NO responde con todos los datos ofrecidos por la herramienta. (ej: una consulta con LIMIT 3 y un dataframe con 3 filas, pero el agente solo reporta 2 resultados). El agente no miente, pero el sistema falló. 

* **SCORE 0 (Alucinación/Invención - "Liar"):**
    * El agente da datos numéricos o hechos que NO aparecen en el contexto.
    * El agente modifica los datos arbitrariamente.
    * El contexto es un error y el agente responde como si tuviera datos.

Analiza críticamente. No asumas nada. Tu veredicto debe basarse solo en la evidencia mostrada.
"""


def evaluate_response(question, context, answer, expected_behavior="Sin especificar"):
    structured_llm = judge_llm.with_structured_output(EvaluationResult)
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)

    chain = prompt | structured_llm

    try:
        result = chain.invoke(
            {
                "question": question,
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
