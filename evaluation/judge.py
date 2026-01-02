from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from orchestrator.config import get_llm

judge_llm = get_llm()

class EvaluationResult(BaseModel):
    score: int = Field(description="Puntuación de 0 a 10. 0=Alucinación total, 10=Perfecto y Fiel.")
    reasoning: str = Field(description="Explicación breve del veredicto.")
    hallucination_type: str = Field(description="Tipo de error: 'None', 'Data_Error', 'Source_Confusion', 'Fabrication'.")


JUDGE_PROMPT = """
Eres un Juez Imparcial evaluando un sistema de IA.
Tu trabajo es detectar ALUCINACIONES.

--- DATOS DEL JUICIO ---
PREGUNTA DEL USUARIO: {question}
CONTEXTO REAL (Lo que devolvió la base de datos/herramienta): 
{context}
------------------------
RESPUESTA DEL AGENTE: 
{answer}
------------------------

--- CRITERIOS DE EVALUACIÓN ---
1. FIDELIDAD: ¿La respuesta del agente se basa EXCLUSIVAMENTE en el 'CONTEXTO REAL'?
   - Si el contexto dice "Error/No data" y el agente da un dato numérico -> ES UNA ALUCINACIÓN (Fabrication). PUNTUACIÓN: 0.
   - Si el contexto trae la columna 'Low' pero el agente dice que es 'Close' -> ES UN ERROR DE DATOS (Data_Error). PUNTUACIÓN: 3.
   - Si el agente responde correctamente usando los datos -> PUNTUACIÓN: 10.

2. ATRIBUCIÓN:
   - Si es un dato histórico pero el agente dice "Predicción" -> Source_Confusion. PUNTUACIÓN: 5.

Evalúa paso a paso. Sé estricto.
"""

def evaluate_response(question, context, answer):
    structured_llm = judge_llm.with_structured_output(EvaluationResult)
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
        return result
    except Exception as e:
        return EvaluationResult(score=0, reasoning=f"Error ejecutando juez: {e}", hallucination_type="System_Error")