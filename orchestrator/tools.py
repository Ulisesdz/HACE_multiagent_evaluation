from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_chroma import Chroma
from orchestrator.config import get_llm, get_embeddings, CRYPTO_DB, WEATHER_DB, CRYPTO_VECTOR_DB, WEATHER_VECTOR_DB
import sqlite3
import pandas as pd
import sys
import os

# Añadimos la raíz al path para poder importar tus módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importamos TUS módulos de predicción (Ajusta los nombres de función según tu código real)
# Ejemplo: from crypto.src.predictor import predict_next_close
# Ejemplo: from weather.src.predictor import predict_tomorrow_temp

llm = get_llm()
embeddings = get_embeddings()

# ==========================================
# HERRAMIENTAS CRYPTO
# ==========================================

@tool
def crypto_history_tool(query: str):
    """Útil para consultar precios PASADOS o datos HISTÓRICOS de criptomonedas en la base de datos."""
    db = SQLDatabase.from_uri(f"sqlite:///{CRYPTO_DB}")
    chain = create_sql_query_chain(llm, db)
    try:
        sql_query = chain.invoke({"question": query})
        clean_query = sql_query.replace("```sql", "").replace("```", "").strip()
        conn = sqlite3.connect(CRYPTO_DB)
        df = pd.read_sql_query(clean_query, conn)
        conn.close()
        return df.to_string() if not df.empty else "No se encontraron datos históricos."
    except Exception as e:
        return f"Error en consulta SQL: {e}"

@tool
def crypto_prediction_tool(coin: str):
    """Útil SOLO para predecir precios FUTUROS o de MAÑANA. Input: Ticker (BTC-USD, ETH-USD, SOL-USD)."""
    try:
        # AQUÍ LLAMAS A TU CÓDIGO REAL
        # prediction = crypto_predictor.predict(coin) 
        # Simulación para el ejemplo:
        return f"La predicción del modelo Random Forest para {coin} mañana es: SUBIDA (Tendencia positiva)" 
    except Exception as e:
        return f"Error en predicción: {e}"

@tool
def crypto_rag_tool(query: str):
    """Útil para definiciones generales sobre Blockchain, qué es Bitcoin, historia de Ethereum, etc."""
    vectorstore = Chroma(persist_directory=CRYPTO_VECTOR_DB, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

# ==========================================
# HERRAMIENTAS WEATHER
# ==========================================

@tool
def weather_history_tool(query: str):
    """Útil para consultar temperaturas PASADAS de ciudades en la base de datos."""
    db = SQLDatabase.from_uri(f"sqlite:///{WEATHER_DB}")
    chain = create_sql_query_chain(llm, db)
    try:
        sql_query = chain.invoke({"question": query})
        clean_query = sql_query.replace("```sql", "").replace("```", "").strip()
        conn = sqlite3.connect(WEATHER_DB)
        df = pd.read_sql_query(clean_query, conn)
        conn.close()
        return df.to_string() if not df.empty else "No se encontraron datos históricos."
    except Exception as e:
        return f"Error en consulta SQL: {e}"

@tool
def weather_prediction_tool(city: str):
    """Útil SOLO para predecir la temperatura de MAÑANA o FUTURA. Input: Nombre ciudad (Madrid, NY, Paris, Tokio)."""
    try:
        # AQUÍ LLAMAS A TU CÓDIGO REAL
        # prediction = weather_predictor.predict(city)
        # Simulación:
        return f"El modelo predice para {city} mañana: 22°C con cielo despejado."
    except Exception as e:
        return f"Error en predicción: {e}"

@tool
def weather_rag_tool(query: str):
    """Útil para información general climática, características de las ciudades, o geografía."""
    vectorstore = Chroma(persist_directory=WEATHER_VECTOR_DB, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])