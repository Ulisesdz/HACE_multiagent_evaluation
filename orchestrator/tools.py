import sqlite3
import pandas as pd
import sys
import os
import joblib
import numpy as np

from langchain_core.tools import tool
from langchain_chroma import Chroma


from orchestrator.utils import log_execution, get_table_columns, get_available_entities
from orchestrator.prompts import get_sql_generation_prompt
from orchestrator.config import (
    get_llm, get_embeddings, CRYPTO_DB, WEATHER_DB, 
    CRYPTO_VECTOR_DB, WEATHER_VECTOR_DB)

# Raíz al path para poder importar tus módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

llm = get_llm()
embeddings = get_embeddings()

# ==========================================
# HERRAMIENTAS CRYPTO
# ==========================================
@tool
@log_execution
def crypto_history_tool(query: str):
    """Útil para consultar precios PASADOS o datos HISTÓRICOS de criptomonedas en la base de datos."""
    
    # 1. INTROSPECCIÓN DEL ESQUEMA
    try:
        # Lista limpia de tablas ['BTC_USD', 'ETH_USD']
        tables = get_available_entities(CRYPTO_DB)
        schema_info = []
        for t in tables:
            cols = get_table_columns(CRYPTO_DB, t)
            schema_info.append(f"Tabla '{t}': {cols}")
        
        schema_str = "\n".join(schema_info)

        # 2. GENERACIÓN DE SQL
        prompt = get_sql_generation_prompt(schema_str, query)

        sql_query = llm.invoke(prompt).content.strip().replace("```sql", "").replace("```", "")
        print(f"   └─ SQL Generado: {sql_query}")

        conn = sqlite3.connect(CRYPTO_DB)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        return df.to_string() if not df.empty else "No se encontraron datos históricos."

    except Exception as e:
        return f"Error en consulta SQL Crypto: {e}"

@tool
@log_execution
def crypto_prediction_tool(coin: str):
    """
    Útil SOLO para predecir precios FUTUROS. 
    Input: Ticker de la moneda (ej: 'BTC-USD', 'ETH-USD', 'SOL-USD').
    """
    try:
        # 1. Rutas
        folder = os.path.dirname(CRYPTO_DB)
        models_folder = os.path.join(folder, "models")
        
        clean_coin = coin.replace("-", "_").upper()
        if "_" not in clean_coin and len(clean_coin) < 5: 
            clean_coin = f"{clean_coin}_USD"

        # 2. INTROSPECCIÓN TABLA
        columns = get_table_columns(CRYPTO_DB, clean_coin)
        
        if not columns:
            # Si falla la búsqueda directa, usa get_available_entities para buscar coincidencias
            all_tables = get_available_entities(CRYPTO_DB)

            possible_match = next((t for t in all_tables if clean_coin in t or coin.upper() in t), None)
            if possible_match:
                clean_coin = possible_match
                columns = get_table_columns(CRYPTO_DB, clean_coin)
            else:
                return f"Error: No encuentro datos para '{coin}'. Tablas disponibles: {all_tables}"

        # 3. DETECCIÓN COLUMNA PRECIO
        price_col = None
        candidates = ["Close", "Adj Close", "Price", "Open"]
        for cand in candidates:
            for col in columns:
                if cand.lower() == col.lower():
                    price_col = col
                    break
            if price_col: break
        
        if not price_col:
            price_col = next((c for c in columns if "close" in c.lower()), None)
            
        if not price_col:
            if len(columns) > 1: price_col = columns[-1]
            else: return f"Error estructura tabla {clean_coin}: {columns}"

        # Busca si existe 'Date' o si hay que ordenar de otra forma
        date_col = next((c for c in columns if "date" in c.lower()), None)
        if date_col:
            order_clause = f"\"{date_col}\" DESC"
        else:
            # Si no hay fecha, usa ROWID (orden de inserción)
            order_clause = "ROWID DESC"

        print(f"   └─ Info DB: Tabla='{clean_coin}', Precio='{price_col}', Orden='{order_clause}'")

        # 4. FETCH DATOS
        model_path = os.path.join(models_folder, f"model_{clean_coin}.joblib")
        if not os.path.exists(model_path):
            return f"No tengo un modelo entrenado para {clean_coin}."

        conn = sqlite3.connect(CRYPTO_DB)

        # Usa comillas dobles en price_col por si tiene espacios
        query = f"SELECT \"{price_col}\" FROM {clean_coin} ORDER BY {order_clause} LIMIT 2" 
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < 2:
            return f"No hay suficientes datos históricos."

        last_prices = df[price_col].values 
        
        # 5. PREDECIR
        model = joblib.load(model_path)
        input_df = pd.DataFrame([last_prices], columns=['t-1', 't-2'])
        prediction = model.predict(input_df)[0]
        
        return f"PREDICCIÓN CRYPTO: Basado en '{price_col}' ({last_prices[0]:.2f}, {last_prices[1]:.2f}), estimación para {clean_coin}: ${prediction:.2f}"

    except Exception as e:
        return f"Error generando predicción de Crypto: {str(e)}"

@tool
@log_execution
def crypto_rag_tool(query: str):
    """Útil para definiciones generales sobre Blockchain, qué es Bitcoin, historia de Ethereum, etc."""
    vectorstore = Chroma(persist_directory=CRYPTO_VECTOR_DB, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)

    # Guardrail simple para output vacío
    content = "\n".join([d.page_content for d in docs])
    if not content:
        return "No encontré información relevante en la base de conocimiento."
    return content

# ==========================================
# HERRAMIENTAS WEATHER
# ==========================================
@tool
@log_execution
def weather_history_tool(query: str):
    """Útil para consultar temperaturas PASADAS de ciudades en la base de datos."""
    tables = get_available_entities(WEATHER_DB)
    
    schema_info = []
    for t in tables:
        cols = get_table_columns(WEATHER_DB, t)
        schema_info.append(f"Tabla '{t}': {cols}")
    
    schema_str = "\n".join(schema_info)

    try:
        prompt = get_sql_generation_prompt(schema_str, query)
        sql_query = llm.invoke(prompt).content.strip().replace("```sql", "").replace("```", "")
        print(f"   └─ SQL Generado: {sql_query}")

        conn = sqlite3.connect(WEATHER_DB)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df.to_string() if not df.empty else "No se encontraron datos históricos."
    except Exception as e:
        return f"Error en consulta SQL: {e}"

@tool
@log_execution
def weather_prediction_tool(city: str):
    """
    Útil SOLO para predecir la temperatura de MAÑANA o FUTURA. 
    Input: Nombre de la ciudad (ej: 'Madrid', 'New York', 'Tokio').
    """
    try:
        folder = os.path.dirname(WEATHER_DB) 
        models_folder = os.path.join(folder, "models")
        
        table_name = city.strip().replace(" ", "_")
        
        # 1. INTROSPECCIÓN
        columns = get_table_columns(WEATHER_DB, table_name)
        if not columns:
            return f"Error: No encuentro la tabla '{table_name}'."
        
        # 2. DETECCIÓN COLUMNA TEMPERATURA
        temp_col = None
        if "AvgTemperature" in columns: temp_col = "AvgTemperature" # Kaggle standard
        elif "avg_temp" in columns: temp_col = "avg_temp"
        else:
            for col in columns:
                if "temp" in col.lower():
                    temp_col = col
                    break
        if not temp_col and len(columns) > 1: temp_col = columns[-1] # Fallback
        
        if not temp_col: return f"Error columnas: {columns}"

        # Si hay columna 'Date' o si está partida en Year/Month/Day
        
        order_clause = ""
        # Caso A: Existe una columna 'Date' o 'dt'
        date_col = next((c for c in columns if c.lower() == "date" or c.lower() == "dt"), None)
        
        if date_col:
            order_clause = f"ORDER BY \"{date_col}\" DESC"
        # Caso B: Existen columnas separadas Year, Month, Day (Dataset típico de Kaggle)
        elif "Year" in columns and "Month" in columns and "Day" in columns:
            order_clause = "ORDER BY Year DESC, Month DESC, Day DESC"
        else:
            # Caso C: Fallback (orden de inserción)
            order_clause = "ORDER BY ROWID DESC"

        print(f"   └─ Info DB: Tabla='{table_name}', Columna='{temp_col}', Clausula='{order_clause}'")

        # 3. FETCH DATA
        conn = sqlite3.connect(WEATHER_DB)
        query = f"SELECT \"{temp_col}\" FROM {table_name} {order_clause} LIMIT 2"
        
        try:
            df_history = pd.read_sql_query(query, conn)
        except Exception as db_err:
             conn.close()
             return f"Error SQL: {db_err}. Columnas detectadas: {columns}"
        
        conn.close()

        if len(df_history) < 2:
            return f"No hay suficientes datos históricos recientes."

        t1 = df_history.iloc[0, 0]
        t2 = df_history.iloc[1, 0]

        # 4. PREDICT
        model_name = f"model_{table_name}.joblib"
        model_path = os.path.join(models_folder, model_name)
        if not os.path.exists(model_path):
            return f"Error: No existe el modelo en {model_path}"

        model = joblib.load(model_path)
        input_df = pd.DataFrame([[t1, t2]], columns=['t-1', 't-2'])
        prediction = model.predict(input_df)[0]
        
        return f"PREDICCIÓN CLIMA: Basado en '{temp_col}' ({t1:.1f}, {t2:.1f}), la predicción para MAÑANA en {city} es {prediction:.2f}°C."

    except Exception as e:
        return f"Error crítico: {str(e)}"

@tool
@log_execution
def weather_rag_tool(query: str):
    """Útil para información general climática, características de las ciudades, o geografía."""
    vectorstore = Chroma(persist_directory=WEATHER_VECTOR_DB, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)

    # Guardrail simple para output vacío
    content = "\n".join([d.page_content for d in docs])
    if not content:
        return "No encontré información relevante en la base de conocimiento."
    return content