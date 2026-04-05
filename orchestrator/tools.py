import sqlite3
import pandas as pd
import sys
import os
import joblib
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun  # Para búsqueda web

from orchestrator.utils import log_execution, get_table_columns, get_available_entities
from orchestrator.prompts import get_sql_generation_prompt
from orchestrator.config import (
    get_llm,
    get_embeddings,
    CRYPTO_DB,
    CRYPTO_VECTOR_DB,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

llm = get_llm()
embeddings = get_embeddings()

# Búsqueda web
search = DuckDuckGoSearchRun()

# ==========================================
# 1. HERRAMIENTAS DE DATOS (SQL)
# ==========================================


@tool
@log_execution
def crypto_history_tool(query: str):
    """
    Útil para consultar precios PASADOS, volúmenes históricos o datos crudos en la base de datos SQL.
    Uso: Cuando necesites saber el precio de cierre, apertura, o máximos de una fecha específica.
    """

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

        sql_query = (
            llm.invoke(prompt).content.strip().replace("```sql", "").replace("```", "")
        )
        print(f"   └─ SQL Generado: {sql_query}")

        conn = sqlite3.connect(CRYPTO_DB)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        output_str = (
            f"METADATA (Proceso Interno):\n"
            f"SQL Generada: {sql_query}\n"
            f"---------------------------------\n"
            f"RESULTADO DE LA DB:\n"
            f"{df.to_string() if not df.empty else 'No se encontraron datos.'}"
        )
        return output_str

    except Exception as e:
        return f"Error en consulta SQL Crypto: {e}"


# ==========================================
# 2. HERRAMIENTAS DE PREDICCIÓN (ML)
# ==========================================


@tool
@log_execution
def crypto_prediction_tool(coin: str):
    """
    Útil SOLO para predecir precios FUTUROS usando modelos de Machine Learning.
    Input: Ticker de la moneda (ej: 'BTC_USD', 'ETH_USD').
    Devuelve: Una estimación numérica del precio de cierre futuro.
    """
    try:
        # 1. Rutas
        folder = os.path.dirname(CRYPTO_DB)
        models_folder = os.path.join(folder, "models")

        # Normalización del nombre (BTC-USD -> BTC_USD)
        clean_coin = coin.replace("-", "_").upper()
        # Si el usuario pone solo "BTC"
        if "_" not in clean_coin and len(clean_coin) < 5:
            # Búsqueda en las tablas disponibles
            all_tables = get_available_entities(CRYPTO_DB)
            match = next((t for t in all_tables if clean_coin in t), None)
            if match:
                clean_coin = match
            else:
                clean_coin = f"{clean_coin}_USD"

        # 2. OBTENER DATOS RECIENTES (LAGS)
        # últimos 3 precios para alimentar el modelo (t-1, t-2, t-3)
        conn = sqlite3.connect(CRYPTO_DB)

        # Columna de fecha para ordenar
        cols = get_table_columns(CRYPTO_DB, clean_coin)
        date_col = next((c for c in cols if "date" in c.lower()), "Date")
        price_col = next((c for c in cols if "close" in c.lower()), "Close")

        query = (
            f'SELECT "{price_col}" FROM {clean_coin} ORDER BY "{date_col}" DESC LIMIT 3'
        )
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < 3:
            return f"No hay suficientes datos históricos recientes para hacer una predicción."

        last_prices = df[price_col].values  # [t-1, t-2, t-3]

        # 3. CARGAR MODELO Y PREDECIR
        model_path = os.path.join(models_folder, f"model_{clean_coin}.joblib")
        if not os.path.exists(model_path):
            return f"No tengo un modelo entrenado para {clean_coin} en {models_folder}."

        model = joblib.load(model_path)
        # El modelo espera shape (1, 3) -> [[d-1, d-2, d-3]]
        input_df = pd.DataFrame([last_prices], columns=["d-1", "d-2", "d-3"])
        prediction = model.predict(input_df)[0]

        return (
            f"PREDICCIÓN ML para {clean_coin}:\n"
            f"Basado en los últimos cierres ({last_prices}), "
            f"el modelo estima un precio futuro de: ${prediction:.2f}"
        )

    except Exception as e:
        return f"Error generando predicción de Crypto: {str(e)}"


# ==========================================
# 3. HERRAMIENTAS DE RIESGO
# ==========================================


@tool
@log_execution
def crypto_volatility_tool(coin: str):
    """
    Calcula el RIESGO y la VOLATILIDAD de un activo basado en sus últimos 30 días.
    Útil para el Agente de Riesgos.

    Args:
        coin: El ticker del activo (ej: 'BTC_USD', 'ETH_USD').
    """
    try:
        clean_coin = coin.replace("-", "_").upper()

        # Conexión DB
        conn = sqlite3.connect(CRYPTO_DB)
        cols = get_table_columns(CRYPTO_DB, clean_coin)

        # Fallback si no encuentra la tabla exacta
        if not cols:
            all_tables = get_available_entities(CRYPTO_DB)
            match = next((t for t in all_tables if clean_coin in t), None)
            if match:
                clean_coin = match
                cols = get_table_columns(CRYPTO_DB, clean_coin)
            else:
                return f"Error: No encuentro datos para {clean_coin}"

        date_col = next((c for c in cols if "date" in c.lower()), "Date")
        price_col = next((c for c in cols if "close" in c.lower()), "Close")

        # 30 días
        query = f'SELECT "{price_col}" FROM {clean_coin} ORDER BY "{date_col}" DESC LIMIT 30'
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return "No hay datos suficientes para calcular riesgo."

        # CÁLCULO DE VOLATILIDAD
        # 1. Retornos diarios porcentuales
        df["returns"] = df[price_col].pct_change()
        # 2. Desviación estándar de los retornos
        std_dev = df["returns"].std()
        # 3. Volatilidad anualizada (aprox) o mensual
        # Devuelve la desviación como indicador de volatilidad reciente
        volatility_score = std_dev * 100

        # Interpretación básica
        risk_label = "BAJO"
        if volatility_score > 2:
            risk_label = "MEDIO"
        if volatility_score > 5:
            risk_label = "ALTO"
        if volatility_score > 10:
            risk_label = "EXTREMO"

        return (
            f"REPORTE DE RIESGO para {clean_coin}:\n"
            f"- Volatilidad (StdDev 30d): {volatility_score:.2f}%\n"
            f"- Nivel de Riesgo: {risk_label}\n"
            f"- Datos analizados: Últimos {len(df)} registros."
        )

    except Exception as e:
        return f"Error calculando volatilidad: {e}"


# ==========================================
# 4. HERRAMIENTAS DE CONTEXTO (RAG + WEB)
# ==========================================


@tool
@log_execution
def crypto_rag_tool(query: str):
    """
    Consulta la base de conocimiento interna (archivos de texto vectorizados).
    Útil para definiciones, conceptos fundamentales y "qué es" una moneda.
    """
    try:
        vectorstore = Chroma(
            persist_directory=CRYPTO_VECTOR_DB, embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(query)

        content = "\n".join([d.page_content for d in docs])
        if not content:
            return "No encontré información relevante en los documentos internos."
        return f"CONTEXTO INTERNO:\n{content}"
    except Exception as e:
        return f"Error en RAG: {e}"


@tool
@log_execution
def crypto_news_tool(query: str):
    """
    Busca NOTICIAS RECIENTES y datos en vivo en INTERNET.
    Uso: "Noticias Ethereum hoy", "Por qué ha bajado Solana".
    """
    try:
        # DuckDuckGo para buscar en la web
        results = search.invoke(query)
        return f"NOTICIAS WEB ENCONTRADAS:\n{results}"
    except Exception as e:
        return f"Error buscando en internet: {e}"


# ==========================================
# 5. HERRAMIENTA DE VISUALIZACIÓN
# ==========================================


@tool
@log_execution
def crypto_chart_tool(coin: str):
    """
    Genera un GRÁFICO de precios y devuelve la ruta del archivo.
    Útil cuando el usuario pide 'ver' la tendencia o un gráfico.
    """
    try:
        clean_coin = coin.replace("-", "_").upper()
        conn = sqlite3.connect(CRYPTO_DB)

        # Intentar encontrar tabla
        cols = get_table_columns(CRYPTO_DB, clean_coin)
        if not cols:
            all_tables = get_available_entities(CRYPTO_DB)
            clean_coin = next((t for t in all_tables if clean_coin in t), clean_coin)

        date_col = next((c for c in cols if "date" in c.lower()), "Date")
        price_col = next((c for c in cols if "close" in c.lower()), "Close")

        # 60 días para el gráfico
        query = f'SELECT "{date_col}", "{price_col}" FROM {clean_coin} ORDER BY "{date_col}" ASC LIMIT 60'
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return "No hay datos para graficar."

        # Reordena cronológicamente (ASC)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col, ascending=True)

        # Generar Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df[date_col], df[price_col], label=f"{clean_coin} Price", color="blue")
        plt.title(f"Tendencia de Precios: {clean_coin}")
        plt.xlabel("Fecha")
        plt.ylabel("Precio (USD)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Guardar archivo
        # Asegura que existe la carpeta 'static'
        plot_dir = "plots_temp"
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"{plot_dir}/{clean_coin}_chart_{uuid.uuid4().hex[:6]}.png"

        plt.savefig(filename)
        plt.close()

        return f"Gráfico generado correctamente: {filename}"

    except Exception as e:
        return f"Error generando gráfico: {e}"
