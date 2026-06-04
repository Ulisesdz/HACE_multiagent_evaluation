import os
import functools
import sqlite3
from typing import List
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage


def log_execution(func):
    """Decorador para imprimir inputs y outputs de las herramientas (Tools)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Capturar Inputs
        arg_str = ", ".join([str(a) for a in args])
        kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])

        print(f"\n[TOOL START] Ejecutando: {func.__name__}")
        print(f"   -> Inputs: {arg_str} {kwarg_str}")

        try:
            # 2. Ejecutar la función real
            result = func(*args, **kwargs)

            # 3. Capturar Output
            # Corta el log si es muy largo (ej. tabla de datos muy grande)
            log_result = str(result)
            if len(log_result) > 200:
                log_result = log_result[:200] + "... (truncado)"

            print(f"   -> Output: {log_result}")
            print(f"[TOOL END] {func.__name__} finalizada.\n")
            return result

        except Exception as e:
            print(f"   -> Error crítico en Tool: {e}")
            raise e

    return wrapper


def get_table_columns(db_path, table_name):
    """
    Función para Schema Introspection.
    Devuelve una lista con los nombres de las columnas de una tabla SQLite.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # En SQLite, PRAGMA table_info devuelve: (cid, name, type, notnull, dflt_value, pk)
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        conn.close()

        # Extrae solo el nombre (segundo elemento de la tupla)
        column_names = [col[1] for col in columns_info]
        return column_names
    except Exception as e:
        print(f"Error leyendo columnas de {table_name}: {e}")
        return []


def get_available_entities(db_path):
    """
    Devuelve una lista limpia de las tablas en la DB.
    Ejemplo: ['BTC_USD', 'ETH_USD', 'SOL_USD']
    """
    if not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Obtiene nombres de tablas, ignorando tablas internas de sqlite
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []


# FUNCIONES AUXILIARES DE FILTRADO DE MENSAJES
def extract_asset_from_task(task: str) -> str:
    """
    Extrae el nombre del activo (criptomoneda) de una tarea.

    Busca nombres de criptomonedas conocidas en la tarea y devuelve el primero
    que encuentre. Si no encuentra ninguno, devuelve "Activo no especificado".

    Args:
        task: String con la descripción de la tarea.

    Returns:
        Nombre del activo encontrado o "Activo no especificado".
    """
    # Diccionario de mapeo de nombres y aliases
    crypto_names = {
        "bitcoin": "Bitcoin",
        "btc": "Bitcoin",
        "ethereum": "Ethereum",
        "eth": "Ethereum",
        "cardano": "Cardano",
        "ada": "Cardano",
        "solana": "Solana",
        "sol": "Solana",
        "dogecoin": "Dogecoin",
        "doge": "Dogecoin",
        "xrp": "XRP",
        "ripple": "XRP",
        "polkadot": "Polkadot",
        "dot": "Polkadot",
        "binance": "Binance Coin",
        "bnb": "Binance Coin",
    }

    task_lower = task.lower()

    # Buscar coincidencias en el diccionario
    for alias, full_name in crypto_names.items():
        if alias in task_lower:
            return full_name

    return "Activo no especificado"


def get_current_turn_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Filtra el historial de mensajes para obtener solo los del turno actual.

    Retrocede desde el final hasta encontrar el primer mensaje humano que no sea
    una instrucción interna del supervisor, y devuelve todos los mensajes desde
    ese punto hacia adelante.

    Args:
        messages: Lista completa de mensajes del historial.

    Returns:
        Lista de mensajes correspondientes al turno actual.
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage):
            # Ignorar instrucciones internas del supervisor
            if m.name == "supervisor_instruction":
                continue
            return messages[i:]
    return messages


def get_last_user_message(messages: List[BaseMessage]) -> BaseMessage:
    """
    Extrae el último mensaje del usuario del historial.

    Args:
        messages: Lista de mensajes del historial.

    Returns:
        El último mensaje de tipo HumanMessage encontrado, o el último mensaje
        de la lista si no se encuentra ninguno humano.
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return messages[-1]


def extract_reports(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Extrae y consolida los reportes generados por los agentes especialistas.

    Busca mensajes que contengan la marca "### REPORTE" y los agrupa en un único
    mensaje consolidado con metadata sobre qué agentes participaron.

    Args:
        messages: Lista de mensajes del turno actual.

    Returns:
        Lista con un único mensaje consolidado que contiene todos los reportes,
        o lista vacía si no se encontraron reportes.
    """
    reports_text = []
    participating_agents = set()

    # Iterar sobre todos los mensajes buscando reportes
    for m in messages:
        if isinstance(m, AIMessage) and "### REPORTE" in m.content:
            reports_text.append(m.content)

            # Identificar qué agente generó el reporte
            if "TECHNICAL_ANALYST" in m.content:
                participating_agents.add("Technical Analyst")
            elif "FUNDAMENTAL_ANALYST" in m.content:
                participating_agents.add("Fundamental Analyst")
            elif "RISK_OFFICER" in m.content:
                participating_agents.add("Risk Officer")

    # Return de lista vacía si no hay reportes
    if not reports_text:
        return []

    # Consolidar todos los reportes en un único mensaje
    joined = "\n\n".join(reports_text)
    agents_list = ", ".join(participating_agents)

    return [
        HumanMessage(
            content=(
                f"--- METADATA: AGENTES PARTICIPANTES ---\n"
                f"Los siguientes agentes han completado su análisis: {agents_list}\n\n"
                f"--- REPORTES TÉCNICOS: ---\n"
                f"{joined}\n\n"
                f"INSTRUCCIÓN: Genera un informe ejecutivo consolidado en español para el usuario final. "
                f"Usa ÚNICAMENTE los datos de los reportes anteriores. "
                f"Organiza la información por activo (Bitcoin, Dogecoin, etc.) de forma clara y profesional."
            )
        )
    ]
