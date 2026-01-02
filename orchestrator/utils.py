import os
import functools
import sqlite3

def log_execution(func):
    """Decorador para imprimir inputs y outputs de las herramientas."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Capturar Inputs
        arg_str = ", ".join([str(a) for a in args])
        kwarg_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        
        print(f"\n[TOOL START] Ejecutando: {func.__name__}")
        print(f"   └─ Inputs: {arg_str} {kwarg_str}")
        
        try:
            # 2. Ejecutar la función real
            result = func(*args, **kwargs)
            
            # 3. Capturar Output
            # Cortamos el log si es muy largo (ej. tabla de datos)
            log_result = str(result)
            if len(log_result) > 200:
                log_result = log_result[:200] + "... (truncado)"
            
            print(f"   └─ Output: {log_result}")
            print(f"[TOOL END] {func.__name__} finalizada.\n")
            return result
            
        except Exception as e:
            print(f"   └─ Error: {e}")
            raise e
            
    return wrapper

def get_table_columns(db_path, table_name):
    """
    Función para Schema Introspection.
    Devuelve una lista con los nombres de las columnas de una tabla.
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
        return []
    
def get_available_entities(db_path):
    """
    Devuelve una lista limpia de las tablas en la DB.
    Ejemplo: ['Madrid', 'New_York', 'BTC_USD']
    """
    if not os.path.exists(db_path):
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Obtiene nombres de tablas, ignorando tablas internas de sqlite
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []
    