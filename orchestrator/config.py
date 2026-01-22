import os
import sys

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# --- LLM CONFIG ---
LLM_MODEL = "llama3.1"  # ollama run llama3.1
TEMPERATURE = 0

# --- EMBEDDING CONFIG ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_MODEL_FOLDER = "embedding_model"


# --- PATHS (Rutas relativas desde la raíz del proyecto) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(BASE_DIR, LOCAL_MODEL_FOLDER)

# Crypto Paths
CRYPTO_DB = os.path.join(BASE_DIR, "crypto", "crypto_data.db")
CRYPTO_RAG_TXT = os.path.join(BASE_DIR, "crypto", "RAG_KNOWLEDGE.txt")
CRYPTO_VECTOR_DB = os.path.join(BASE_DIR, "crypto", "chroma_db")

# Weather Paths
WEATHER_DB = os.path.join(BASE_DIR, "weather", "weather_data.db")
WEATHER_RAG_TXT = os.path.join(BASE_DIR, "weather", "RAG_KNOWLEDGE.txt")
WEATHER_VECTOR_DB = os.path.join(BASE_DIR, "weather", "chroma_db")


def get_llm():
    """Devuelve la instancia del LLM local (Ollama)."""
    return ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        base_url="http://localhost:11434"
    )



def get_embeddings():
    """
    Gestión del modelo de Embeddings:
    """
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        # CASO 1: MODO OFFLINE (Ya descargado)
        print(f"[Config] Cargando embeddings desde caché local: {MODEL_PATH}")
        return HuggingFaceEmbeddings(model_name=MODEL_PATH)

    else:
        # CASO 2: PRIMERA EJECUCIÓN (Descarga)
        print(f"[Config] Modelo no encontrado en {MODEL_PATH}.")
        print(f"[Config] Descargando '{EMBEDDING_MODEL_NAME}' desde HuggingFace")

        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            model.save(MODEL_PATH)
            print(f"[Config] Modelo guardado permanentemente en: {MODEL_PATH}")

            return HuggingFaceEmbeddings(model_name=MODEL_PATH)

        except Exception as e:
            print(f"[Config] Error crítico descargando el modelo: {e}")
            print("Asegúrate de tener conexión a internet la primera vez.")
            sys.exit(1)
