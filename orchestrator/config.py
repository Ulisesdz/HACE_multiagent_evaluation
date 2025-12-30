import os
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# --- LLM CONFIG ---
LLM_MODEL = "llama3.1" # ollama run llama3.1
TEMPERATURE = 0

def get_llm():
    return ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- PATHS (Rutas relativas desde la raíz del proyecto) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Crypto Paths
CRYPTO_DB = os.path.join(BASE_DIR, "crypto", "crypto_data.db")
CRYPTO_RAG_TXT = os.path.join(BASE_DIR, "crypto", "RAG_KNOWLEDGE.txt")
CRYPTO_VECTOR_DB = os.path.join(BASE_DIR, "crypto", "chroma_db")

# Weather Paths
WEATHER_DB = os.path.join(BASE_DIR, "weather", "weather_data.db")
WEATHER_RAG_TXT = os.path.join(BASE_DIR, "weather", "RAG_KNOWLEDGE.txt")
WEATHER_VECTOR_DB = os.path.join(BASE_DIR, "weather", "chroma_db")