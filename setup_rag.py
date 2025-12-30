import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from orchestrator.config import get_embeddings, CRYPTO_RAG_TXT, WEATHER_RAG_TXT, CRYPTO_VECTOR_DB, WEATHER_VECTOR_DB

def create_vector_store(txt_path, db_path, name):
    print(f"--- Procesando {name} ---")
    if not os.path.exists(txt_path):
        print(f"ERROR: No existe {txt_path}")
        return

    loader = TextLoader(txt_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200, # Un poco de solapamiento ayuda a no cortar ideas a la mitad
            separators=["\n\n", "\n", ".", " ", ""] # Intenta cortar por orden de preferencia
        )
    docs = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    Chroma.from_documents(docs, embeddings, persist_directory=db_path)
    print(f"Base vectorial guardada en: {db_path}")

if __name__ == "__main__":
    create_vector_store(CRYPTO_RAG_TXT, CRYPTO_VECTOR_DB, "Crypto RAG")
    create_vector_store(WEATHER_RAG_TXT, WEATHER_VECTOR_DB, "Weather RAG")