import pandas as pd
import os

CSV_INPUT = "evaluation/llm_j/dataset_results.csv"
CSV_OUTPUT = "evaluation/hitl/golden_dataset.csv"

def load_data_for_labeling():
    """Carga los resultados del LLM-Juez que aún no han sido revisados por humanos."""
    if not os.path.exists(CSV_INPUT):
        return pd.DataFrame() # Vacío
    
    df_llm = pd.read_csv(CSV_INPUT)
    
    # Si ya existe el golden, carga de lo ya hecho para no repetir
    if os.path.exists(CSV_OUTPUT):
        df_gold = pd.read_csv(CSV_OUTPUT)
        # Filtro IDs que ya están en el gold
        processed_ids = df_gold["id"].tolist()
        df_to_label = df_llm[~df_llm["id"].isin(processed_ids)]
    else:
        df_to_label = df_llm
        
    return df_to_label

def save_human_label(row_data, human_score, human_notes):
    """Guarda una fila corregida en el Golden Dataset."""
    # Campos humanos
    row_data["human_score"] = human_score
    row_data["human_notes"] = human_notes
    
    df_new = pd.DataFrame([row_data])
    
    if not os.path.exists(CSV_OUTPUT):
        df_new.to_csv(CSV_OUTPUT, index=False)
    else:
        df_new.to_csv(CSV_OUTPUT, mode='a', header=False, index=False)