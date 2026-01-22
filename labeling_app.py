import streamlit as st
import os
import pandas as pd
from evaluation.hitl.storage import load_data_for_labeling, save_human_label

st.set_page_config(page_title="Herramienta HITL - Labeling", layout="wide")

st.title("HITL: Consola de Etiquetado Humano")
st.markdown("Revisión de evaluaciones automáticas para generar el **Gold Standard**.")

# 1. Cargar datos pendientes
df_pending = load_data_for_labeling()

if df_pending.empty:
    st.success("¡Todo listo! No hay más casos pendientes de revisar en 'evaluation/hitl/dataset.csv'.")
    st.stop()

# 2. Seleccionar el caso actual (el primero disponible)
current_case = df_pending.iloc[0]

# --- VISTA DEL CASO ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Caso ID: {current_case['id']}")
    st.info(f"**Pregunta:** {current_case['question']}")
    
    with st.expander("Ver Contexto Real (Evidence)", expanded=True):
        st.code(current_case['real_context'], language="text")
        
    st.markdown("### Respuesta del Agente")
    st.markdown(f"> {current_case['agent_answer']}")

with col2:
    st.subheader("Opinión del Juez IA")
    
    # Colores para la nota del LLM
    llm_score = current_case['score']
    color = "green" if llm_score == 10 else "orange" if llm_score == 5 else "red"
    
    st.markdown(f"## :{color}[{llm_score}/10]")
    st.caption(f"Tipo Error: {current_case['error_type']}")
    st.write(f"**Razonamiento:** {current_case['judge_analysis']}")
    
    st.divider()
    
    st.subheader("Tu Veredicto (Gold)")
    
    # Formulario para el Humano
    with st.form("human_eval_form"):
        human_score = st.radio(
            "Tu Puntuación:", 
            [10, 5, 0], 
            horizontal=True,
            format_func=lambda x: f"{x} (Perfecto)" if x==10 else f"{x} (Error Proc.)" if x==5 else f"{x} (Alucinación)"
        )
        
        human_notes = st.text_area("Notas / Corrección:", placeholder="¿Por qué estás de acuerdo o en desacuerdo?")
        
        submitted = st.form_submit_button("✅ Guardar y Siguiente")
        
        if submitted:
            save_human_label(current_case.to_dict(), human_score, human_notes)
            st.success("Guardado!")
            st.rerun()
            
# Barra de progreso
total_cases = len(pd.read_csv("evaluation/hitl/dataset.csv")) if os.path.exists("evaluation/hitl/dataset.csv") else 0
done_cases = total_cases - len(df_pending)
st.progress(done_cases / total_cases if total_cases > 0 else 0)
st.caption(f"Progreso: {done_cases}/{total_cases}")