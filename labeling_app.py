import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from evaluation.hitl.storage import load_data_for_labeling, save_human_label

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="HITL & Analytics Hub", layout="wide", page_icon="⚖️")

# Estilos CSS
st.markdown("""
<style>
    .card-routing {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #007bff;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #fafafa;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        color: #0068c9;
    }
</style>
""", unsafe_allow_html=True)

# --- NAVEGACIÓN ---
with st.sidebar:
    st.title("🎛️ Panel de Control")
    mode = st.radio("Selecciona Módulo:", ["🏷️ Etiquetado (HITL)", "📊 Dashboard Analítico"])
    st.divider()

# ==============================================================================
# MÓDULO 1: ETIQUETADO (HITL)
# ==============================================================================
if mode == "🏷️ Etiquetado (HITL)":
    st.title("Consola de Etiquetado Humano")
    st.markdown("Revisa las decisiones del **Juez IA** y genera el Gold Standard.")

    # 1. Cargar datos pendientes
    df_pending = load_data_for_labeling()

    if df_pending.empty:
        st.success("¡Misión Cumplida! No hay casos pendientes.")
        st.info("Ve a la pestaña 'Dashboard Analítico' para ver los resultados.")
        st.stop()

    # 2. Seleccionar caso
    current_case = df_pending.iloc[0]

    col_details, col_verdict = st.columns([2, 1])

    with col_details:
        st.subheader(f"🆔 Caso: {current_case.get('id', 'N/A')}")
        st.markdown(f"**Pregunta del Inversor:**")
        st.info(f"🗣️ {current_case['question']}")
        
        agent_sel = current_case.get('agent_selected', 'Desconocido')
        st.markdown(f"""<div class="card-routing"><strong>📡 Supervisor Eligió:</strong> <code>{agent_sel}</code></div>""", unsafe_allow_html=True)
        
        st.markdown(f"**Comportamiento Esperado:**")
        st.caption(current_case.get('expected', 'No especificado'))

        st.markdown("### 📝 Respuesta del Agente")
        st.success(f"{current_case['agent_answer']}")
        
        ctx_content = current_case.get('real_context', current_case.get('context', 'No context saved'))
        with st.expander("🔍 Ver Evidencia Técnica (Tool Outputs)", expanded=False):
            st.code(ctx_content, language="json")

    with col_verdict:
        st.markdown("### 🤖 Opinión del Juez IA")
        
        try:
            llm_score = int(float(current_case['score']))
        except:
            llm_score = 0
            
        error_type = current_case.get('error_type', 'Unknown')
        
        if llm_score == 10:
            st.success(f"## ✅ {llm_score}/10")
        elif llm_score >= 5:
            st.warning(f"## ⚠️ {llm_score}/10")
        else:
            st.error(f"## 🚨 {llm_score}/10")
            
        st.markdown(f"**Tipo de Error:** `{error_type}`")
        st.write(f"**Análisis:** {current_case['judge_analysis']}")
        
        st.divider()
        
        st.subheader("Tu Veredicto")
        
        with st.form("human_eval_form"):
            human_score = st.radio(
                "Puntuación Correcta:", 
                [10, 5, 0], 
                horizontal=True,
                format_func=lambda x: "✅ 10" if x==10 else "⚠️ 5" if x==5 else "❌ 0"
            )
            
            human_notes = st.text_area("Notas:", placeholder="Justificación...")
            
            if st.form_submit_button("💾 Guardar", type="primary"):
                save_human_label(current_case.to_dict(), human_score, human_notes)
                st.toast("Guardado!", icon="✅")
                st.rerun()

    try:
        total_cases = len(pd.read_csv("evaluation/llm_j/dataset_results.csv"))
        done_cases = total_cases - len(df_pending)
        progress_val = done_cases / total_cases if total_cases > 0 else 0
    except:
        progress_val = 0
        done_cases = 0
        total_cases = 0

    st.progress(progress_val)
    st.caption(f"Progreso: {done_cases}/{total_cases}")


# ==============================================================================
# MÓDULO 2: DASHBOARD ANALÍTICO
# ==============================================================================
elif mode == "📊 Dashboard Analítico":
    st.title("Reporte de Alineación Humano vs IA")
    
    csv_path = "evaluation/hitl/golden_dataset.csv"
    if not os.path.exists(csv_path):
        st.warning("Aún no has etiquetado datos. Ve a la pestaña de Etiquetado primero.")
        st.stop()
        
    df = pd.read_csv(csv_path)
    
    if len(df) < 5:
        st.info(f"Tienes pocos datos etiquetados ({len(df)}). Etiqueta más para ver métricas fiables.")
    
    # --- CÁLCULO DE MÉTRICAS ---
    y_human = df['human_score']
    y_ai = df['score']
    
    acc = accuracy_score(y_human, y_ai)
    kappa = cohen_kappa_score(y_human, y_ai, labels=[0, 5, 10])
    
    # 1. KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Casos Evaluados", len(df))
    with col2:
        st.metric("Exact Match (Accuracy)", f"{acc:.2%}")
    with col3:
        delta_color = "normal"
        if kappa > 0.8: delta_color = "normal" # Verde por defecto
        elif kappa < 0.4: delta_color = "inverse" # Rojo
        
        st.metric("Cohen's Kappa", f"{kappa:.3f}", 
                  help="<0.4: Pobre | 0.4-0.6: Moderado | >0.8: Excelente")

    st.divider()

    # 2. VISUALIZACIONES
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Matriz de Confusión")
        st.markdown("Compara tu nota (Eje Y) vs la nota de la IA (Eje X).")
        
        labels = [0, 5, 10]
        cm = confusion_matrix(y_human, y_ai, labels=labels)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Puntuación IA')
        ax.set_ylabel('Puntuación Humana')
        st.pyplot(fig)
        
    with c2:
        st.subheader("📉 Interpretación")
        if kappa > 0.75:
            st.success("**Alineación Excelente:** El Juez IA piensa casi igual que tú.")
        elif kappa > 0.4:
            st.warning("**Alineación Moderada:** Hay discrepancias. Revisa si la IA es demasiado estricta.")
        else:
            st.error("**Alineación Pobre:** El criterio del Juez IA es muy diferente al tuyo. Revisa el Prompt del Juez.")
            
        st.info("""
        **Cómo leer la matriz:**
        * **Diagonal:** Casos donde ambos coinciden.
        * **Arriba-Derecha:** La IA dio más nota que tú (IA sobreestima).
        * **Abajo-Izquierda:** La IA dio menos nota que tú (IA castiga más).
        """)

    # 3. TABLA DE DISCREPANCIAS
    st.divider()
    st.subheader("🚩 Discrepancias Destacadas")
    st.markdown("Casos donde el Humano y la IA no estuvieron de acuerdo.")
    
    disagreements = df[df['human_score'] != df['score']].copy()
    
    if not disagreements.empty:
        # Formateo para la tabla
        disagreements = disagreements[['id', 'question', 'human_score', 'score', 'human_notes', 'agent_selected']]
        disagreements.columns = ['ID', 'Pregunta', 'Tu Nota', 'Nota IA', 'Tus Notas', 'Routing']
        
        # Función para colorear
        def highlight_diff(row):
            colors = ['' for _ in row]
            if row['Tu Nota'] > row['Nota IA']:
                # IA Castigó demasiado
                colors[3] = 'background-color: #ffcccc' # Rojo suave en nota IA
            else:
                # IA Regaló nota
                colors[3] = 'background-color: #ccffcc' # Verde suave en nota IA
            return colors

        st.dataframe(disagreements.style.apply(highlight_diff, axis=1), use_container_width=True)
    else:
        st.success("¡Increíble! No hay discrepancias. Alineación perfecta del 100%.")