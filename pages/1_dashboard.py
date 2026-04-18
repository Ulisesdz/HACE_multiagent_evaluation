import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Configuración de página
st.set_page_config(page_title="AI Investment", page_icon="🏛️", layout="wide")

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: left;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.2rem;
        font-weight: 600;
    }
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ========== DATA LOADING ==========
@st.cache_data(ttl=60)
def load_data(source: str = "offline") -> pd.DataFrame:
    """
    Carga datos de evaluaciones.

    Args:
        source: 'offline' (dataset), 'online' (chat), o 'both' (combinados)
    """
    offline_path = Path("evaluation/accumulated_data/offline_metrics.csv")
    online_path = Path("evaluation/accumulated_data/online_metrics.csv")

    frames = []

    if source in ("offline", "both"):
        if offline_path.exists():
            df_off = pd.read_csv(offline_path)
            df_off = df_off[df_off["source"] == "offline"].copy()
            frames.append(df_off)
        elif source == "offline":
            st.warning(
                "No se encontró `offline_metrics.csv`. "
                "Ejecuta primero: `python -m evaluation.baseline.run_eval`"
            )

    if source in ("online", "both"):
        if online_path.exists():
            df_on = pd.read_csv(online_path)
            df_on = df_on[df_on["source"] == "online"].copy()
            frames.append(df_on)
        elif source == "online":
            st.warning(
                "No se encontró `online_metrics.csv`. "
                "Evalúa algunas consultas desde la interfaz del chat primero."
            )

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    numeric_cols = [
        "baseline_score",
        "baseline_time",
        "llm_judge_overall",
        "llm_judge_time",
        "HACE_score",
        "HACE_time",
        "HACE_layer3_used",
        "HACE_layer1",
        "HACE_layer2",
        "HACE_layer3",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ========== HEADER ==========
st.markdown('<h1 class="main-header">HACE Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Hybrid Agent Consensus Evaluator** - Sistema Híbrido de Evaluación")

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("Comité de Inversión")

    st.markdown("---")
    if st.button("Volver al Asesor Financiero", width="stretch"):
        st.switch_page("app.py")

    st.markdown("---")
    st.header("Fuente de Datos")

    data_source = st.radio(
        "Mostrar evaluaciones:",
        options=["offline", "online", "both"],
        format_func=lambda x: {
            "offline": "Offline (dataset)",
            "online": "Online (chat)",
            "both": "Ambas",
        }[x],
        index=0,
        help=(
            "**Offline:** casos del dataset de evaluación sistemática.\n\n"
            "**Online:** evaluaciones generadas durante conversaciones en la interfaz.\n\n"
            "**Ambas:** combina las dos fuentes."
        ),
    )

    st.markdown("---")
    st.header("Filtros")

# ========== LOAD DATA ==========
df = load_data(source=data_source)

if df.empty:
    st.stop()

# Badge de fuente activa
source_labels = {
    "offline": ("Dataset Offline", "#3498db"),
    "online": ("Evaluaciones Online", "#2ecc71"),
    "both": ("Offline + Online", "#9b59b6"),
}
label, color = source_labels[data_source]
st.markdown(
    f'<span style="background:{color};color:white;padding:4px 12px;'
    f'border-radius:12px;font-size:0.85rem;">{label} — {len(df)} casos</span>',
    unsafe_allow_html=True,
)
st.markdown("")

# ========== FILTROS ==========
difficulties = ["All"] + sorted(df["difficulty"].dropna().unique().tolist())
selected_difficulty = st.sidebar.selectbox("Dificultad", difficulties)

categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Categoría", categories)

method_filter = st.sidebar.multiselect(
    "Métodos a mostrar",
    ["Baseline", "LLM-Judge", "HACE"],
    default=["Baseline", "LLM-Judge", "HACE"],
)

df_filtered = df.copy()

if selected_difficulty != "All":
    df_filtered = df_filtered[df_filtered["difficulty"] == selected_difficulty]

if selected_category != "All":
    df_filtered = df_filtered[df_filtered["category"] == selected_category]

# ========== MÉTRICAS PRINCIPALES ==========
st.header("Métricas Globales")

col1, col2, col3, col4 = st.columns(4)

baseline_count = df_filtered["baseline_score"].notna().sum()
llm_count = df_filtered["llm_judge_overall"].notna().sum()
HACE_count = df_filtered["HACE_score"].notna().sum()

with col1:
    st.metric(
        label="Total Casos Evaluados",
        value=len(df_filtered),
        delta=f"Filtrados de {len(df)}",
    )

with col2:
    baseline_avg = df_filtered["baseline_score"].mean() if baseline_count > 0 else 0
    st.metric(
        label="Baseline (Avg)",
        value=f"{baseline_avg:.3f}",
        delta=f"{baseline_count} casos",
    )

with col3:
    llm_avg = df_filtered["llm_judge_overall"].mean() / 4 if llm_count > 0 else 0
    st.metric(
        label="LLM-Judge (Avg)",
        value=f"{llm_avg:.3f}",
        delta=f"{llm_count} casos",
    )

with col4:
    HACE_avg = df_filtered["HACE_score"].mean() if HACE_count > 0 else 0
    st.metric(
        label="HACE (Avg)",
        value=f"{HACE_avg:.3f}",
        delta=f"{HACE_count} casos",
    )

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Comparación Global",
        "HACE Detallado",
        "Análisis de Tiempos",
        "Análisis por Dificultad",
        "Explorador de Datos",
    ]
)

# ========== TAB 1: COMPARACIÓN GLOBAL ==========
with tab1:
    st.subheader("Comparación de Scores (3 Métodos)")

    comparison_data = []

    if "Baseline" in method_filter and baseline_count > 0:
        comparison_data.extend(
            [
                {"Método": "Baseline", "Score": score}
                for score in df_filtered["baseline_score"].dropna()
            ]
        )

    if "LLM-Judge" in method_filter and llm_count > 0:
        comparison_data.extend(
            [
                {"Método": "LLM-Judge", "Score": score / 4}
                for score in df_filtered["llm_judge_overall"].dropna()
            ]
        )

    if "HACE" in method_filter and HACE_count > 0:
        comparison_data.extend(
            [
                {"Método": "HACE", "Score": score}
                for score in df_filtered["HACE_score"].dropna()
            ]
        )

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)

        col1, col2 = st.columns(2)

        with col1:
            fig_hist = px.histogram(
                df_comparison,
                x="Score",
                color="Método",
                nbins=20,
                title="Distribución de Scores",
                barmode="overlay",
                opacity=0.7,
                color_discrete_map={
                    "Baseline": "#3498db",
                    "LLM-Judge": "#e74c3c",
                    "HACE": "#9b59b6",
                },
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, width="stretch")

        with col2:
            fig_box = px.box(
                df_comparison,
                x="Método",
                y="Score",
                title="Comparación de Scores (Boxplot)",
                color="Método",
                color_discrete_map={
                    "Baseline": "#3498db",
                    "LLM-Judge": "#e74c3c",
                    "HACE": "#9b59b6",
                },
            )
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, width="stretch")
    else:
        st.warning("No hay datos disponibles con los filtros actuales.")

# ========== TAB 2: HACE DETALLADO ==========
with tab2:
    st.subheader("HACE - Análisis Detallado")

    if HACE_count == 0:
        st.warning("No hay datos de HACE disponibles.")
    else:
        df_HACE = df_filtered[df_filtered["HACE_score"].notna()].copy()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            layer3_usage = (df_HACE["HACE_layer3_used"].sum() / len(df_HACE)) * 100
            st.metric(
                label="Uso de Layer 3",
                value=f"{layer3_usage:.1f}%",
                delta=f"{int(df_HACE['HACE_layer3_used'].sum())} casos",
            )

        with col2:
            avg_time = df_HACE["HACE_time"].mean()
            st.metric(label="Tiempo Promedio", value=f"{avg_time:.2f}s")

        with col3:
            high_conf = (df_HACE["HACE_confidence"] == "high").sum()
            high_conf_pct = (high_conf / len(df_HACE)) * 100
            st.metric(
                label="Confianza Alta",
                value=f"{high_conf_pct:.1f}%",
                delta=f"{high_conf} casos",
            )

        with col4:
            excellent = (df_HACE["HACE_quality"] == "Excelente").sum()
            excellent_pct = (excellent / len(df_HACE)) * 100
            st.metric(
                label="Calidad Excelente",
                value=f"{excellent_pct:.1f}%",
                delta=f"{excellent} casos",
            )

        col1, col2 = st.columns(2)

        with col1:
            fig_layers = go.Figure()
            fig_layers.add_trace(
                go.Box(y=df_HACE["HACE_layer1"], name="Layer 1", marker_color="#3498db")
            )
            fig_layers.add_trace(
                go.Box(y=df_HACE["HACE_layer2"], name="Layer 2", marker_color="#e74c3c")
            )
            if df_HACE["HACE_layer3"].notna().any():
                fig_layers.add_trace(
                    go.Box(
                        y=df_HACE["HACE_layer3"].dropna(),
                        name="Layer 3",
                        marker_color="#f39c12",
                    )
                )
            fig_layers.update_layout(
                title="Distribución de Scores por Capa",
                yaxis_title="Score (0-1)",
                height=400,
            )
            st.plotly_chart(fig_layers, width="stretch")

        with col2:
            layer3_counts = df_HACE["HACE_layer3_used"].value_counts()
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=["Sin Layer 3", "Con Layer 3"],
                        values=[layer3_counts.get(0, 0), layer3_counts.get(1, 0)],
                        marker_colors=["#2ecc71", "#e74c3c"],
                        hole=0.4,
                    )
                ]
            )
            fig_pie.update_layout(title="Distribución de Uso de Layer 3", height=400)
            st.plotly_chart(fig_pie, width="stretch")

        st.subheader("Distribución de Calidad")

        quality_counts = df_HACE["HACE_quality"].value_counts()
        fig_quality = px.bar(
            x=quality_counts.index,
            y=quality_counts.values,
            title="Distribución de Quality Labels",
            labels={"x": "Quality Label", "y": "Frecuencia"},
            color=quality_counts.index,
            color_discrete_map={
                "Excelente": "#2ecc71",
                "Bueno": "#3498db",
                "Mejorable": "#f39c12",
                "Crítico": "#e74c3c",
            },
        )
        fig_quality.update_layout(showlegend=False)
        st.plotly_chart(fig_quality, width="stretch")

        st.markdown("---")
        st.subheader("Análisis de Embeddings (Layer 2)")

        if "HACE_layer2" in df_HACE.columns:
            st.markdown("#### Distribución de Scores Semánticos")

            fig_sem_dist = px.histogram(
                df_HACE,
                x="HACE_layer2",
                nbins=20,
                title="Distribución de Scores de Layer 2 (Embeddings)",
                labels={"HACE_layer2": "Score Semántico"},
                color_discrete_sequence=["#e74c3c"],
            )
            fig_sem_dist.add_vline(
                x=df_HACE["HACE_layer2"].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Media: {df_HACE['HACE_layer2'].mean():.3f}",
            )
            fig_sem_dist.update_layout(height=400)
            st.plotly_chart(fig_sem_dist, width="stretch")

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Media", f"{df_HACE['HACE_layer2'].mean():.3f}")
            with col_stat2:
                st.metric("Desv. Estándar", f"{df_HACE['HACE_layer2'].std():.3f}")
            with col_stat3:
                threshold = 0.7
                below_threshold = (df_HACE["HACE_layer2"] < threshold).sum()
                pct_below = (below_threshold / len(df_HACE)) * 100
                st.metric(
                    f"Casos < {threshold}",
                    f"{pct_below:.1f}%",
                    help="Casos que escalaron a Layer 3 por baja similitud",
                )

        if df_HACE["HACE_layer3"].notna().any():
            st.markdown("#### Correlación Layer 2 (Embeddings) vs Layer 3 (LLM)")

            df_with_l3 = df_HACE[df_HACE["HACE_layer3"].notna()].copy()

            fig_corr = px.scatter(
                df_with_l3,
                x="HACE_layer2",
                y="HACE_layer3",
                title="Layer 2 (Embeddings) vs Layer 3 (LLM-Judge)",
                labels={
                    "HACE_layer2": "Score Layer 2 (Embeddings)",
                    "HACE_layer3": "Score Layer 3 (LLM-Judge)",
                },
                trendline="ols",
                color_discrete_sequence=["#9b59b6"],
            )
            fig_corr.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="gray", dash="dash"),
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, width="stretch")

            correlation = df_with_l3[["HACE_layer2", "HACE_layer3"]].corr().iloc[0, 1]
            if correlation > 0.7:
                st.success(
                    f"Alta correlación ({correlation:.3f}): Los embeddings predicen bien el score LLM"
                )
            elif correlation > 0.4:
                st.info(
                    f"Correlación moderada ({correlation:.3f}): Algunas discrepancias entre Layer 2 y 3"
                )
            else:
                st.warning(
                    f"Baja correlación ({correlation:.3f}): Embeddings y LLM discrepan frecuentemente"
                )

        st.markdown("#### Patrón de Escalación a Layer 3")

        df_HACE["layer2_bin"] = pd.cut(
            df_HACE["HACE_layer2"],
            bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
            labels=["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-1.0"],
        )
        escalation_pivot = df_HACE.groupby("layer2_bin")["HACE_layer3_used"].agg(
            ["sum", "count"]
        )
        escalation_pivot["escalation_rate"] = (
            escalation_pivot["sum"] / escalation_pivot["count"]
        ) * 100

        fig_escalation = px.bar(
            escalation_pivot.reset_index(),
            x="layer2_bin",
            y="escalation_rate",
            title="Tasa de Escalación a Layer 3 por Rango de Score Layer 2",
            labels={
                "layer2_bin": "Rango de Score Layer 2",
                "escalation_rate": "Tasa de Escalación (%)",
            },
            color="escalation_rate",
            color_continuous_scale="Reds",
        )
        fig_escalation.update_layout(height=400)
        st.plotly_chart(fig_escalation, width="stretch")
        st.caption(
            "Scores bajos en Layer 2 deberían correlacionar con mayor escalación a Layer 3"
        )

# ========== TAB 3: ANÁLISIS DE TIEMPOS ==========
with tab3:
    st.subheader("Comparación de Tiempos de Evaluación")

    times = []
    if baseline_count > 0:
        times.append(
            {"Método": "Baseline", "Tiempo (s)": df_filtered["baseline_time"].mean()}
        )
    if llm_count > 0:
        times.append(
            {"Método": "LLM-Judge", "Tiempo (s)": df_filtered["llm_judge_time"].mean()}
        )
    if HACE_count > 0:
        times.append({"Método": "HACE", "Tiempo (s)": df_filtered["HACE_time"].mean()})

    if times:
        df_times = pd.DataFrame(times)

        col1, col2 = st.columns(2)

        with col1:
            fig_time = px.bar(
                df_times,
                x="Método",
                y="Tiempo (s)",
                title="Tiempo Promedio por Método",
                color="Método",
                color_discrete_map={
                    "Baseline": "#3498db",
                    "LLM-Judge": "#e74c3c",
                    "HACE": "#9b59b6",
                },
            )
            fig_time.update_layout(showlegend=False)
            st.plotly_chart(fig_time, width="stretch")

        with col2:
            llm_t = next(
                (t["Tiempo (s)"] for t in times if t["Método"] == "LLM-Judge"), None
            )
            HACE_t = next(
                (t["Tiempo (s)"] for t in times if t["Método"] == "HACE"), None
            )
            base_t = next(
                (t["Tiempo (s)"] for t in times if t["Método"] == "Baseline"), None
            )

            st.markdown("### Comparación LLM-Judge vs HACE")
            st.caption(
                "La comparación relativa se hace solo entre LLM-Judge y HACE. "
                "Baseline (~0.001s) tiene una diferencia de magnitud tan grande "
                "que los multiplicadores perderían sentido informativo."
            )

            if llm_t and HACE_t:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("LLM-Judge", f"{llm_t:.2f}s")
                with col_b:
                    if HACE_t < llm_t:
                        ratio = llm_t / HACE_t
                        delta_value = -1.0  # negativo → verde con inverse
                        caption = f"×{ratio:.1f} más rápido"
                    else:
                        ratio = HACE_t / llm_t
                        delta_value = 1.0  # positivo → rojo con inverse
                        caption = f"×{ratio:.1f} más lento"

                    st.metric(
                        label="HACE",
                        value=f"{HACE_t:.2f}s",
                        delta=delta_value,
                        delta_color="inverse",
                    )
                    st.caption(caption)

            if base_t:
                st.markdown("---")
                st.markdown("**Baseline (referencia):**")
                st.metric(
                    label="Baseline",
                    value=f"{base_t:.4f}s",
                    help="Puramente determinista. Se muestra para contexto, no como comparación directa.",
                )

    if HACE_count > 0:
        st.markdown("---")
        st.subheader("Impacto de Layer 3 en Latencia HACE")

        df_HACE_t = df_filtered[df_filtered["HACE_score"].notna()].copy()
        no_layer3 = df_HACE_t[df_HACE_t["HACE_layer3_used"] == 0]["HACE_time"]
        with_layer3 = df_HACE_t[df_HACE_t["HACE_layer3_used"] == 1]["HACE_time"]

        if len(no_layer3) > 0 and len(with_layer3) > 0:
            fig_layer3_time = go.Figure()
            fig_layer3_time.add_trace(
                go.Box(y=no_layer3, name="Sin Layer 3", marker_color="#2ecc71")
            )
            fig_layer3_time.add_trace(
                go.Box(y=with_layer3, name="Con Layer 3", marker_color="#e74c3c")
            )
            fig_layer3_time.update_layout(
                title="Tiempos HACE: Con vs Sin Layer 3",
                yaxis_title="Tiempo (segundos)",
                height=400,
            )
            st.plotly_chart(fig_layer3_time, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Sin Layer 3 (promedio)", f"{no_layer3.mean():.3f}s")
            with c2:
                st.metric("Con Layer 3 (promedio)", f"{with_layer3.mean():.1f}s")

# ========== TAB 4: ANÁLISIS POR DIFICULTAD ==========
with tab4:
    st.subheader("Desempeño por Nivel de Dificultad")

    difficulty_order = ["Easy", "Medium", "Hard", "Very Hard"]
    difficulty_data = []

    for diff in difficulty_order:
        df_diff = df_filtered[df_filtered["difficulty"] == diff]

        if "Baseline" in method_filter:
            v = df_diff["baseline_score"].mean()
            if not pd.isna(v):
                difficulty_data.append(
                    {"Dificultad": diff, "Método": "Baseline", "Score": v}
                )

        if "LLM-Judge" in method_filter:
            v = df_diff["llm_judge_overall"].mean() / 4
            if not pd.isna(v):
                difficulty_data.append(
                    {"Dificultad": diff, "Método": "LLM-Judge", "Score": v}
                )

        if "HACE" in method_filter:
            v = df_diff["HACE_score"].mean()
            if not pd.isna(v):
                difficulty_data.append(
                    {"Dificultad": diff, "Método": "HACE", "Score": v}
                )

    if difficulty_data:
        df_difficulty = pd.DataFrame(difficulty_data)

        fig_difficulty = px.bar(
            df_difficulty,
            x="Dificultad",
            y="Score",
            color="Método",
            barmode="group",
            title="Score Promedio por Dificultad",
            color_discrete_map={
                "Baseline": "#3498db",
                "LLM-Judge": "#e74c3c",
                "HACE": "#9b59b6",
            },
            category_orders={"Dificultad": difficulty_order},
        )
        fig_difficulty.update_layout(height=500)
        st.plotly_chart(fig_difficulty, width="stretch")

        st.subheader("Tabla de Estadísticas")
        pivot_table = df_difficulty.pivot(
            index="Dificultad", columns="Método", values="Score"
        ).reindex(difficulty_order)
        st.dataframe(
            pivot_table.style.format("{:.3f}").background_gradient(
                cmap="RdYlGn", axis=None
            )
        )

# ========== TAB 5: EXPLORADOR DE DATOS ==========
with tab5:
    st.subheader("Explorador de Datos Raw")

    all_columns = df_filtered.columns.tolist()

    default_columns = [
        "query_id",
        "user_query",
        "difficulty",
        "category",
        "baseline_score",
        "llm_judge_overall",
        "HACE_score",
        "HACE_quality",
        "HACE_confidence",
        "HACE_layer3_used",
    ]
    available_defaults = [col for col in default_columns if col in all_columns]

    selected_columns = st.multiselect(
        "Selecciona columnas a mostrar:", all_columns, default=available_defaults
    )

    if selected_columns:
        st.dataframe(df_filtered[selected_columns], width="stretch", height=400)

        csv = df_filtered[selected_columns].to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name="HACE_filtered_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("Selecciona al menos una columna.")
