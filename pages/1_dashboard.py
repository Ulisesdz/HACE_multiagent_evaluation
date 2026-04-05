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
    /* Ocultar el menú de navegación por defecto de Streamlit */
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


@st.cache_data(ttl=60)  # Cache de 60 segundos para refrescar automáticamente
def load_data():
    """Carga datos de evaluaciones offline"""
    csv_path = Path("evaluation/accumulated_data/offline_metrics.csv")

    if not csv_path.exists():
        st.error(f"No se encontró {csv_path}")
        st.info("Ejecuta primero: `python -m evaluation.baseline.run_eval`")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df_offline = df[df["source"] == "offline"].copy()

    # Convertir columnas numéricas
    numeric_cols = [
        "baseline_score",
        "baseline_time",
        "llm_judge_overall",
        "llm_judge_time",
        "mace_score",
        "mace_time",
        "mace_layer3_used",
        "mace_layer1",
        "mace_layer2",
        "mace_layer3",
    ]

    for col in numeric_cols:
        if col in df_offline.columns:
            df_offline[col] = pd.to_numeric(df_offline[col], errors="coerce")

    return df_offline


# ========== HEADER ==========

st.markdown('<h1 class="main-header">MACE Dashboard</h1>', unsafe_allow_html=True)
st.markdown(
    "**Multi-layered Agent Consensus Evaluator** - Sistema Híbrido de Evaluación"
)

# ========== LOAD DATA ==========

df = load_data()

if df.empty:
    st.stop()

# ========== SIDEBAR FILTERS ==========

with st.sidebar:
    st.title("Comité de Inversión")

    # Botón la app principal
    st.markdown("---")
    if st.button("Volver al Asesor Financiero", use_container_width=True):
        st.switch_page("app.py")

    st.markdown("---")
    st.header("Filtros")

# Filtro de dificultad
difficulties = ["All"] + sorted(df["difficulty"].dropna().unique().tolist())
selected_difficulty = st.sidebar.selectbox("Dificultad", difficulties)

# Filtro de categoría
categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Categoría", categories)

# Filtro de método
method_filter = st.sidebar.multiselect(
    "Métodos a mostrar",
    ["Baseline", "LLM-Judge", "MACE"],
    default=["Baseline", "LLM-Judge", "MACE"],
)

# Aplicar filtros
df_filtered = df.copy()

if selected_difficulty != "All":
    df_filtered = df_filtered[df_filtered["difficulty"] == selected_difficulty]

if selected_category != "All":
    df_filtered = df_filtered[df_filtered["category"] == selected_category]

# ========== MÉTRICAS PRINCIPALES ==========

st.header("Métricas Globales")

col1, col2, col3, col4 = st.columns(4)

# Contar casos por método
baseline_count = df_filtered["baseline_score"].notna().sum()
llm_count = df_filtered["llm_judge_overall"].notna().sum()
mace_count = df_filtered["mace_score"].notna().sum()

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
        label="LLM-Judge (Avg)", value=f"{llm_avg:.3f}", delta=f"{llm_count} casos"
    )

with col4:
    mace_avg = df_filtered["mace_score"].mean() if mace_count > 0 else 0
    st.metric(label="MACE (Avg)", value=f"{mace_avg:.3f}", delta=f"{mace_count} casos")

# ========== TABS ==========
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Comparación Global",
        "MACE Detallado",
        "Análisis de Tiempos",
        "Análisis por Dificultad",
        "Explorador de Datos",
    ]
)

# ========== TAB 1: COMPARACIÓN GLOBAL ==========
with tab1:
    st.subheader("Comparación de Scores (3 Métodos)")

    # Preparar datos para gráfica
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

    if "MACE" in method_filter and mace_count > 0:
        comparison_data.extend(
            [
                {"Método": "MACE", "Score": score}
                for score in df_filtered["mace_score"].dropna()
            ]
        )

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)

        col1, col2 = st.columns(2)

        with col1:
            # Histogram
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
                    "MACE": "#9b59b6",
                },
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, width="stretch")

        with col2:
            # Box plot
            fig_box = px.box(
                df_comparison,
                x="Método",
                y="Score",
                title="Comparación de Scores (Boxplot)",
                color="Método",
                color_discrete_map={
                    "Baseline": "#3498db",
                    "LLM-Judge": "#e74c3c",
                    "MACE": "#9b59b6",
                },
            )
            fig_box.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_box, width="stretch")
    else:
        st.warning("No hay datos disponibles con los filtros actuales.")

# ========== TAB 2: MACE DETALLADO ==========
with tab2:
    st.subheader("MACE - Análisis Detallado")

    if mace_count == 0:
        st.warning("No hay datos de MACE disponibles.")
    else:
        df_mace = df_filtered[df_filtered["mace_score"].notna()].copy()

        # Métricas MACE
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            layer3_usage = (df_mace["mace_layer3_used"].sum() / len(df_mace)) * 100
            st.metric(
                label="Uso de Layer 3",
                value=f"{layer3_usage:.1f}%",
                delta=f"{int(df_mace['mace_layer3_used'].sum())} casos",
            )

        with col2:
            avg_time = df_mace["mace_time"].mean()
            st.metric(label="Tiempo Promedio", value=f"{avg_time:.2f}s")

        with col3:
            high_conf = (df_mace["mace_confidence"] == "high").sum()
            high_conf_pct = (high_conf / len(df_mace)) * 100
            st.metric(
                label="Confianza Alta",
                value=f"{high_conf_pct:.1f}%",
                delta=f"{high_conf} casos",
            )

        with col4:
            excellent = (df_mace["mace_quality"] == "Excelente").sum()
            excellent_pct = (excellent / len(df_mace)) * 100
            st.metric(
                label="Calidad Excelente",
                value=f"{excellent_pct:.1f}%",
                delta=f"{excellent} casos",
            )

        # Gráficas de capas
        col1, col2 = st.columns(2)

        with col1:
            # Distribución de scores por capa
            fig_layers = go.Figure()

            fig_layers.add_trace(
                go.Box(y=df_mace["mace_layer1"], name="Layer 1", marker_color="#3498db")
            )

            fig_layers.add_trace(
                go.Box(y=df_mace["mace_layer2"], name="Layer 2", marker_color="#e74c3c")
            )

            if df_mace["mace_layer3"].notna().any():
                fig_layers.add_trace(
                    go.Box(
                        y=df_mace["mace_layer3"].dropna(),
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
            # Uso de Layer 3
            layer3_counts = df_mace["mace_layer3_used"].value_counts()

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

        # Quality labels
        st.subheader("Distribución de Calidad")

        quality_counts = df_mace["mace_quality"].value_counts()

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

        # 🆕 SECCIÓN: ANÁLISIS DE EMBEDDINGS
        st.markdown("---")
        st.subheader("🧠 Análisis de Embeddings (Layer 2)")

        # Verificar si hay datos de similitud en el CSV
        similarity_cols = [
            "mace_layer2_task_fidelity",
            "mace_layer2_agent_fidelity_avg",
        ]

        # Como el CSV actual no tiene columnas de similitud detallada,
        # vamos a usar los datos que SÍ tenemos: layer2_score

        if "mace_layer2" in df_mace.columns:
            st.markdown("#### 📊 Distribución de Scores Semánticos")

            # Histograma de Layer 2 scores
            fig_sem_dist = px.histogram(
                df_mace,
                x="mace_layer2",
                nbins=20,
                title="Distribución de Scores de Layer 2 (Embeddings)",
                labels={"mace_layer2": "Score Semántico"},
                color_discrete_sequence=["#e74c3c"],
            )

            fig_sem_dist.add_vline(
                x=df_mace["mace_layer2"].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Media: {df_mace['mace_layer2'].mean():.3f}",
            )

            fig_sem_dist.update_layout(height=400)
            st.plotly_chart(fig_sem_dist, use_container_width=True)

            # Estadísticas
            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                st.metric("Media", f"{df_mace['mace_layer2'].mean():.3f}")

            with col_stat2:
                st.metric("Desv. Estándar", f"{df_mace['mace_layer2'].std():.3f}")

            with col_stat3:
                threshold = 0.7  # Threshold típico
                below_threshold = (df_mace["mace_layer2"] < threshold).sum()
                pct_below = (below_threshold / len(df_mace)) * 100
                st.metric(
                    f"Casos < {threshold}",
                    f"{pct_below:.1f}%",
                    help=f"Casos que escalaron a Layer 3 por baja similitud",
                )

        # Correlación Layer 2 vs Layer 3
        if df_mace["mace_layer3"].notna().any():
            st.markdown("#### 🔗 Correlación Layer 2 (Embeddings) vs Layer 3 (LLM)")

            df_with_l3 = df_mace[df_mace["mace_layer3"].notna()].copy()

            fig_corr = px.scatter(
                df_with_l3,
                x="mace_layer2",
                y="mace_layer3",
                title="Layer 2 (Embeddings) vs Layer 3 (LLM-Judge)",
                labels={
                    "mace_layer2": "Score Layer 2 (Embeddings)",
                    "mace_layer3": "Score Layer 3 (LLM-Judge)",
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
                name="y=x",
            )

            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Calcular correlación
            correlation = df_with_l3[["mace_layer2", "mace_layer3"]].corr().iloc[0, 1]

            if correlation > 0.7:
                st.success(
                    f"✅ Alta correlación ({correlation:.3f}): Los embeddings predicen bien el score LLM"
                )
            elif correlation > 0.4:
                st.info(
                    f"ℹ️ Correlación moderada ({correlation:.3f}): Algunas discrepancias entre Layer 2 y 3"
                )
            else:
                st.warning(
                    f"⚠️ Baja correlación ({correlation:.3f}): Embeddings y LLM discrepan frecuentemente"
                )

        # Heatmap de escalación
        st.markdown("#### 🔺 Patrón de Escalación a Layer 3")

        # Crear bins para Layer 2
        df_mace["layer2_bin"] = pd.cut(
            df_mace["mace_layer2"],
            bins=[0, 0.5, 0.6, 0.7, 0.8, 1.0],
            labels=["0.0-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-1.0"],
        )

        escalation_pivot = df_mace.groupby("layer2_bin")["mace_layer3_used"].agg(
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
        st.plotly_chart(fig_escalation, use_container_width=True)

        st.caption(
            "💡 **Insight:** Scores bajos en Layer 2 deberían correlacionar con mayor escalación a Layer 3"
        )

# ========== TAB 3: ANÁLISIS DE TIEMPOS ==========
with tab3:
    st.subheader("Comparación de Tiempos de Evaluación")

    # Calcular tiempos promedio
    times = []

    if baseline_count > 0:
        baseline_time = df_filtered["baseline_time"].mean()
        times.append({"Método": "Baseline", "Tiempo (s)": baseline_time})

    if llm_count > 0:
        llm_time = df_filtered["llm_judge_time"].mean()
        times.append({"Método": "LLM-Judge", "Tiempo (s)": llm_time})

    if mace_count > 0:
        mace_time = df_filtered["mace_time"].mean()
        times.append({"Método": "MACE", "Tiempo (s)": mace_time})

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
                    "MACE": "#9b59b6",
                },
            )

            fig_time.update_layout(showlegend=False)
            st.plotly_chart(fig_time, width="stretch")

        with col2:
            # Speedup calculation
            if len(times) >= 2:
                baseline_t = next(
                    (t["Tiempo (s)"] for t in times if t["Método"] == "Baseline"), None
                )
                llm_t = next(
                    (t["Tiempo (s)"] for t in times if t["Método"] == "LLM-Judge"), None
                )
                mace_t = next(
                    (t["Tiempo (s)"] for t in times if t["Método"] == "MACE"), None
                )

                st.markdown("### Speedup Comparativo")

                if mace_t and llm_t:
                    speedup_llm = ((llm_t - mace_t) / llm_t) * 100
                    st.success(
                        f"✅ MACE es **{speedup_llm:.1f}%** más rápido que LLM-Judge"
                    )

                if mace_t and baseline_t:
                    slowdown = ((mace_t - baseline_t) / baseline_t) * 100
                    if slowdown > 0:
                        if slowdown > 100:
                            multiplier = mace_t / baseline_t
                            st.info(
                                f"ℹ️ MACE es **×{multiplier:.0f} más lento** que Baseline (esperado: Baseline es puramente determinista)"
                            )
                        else:
                            st.info(
                                f"ℹ️ MACE es **{slowdown:.1f}% más lento** que Baseline"
                            )
                    else:
                        st.success(
                            f"✅ MACE es **{abs(slowdown):.1f}% más rápido** que Baseline"
                        )

        # Comparación Layer 3 vs No Layer 3
        if mace_count > 0:
            st.markdown("---")
            st.subheader("Impacto de Layer 3 en Latencia")

            df_mace = df_filtered[df_filtered["mace_score"].notna()].copy()

            no_layer3 = df_mace[df_mace["mace_layer3_used"] == 0]["mace_time"]
            with_layer3 = df_mace[df_mace["mace_layer3_used"] == 1]["mace_time"]

            if len(no_layer3) > 0 and len(with_layer3) > 0:
                fig_layer3_time = go.Figure()

                fig_layer3_time.add_trace(
                    go.Box(y=no_layer3, name="Sin Layer 3", marker_color="#2ecc71")
                )

                fig_layer3_time.add_trace(
                    go.Box(y=with_layer3, name="Con Layer 3", marker_color="#e74c3c")
                )

                fig_layer3_time.update_layout(
                    title="Comparación de Tiempos: Con vs Sin Layer 3",
                    yaxis_title="Tiempo (segundos)",
                    height=400,
                )

                st.plotly_chart(fig_layer3_time, width="stretch")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("⚡ Promedio Sin Layer 3", f"{no_layer3.mean():.3f}s")
                with col2:
                    st.metric("🔺 Promedio Con Layer 3", f"{with_layer3.mean():.3f}s")

# ========== TAB 4: ANÁLISIS POR DIFICULTAD ==========
with tab4:
    st.subheader("Desempeño por Nivel de Dificultad")

    difficulty_order = ["Easy", "Medium", "Hard", "Very Hard"]

    # Preparar datos
    difficulty_data = []

    for diff in difficulty_order:
        df_diff = df_filtered[df_filtered["difficulty"] == diff]

        if "Baseline" in method_filter:
            baseline_avg = df_diff["baseline_score"].mean()
            if not pd.isna(baseline_avg):
                difficulty_data.append(
                    {"Dificultad": diff, "Método": "Baseline", "Score": baseline_avg}
                )

        if "LLM-Judge" in method_filter:
            llm_avg = df_diff["llm_judge_overall"].mean() / 4
            if not pd.isna(llm_avg):
                difficulty_data.append(
                    {"Dificultad": diff, "Método": "LLM-Judge", "Score": llm_avg}
                )

        if "MACE" in method_filter:
            mace_avg = df_diff["mace_score"].mean()
            if not pd.isna(mace_avg):
                difficulty_data.append(
                    {"Dificultad": diff, "Método": "MACE", "Score": mace_avg}
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
                "MACE": "#9b59b6",
            },
            category_orders={"Dificultad": difficulty_order},
        )

        fig_difficulty.update_layout(height=500)
        st.plotly_chart(fig_difficulty, width="stretch")

        # Tabla de estadísticas
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

    # Selección de columnas
    all_columns = df_filtered.columns.tolist()

    default_columns = [
        "query_id",
        "user_query",
        "difficulty",
        "category",
        "baseline_score",
        "llm_judge_overall",
        "mace_score",
        "mace_quality",
        "mace_confidence",
        "mace_layer3_used",
    ]

    available_defaults = [col for col in default_columns if col in all_columns]

    selected_columns = st.multiselect(
        "Selecciona columnas a mostrar:", all_columns, default=available_defaults
    )

    if selected_columns:
        st.dataframe(df_filtered[selected_columns], width="stretch", height=400)

        # Download button
        csv = df_filtered[selected_columns].to_csv(index=False)
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name="mace_filtered_data.csv",
            mime="text/csv",
        )
    else:
        st.warning("Selecciona al menos una columna.")
