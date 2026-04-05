import os
import re
import time
import streamlit as st
import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from evaluation.baseline.evaluator import evaluate_baseline
from evaluation.metrics_accumulator.logger import MetricsLogger
from evaluation.hybrid import HybridEvaluator

from evaluation.llm_j.judge import (
    evaluate_planner,
    evaluate_supervisor,
    evaluate_agent,
    evaluate_final_output,
    evaluate_comprehensive,
)
from orchestrator.graph import build_graph

# --- CONFIGURACIÓN ML FLOW ---
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("TFG_Financial_Agent")
mlflow.langchain.autolog(disable=False)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Investment", page_icon="🏛️", layout="wide")

# --- ESTILOS CSS ---
st.markdown(
    """
<style>
    .stChatMessage {border-radius: 10px; background-color: #f0f2f6;}
    .stStatus {border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px;}
    
    /* Estilos para las tarjetas de Reporte de Agentes */
    .agent-card {
        border-left: 5px solid #ccc;
        background-color: #f9f9f9;
        padding: 12px;
        margin-top: 10px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-size: 0.95rem;
    }
    .card-tech { border-left-color: #0068c9; background-color: #f0f7ff; }
    .card-fund { border-left-color: #00cc96; background-color: #f0fff8; }
    .card-risk { border-left-color: #ff2b2b; background-color: #fff5f5; }
    
    .card-title {
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        color: #333;
    }
    
    /* Estilos para el panel de auditoría */
    .audit-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    .baseline-panel {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    .metric-card {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-title {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    
    .score-excellent { color: #00ff88; }
    .score-good { color: #ffeb3b; }
    .score-fair { color: #ff9800; }
    .score-poor { color: #ff5252; }
            
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


# --- INICIALIZACIÓN ---
@st.cache_resource
def load_app():
    return build_graph()


app = load_app()

# Inicializar MetricsLogger
if "metrics_logger" not in st.session_state:
    st.session_state.metrics_logger = MetricsLogger()


# ============================================================================
# TRACE COLLECTOR (Para capturar datos del pipeline)
# ============================================================================
class TraceCollector:
    """Colector simplificado de trazas para la UI"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.planner_tasks = []
        self.routing_trace = []
        self.agent_executions = []
        self.sql_queries = []
        self.final_answer = ""
        self.user_question = ""

    def capture_planner(self, state: dict):
        """Captura las tareas del Planner"""
        self.planner_tasks = state.get("pending_tasks", [])

    def capture_supervisor_decision(self, state: dict):
        """Captura decisión de routing"""
        current_task = state.get("current_task", "")
        next_agent = state.get("next", "")

        if next_agent != "FINISH" and current_task:
            self.routing_trace.append({"task": current_task, "agent": next_agent})

    def capture_agent_execution(
        self, agent_name: str, messages: list, current_task: str = ""
    ):
        """Captura ejecución del agente"""
        tools_used = []
        tool_outputs_text = []
        agent_response = ""

        for msg in messages:
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tools_used.append(tool_call.get("name", "unknown"))

                if msg.content and not msg.tool_calls:
                    agent_response = msg.content

            elif isinstance(msg, ToolMessage):
                tool_outputs_text.append(f"[{msg.name}]: {msg.content}")

        if tools_used or agent_response:
            self.agent_executions.append(
                {
                    "agent": agent_name,
                    "task": current_task,
                    "tools_used": tools_used,
                    "tool_outputs": "\n".join(tool_outputs_text),
                    "agent_response": agent_response,
                }
            )

    def capture_sql_query(self, task: str, tool_call_args: dict):
        """Captura queries SQL ejecutadas"""
        if "query" in tool_call_args:
            self.sql_queries.append({"task": task, "sql": tool_call_args["query"]})


# ============================================================================
# FUNCIONES DE RENDERIZADO DE EVALUACIONES
# ============================================================================
def render_llm_judge_panel(llm_judge_data):
    """Renderiza el panel de LLM-Judge (escala 1-4)"""
    comprehensive_eval = llm_judge_data["comprehensive_eval"]
    planner_eval = llm_judge_data["planner_eval"]
    planner_score = llm_judge_data["planner_score"]
    supervisor_eval = llm_judge_data["supervisor_eval"]
    supervisor_score = llm_judge_data["supervisor_score"]
    agents_eval = llm_judge_data["agents_eval"]
    agents_avg_score = llm_judge_data["agents_avg_score"]
    final_eval = llm_judge_data["final_eval"]
    final_output_score = llm_judge_data["final_output_score"]

    # Panel de LLM-Judge (escala 1-4)
    overall_score = comprehensive_eval.overall_score

    # Nuevos umbrales para escala 1-4
    if overall_score >= 3.5:
        score_class = "score-excellent"
        score_label = "EXCELENTE"
    elif overall_score >= 2.5:
        score_class = "score-good"
        score_label = "BUENO"
    elif overall_score >= 1.5:
        score_class = "score-fair"
        score_label = "MEJORABLE"
    else:
        score_class = "score-poor"
        score_label = "CRÍTICO"

    st.markdown(
        f"""
        <div class="audit-panel">
            <h3 style="margin-top: 0; text-align: center">LLM-Judge (Evaluación Cualitativa)</h3>
            <div style="text-align: center; margin: 20px 0;">
                <div class="metric-value {score_class}">{overall_score:.2f}/4</div>
                <div style="font-size: 1.2rem; margin-top: 10px; opacity: 0.9;">{score_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Métricas por Módulo (escala 1-4)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="📋 Planner", value=f"{planner_score:.1f}/4")
        with st.expander("📊 Detalles"):
            st.write(f"**Correctitud:** {planner_eval.correctness}/4")
            st.write(f"**Completitud:** {planner_eval.completeness}/4")
            st.write(f"**Precisión:** {planner_eval.precision}/4")
            st.write(f"**Descomposición:** {planner_eval.task_decomposition}/4")

    with col2:
        st.metric(label="🎯 Supervisor", value=f"{supervisor_score:.1f}/4")
        with st.expander("📊 Detalles"):
            st.write(f"**Routing Accuracy:** {supervisor_eval.routing_accuracy}/4")
            st.write(f"**Task Completion:** {supervisor_eval.task_completion}/4")

    with col3:
        st.metric(label="⚙️ Agentes", value=f"{agents_avg_score:.1f}/4")
        with st.expander("📊 Detalles"):
            for agent_eval in agents_eval:
                st.write(f"**{agent_eval.agent_name}:**")
                st.write(f"  • Tool Selection: {agent_eval.tool_selection}/4")
                st.write(f"  • Tool Execution: {agent_eval.tool_execution}/4")
                st.write(f"  • Output Fidelity: {agent_eval.output_fidelity}/4")
                st.write(f"  • Completeness: {agent_eval.output_completeness}/4")
                st.write(f"  • Hallucination: {agent_eval.hallucination_check}/4")

    with col4:
        st.metric(label="📄 Output Final", value=f"{final_output_score:.1f}/4")
        with st.expander("📊 Detalles"):
            st.write(f"**Completitud:** {final_eval.completeness}/4")
            st.write(f"**Precisión:** {final_eval.accuracy}/4")
            st.write(f"**Estructura:** {final_eval.structure}/4")
            st.write(f"**Chart Attribution:** {final_eval.chart_attribution}/4")

    st.markdown("### 📝 Resumen Ejecutivo")
    st.info(comprehensive_eval.executive_summary)

    if comprehensive_eval.error_category != "None":
        st.warning(f"**Categoría de Error:** {comprehensive_eval.error_category}")

    with st.expander("🔍 Ver Análisis Detallado"):
        st.markdown("#### Análisis del Planner")
        st.write(planner_eval.analysis)
        st.markdown("#### Análisis del Supervisor")
        st.write(supervisor_eval.analysis)
        st.markdown("#### Análisis de Agentes")
        for agent_eval in agents_eval:
            st.write(f"**{agent_eval.agent_name}:** {agent_eval.analysis}")
        st.markdown("#### Análisis del Output Final")
        st.write(final_eval.analysis)


def render_baseline_panel(baseline_eval):
    """Renderiza el panel de Baseline Metrics"""
    st.markdown(
        f"""
        <div class="baseline-panel">
            <h3 style="margin-top: 0; text-align: center">Baseline Metrics (Automáticas)</h3>
            <div style="text-align: center; margin: 20px 0;">
                <div style="font-size: 2rem; font-weight: bold;">{baseline_eval.baseline_score:.3f}</div>
                <div style="font-size: 1rem; margin-top: 10px; opacity: 0.9;">Score Global (0-1)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Métricas Detalladas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Routing F1", value=f"{baseline_eval.routing_metrics.f1:.3f}"
        )
        with st.expander("📊 Detalles"):
            st.write(f"**Accuracy:** {baseline_eval.routing_metrics.accuracy:.3f}")
            st.write(f"**Precision:** {baseline_eval.routing_metrics.precision:.3f}")
            st.write(f"**Recall:** {baseline_eval.routing_metrics.recall:.3f}")
            if baseline_eval.routing_metrics.per_class:
                st.write("**Por Agente:**")
                for agent, metrics in baseline_eval.routing_metrics.per_class.items():
                    st.write(f"  • {agent}: F1={metrics['f1']:.3f}")

    with col2:
        halluc_rate = baseline_eval.numeric_metrics.hallucination_rate
        st.metric(
            label="🔢 Numeric F1", value=f"{baseline_eval.numeric_metrics.f1:.3f}"
        )
        with st.expander("📊 Detalles"):
            st.write(f"**Precision:** {baseline_eval.numeric_metrics.precision:.3f}")
            st.write(f"**Recall:** {baseline_eval.numeric_metrics.recall:.3f}")
            st.write(f"**Hallucination Rate:** {halluc_rate:.1%}")
            if halluc_rate > 0.1:
                st.warning("⚠️ Alta tasa de alucinaciones (>10%)")

    with col3:
        st.metric(
            label="✅ Task Coverage",
            value=f"{baseline_eval.task_coverage_metrics.coverage:.1%}",
        )
        with st.expander("📊 Detalles"):
            st.write(
                f"**Planificadas:** {baseline_eval.task_coverage_metrics.planned_tasks}"
            )
            st.write(
                f"**Completadas:** {baseline_eval.task_coverage_metrics.completed_tasks}"
            )
            st.write(
                f"**Omisión:** {baseline_eval.task_coverage_metrics.omission_rate:.1%}"
            )

    with col4:
        st.metric(
            label="💾 SQL Correctness",
            value=f"{baseline_eval.sql_metrics.correctness:.1%}",
        )
        with st.expander("📊 Detalles"):
            st.write(
                f"**Queries Evaluadas:** {baseline_eval.sql_metrics.total_queries}"
            )
            st.write(
                f"**Queries Correctas:** {baseline_eval.sql_metrics.correct_queries}"
            )
            if baseline_eval.sql_metrics.violations:
                st.warning("**Violaciones SQL:**")
                for violation in baseline_eval.sql_metrics.violations[:3]:
                    st.write(f"• {violation}")


def render_hybrid_panel(hybrid_eval):
    """
    Renderiza el panel de MACE (Hybrid Evaluation) con DETALLES COMPLETOS

    Args:
        hybrid_eval: Dict resultado de HybridEvaluator.evaluate()
    """
    # ========== HEADER PRINCIPAL ==========
    final_score = float(hybrid_eval.get("final_score", 0))
    quality_label = str(hybrid_eval.get("quality_label", "N/A"))
    confidence = str(hybrid_eval.get("confidence", "N/A"))

    # Mapeo de colores por calidad
    quality_colors = {
        "Excelente": "score-excellent",
        "Bueno": "score-good",
        "Mejorable": "score-fair",
        "Crítico": "score-poor",
    }
    score_class = quality_colors.get(quality_label, "score-good")

    st.markdown(
        f"""
        <div class="audit-panel" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3 style="margin-top: 0; text-align: center">MACE - Hybrid Evaluation</h3>
            <div style="text-align: center; margin: 20px 0;">
                <div class="metric-value {score_class}">{final_score:.3f}</div>
                <div style="font-size: 1.2rem; margin-top: 10px; opacity: 0.9;">{quality_label}</div>
                <div style="font-size: 0.9rem; margin-top: 5px; opacity: 0.8;">Confianza: {confidence.upper()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========== MÉTRICAS POR CAPA ==========
    st.markdown("### 📊 Scores por Capa")

    col1, col2, col3 = st.columns(3)

    layer1_score = float(hybrid_eval.get("layer1_score", 0))
    layer2_score = float(hybrid_eval.get("layer2_score", 0))
    layer3_score = hybrid_eval.get("layer3_score")
    layer3_used = bool(hybrid_eval.get("layer3_used", False))

    with col1:
        st.metric(
            label="🛡️ Layer 1 (Guardrails)",
            value=f"{layer1_score:.3f}",
            help="Validadores deterministas de reglas de negocio",
        )
        st.caption(f"⏱️ {float(hybrid_eval.get('layer1_time', 0)):.4f}s")

    with col2:
        st.metric(
            label="🧠 Layer 2 (Semantic)",
            value=f"{layer2_score:.3f}",
            help="Similitud semántica con embeddings",
        )
        st.caption(f"⏱️ {float(hybrid_eval.get('layer2_time', 0)):.3f}s")

    with col3:
        if layer3_used and layer3_score is not None:
            st.metric(
                label="⚖️ Layer 3 (LLM-Judge)",
                value=f"{float(layer3_score):.3f}",
                help="Análisis profundo con LLM",
            )
            st.caption(f"⏱️ {float(hybrid_eval.get('layer3_time', 0)):.2f}s")
        else:
            st.metric(
                label="⚖️ Layer 3 (LLM-Judge)",
                value="N/A",
                delta="No usado",
                help="Caso resuelto sin necesidad de LLM",
            )
            st.caption("✅ Evaluación rápida")

    # ========== ANÁLISIS DEL SISTEMA ==========
    st.markdown("---")
    st.markdown("### 📈 Análisis del Sistema")

    col_meta1, col_meta2, col_meta3 = st.columns(3)

    with col_meta1:
        eval_time = float(hybrid_eval.get("evaluation_time", 0))
        st.metric(label="⏱️ Tiempo Total", value=f"{eval_time:.3f}s")

        # Comparación con LLM-Judge
        if layer3_used:
            st.caption("Evaluación profunda (3 capas)")
        else:
            st.caption("Evaluación rápida (2 capas)")

    with col_meta2:
        # Speedup vs LLM-Judge
        llm_judge_baseline_time = 180.0  # Tiempo promedio de LLM-Judge

        if eval_time < llm_judge_baseline_time:
            speedup = (
                (llm_judge_baseline_time - eval_time) / llm_judge_baseline_time
            ) * 100
            st.metric(
                label="🚀 Speedup vs LLM-Judge",
                value=f"{speedup:.1f}%",
                delta="Más rápido",
            )
        else:
            st.metric(label="⚡ Vs LLM-Judge", value="Similar", delta=None)

    with col_meta3:
        # Razón de escalación
        if layer3_used:
            escalation_reason = hybrid_eval.get("escalation_reason", "Unknown")
            st.warning(f"**🔺 Escalado a Layer 3**")
            st.caption(f"Razón: {escalation_reason}")
        else:
            st.success("**✅ Resolución Rápida**")
            st.caption("Capas 1-2 fueron suficientes")

    # ========== DETALLES DE LAYER 1: GUARDRAILS ==========
    st.markdown("---")
    st.markdown("### Layer 1: Guardrails (Validadores Deterministas)")

    layer1_details = hybrid_eval.get("layer1_details", {})

    if layer1_details:
        validator_names = list(layer1_details.keys())

        if validator_names:
            # --- Tarjetas de resumen (ticks/cruces) ---
            num_cols = min(len(validator_names), 3)
            cols_l1 = st.columns(num_cols)

            for idx, (validator_name, result) in enumerate(layer1_details.items()):
                col_idx = idx % num_cols
                with cols_l1[col_idx]:
                    status = result.get("pass", False)
                    status_icon = "✅" if status else "❌"
                    status_color = "#2ecc71" if status else "#e74c3c"
                    display_name = (
                        validator_name.replace("validate_", "")
                        .replace("_", " ")
                        .title()
                    )
                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid {status_color}; padding: 10px; margin: 5px 0;
                                    background-color: rgba(0,0,0,0.02); border-radius: 4px;">
                            <div style="font-weight: bold;">{status_icon} {display_name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # --- Expander con detalles completos ---
            with st.expander("🔍 Ver Detalles Completos de Layer 1"):
                for validator_name, result in layer1_details.items():
                    status = result.get("pass", False)
                    status_icon = "✅" if status else "❌"
                    display_name = (
                        validator_name.replace("validate_", "")
                        .replace("_", " ")
                        .title()
                    )

                    st.markdown(f"#### {status_icon} {display_name}")

                    if validator_name == "validate_completeness":
                        coverage = result.get("coverage_rate")
                        if coverage is not None:
                            st.metric("Cobertura de Tareas", f"{float(coverage):.1%}")

                        planned = result.get("planned_count", "N/A")
                        executed = result.get("executed_count", "N/A")
                        st.write(
                            f"**Tareas planificadas:** {planned} | **Ejecuciones capturadas:** {executed}"
                        )

                        missing = result.get("missing_tasks", [])
                        if missing:
                            st.warning(
                                f"**Tareas sin ejecución detectada:** {', '.join(missing)}"
                            )
                        else:
                            st.success(
                                "Todas las tareas tienen ejecución correspondiente."
                            )

                        reason = result.get("reason")
                        if reason:
                            st.caption(f"ℹ️ {reason}")

                    elif validator_name == "validate_routing_syntax":
                        total = result.get("total_decisions", 0)
                        st.write(f"**Decisiones de routing evaluadas:** {total}")

                        invalid = result.get("invalid_routings", [])
                        if invalid:
                            st.error(f"**{len(invalid)} routing(s) inválido(s):**")
                            for inv in invalid:
                                st.write(
                                    f"  • Índice {inv.get('index', '?')} | "
                                    f"Tarea: `{inv.get('task', 'N/A')}` → "
                                    f"Agente inválido: `{inv.get('invalid_agent', 'N/A')}`"
                                )
                        else:
                            st.success("Todos los routings usan agentes válidos.")

                    elif validator_name == "validate_numeric_ranges":
                        total_checked = result.get("total_numbers_checked", 0)
                        st.write(f"**Números verificados:** {total_checked}")

                        anomalies = result.get("anomalies", [])
                        if anomalies:
                            st.error(f"**{len(anomalies)} anomalía(s) detectada(s):**")
                            for anomaly in anomalies[:5]:
                                st.write(
                                    f"  • Agente: `{anomaly.get('agent', 'N/A')}` | "
                                    f"Valor: `{anomaly.get('value', 'N/A')}` | "
                                    f"Razón: {anomaly.get('reason', 'N/A')}"
                                )
                            if len(anomalies) > 5:
                                st.caption(
                                    f"... y {len(anomalies) - 5} anomalía(s) más"
                                )
                        else:
                            st.success("Sin anomalías numéricas detectadas.")

                    elif validator_name == "validate_chart_mentions":
                        mentioned = result.get("mentioned_charts", [])
                        phantom = result.get("phantom_charts", [])

                        if mentioned:
                            st.write(
                                f"**Gráficos mencionados en el informe:** {len(mentioned)}"
                            )
                            for chart in mentioned:
                                exists = chart not in phantom
                                icon = "✅" if exists else "❌"
                                st.write(f"  {icon} `{chart}`")
                        else:
                            st.caption("No se mencionaron gráficos en este turno.")

                        if phantom:
                            st.error(
                                f"**{len(phantom)} gráfico(s) fantasma** (mencionados pero el archivo no existe)."
                            )

                    elif validator_name == "validate_task_agent_mapping":
                        coverage = result.get("routing_coverage")
                        if coverage is not None:
                            st.metric("Cobertura de Routing", f"{float(coverage):.1%}")

                        unrouted = result.get("unrouted_tasks", [])
                        if unrouted:
                            st.warning(
                                f"**Tareas sin agente asignado detectado:** {', '.join(unrouted)}"
                            )
                        else:
                            st.success(
                                "Todas las tareas del Planner tienen agente asignado."
                            )

                        reason = result.get("reason")
                        if reason:
                            st.caption(f"ℹ️ {reason}")

                    st.markdown("---")
    else:
        st.info("No hay detalles disponibles de Layer 1.")

    # ========== DETALLES DE LAYER 2: SEMANTIC ==========
    st.markdown("### Layer 2: Semantic (Embeddings)")

    layer2_details = hybrid_eval.get("layer2_details", {})

    if layer2_details:
        col_sem1, col_sem2 = st.columns(2)

        with col_sem1:
            st.write(
                f"**Método:** {layer2_details.get('evaluation_method', 'embeddings')}"
            )
            st.write(
                f"**Modelo:** {layer2_details.get('model_name', 'all-MiniLM-L6-v2')}"
            )

            avg_score = float(layer2_details.get("avg_score", 0))
            st.metric("Score Promedio Semántico", f"{avg_score:.3f}")

        with col_sem2:
            # Threshold usado
            threshold = layer2_details.get("threshold", 0.7)
            st.write(f"**Threshold de Escalación:** {threshold}")

            # Número de evaluaciones
            num_evals = layer2_details.get("num_evaluations", 0)
            st.write(f"**Evaluaciones Realizadas:** {num_evals}")

        # Task Fidelity
        with st.expander("Task Fidelity (User Query → Planner Tasks)"):
            task_fid = layer2_details.get("task_fidelity", {})

            if task_fid:
                status = task_fid.get("pass", False)
                status_icon = "✅" if status else "❌"
                avg_sim = float(task_fid.get("avg_similarity", 0))

                st.markdown(f"### {status_icon} Similitud: {avg_sim:.3f}")

                if status:
                    st.success(
                        "Las tareas generadas son **semánticamente coherentes** con la consulta del usuario."
                    )
                else:
                    st.warning(
                        "Las tareas generadas tienen **baja coherencia semántica** con la consulta original."
                    )

                # Similitudes individuales
                if "similarities" in task_fid and task_fid["similarities"]:
                    st.write("**Similitudes por Tarea:**")
                    for idx, sim in enumerate(task_fid["similarities"], 1):
                        st.write(f"  {idx}. {float(sim):.3f}")
            else:
                st.info("No hay datos de Task Fidelity.")

        # Agent Fidelities
        with st.expander("Agent Fidelities (Tool Output → Agent Response)"):
            agent_fids = layer2_details.get("agent_fidelities", [])

            if agent_fids:
                for af in agent_fids:
                    agent_name = af.get("agent", "unknown")
                    status = af.get("pass", False)
                    score = float(af.get("score", 0))

                    status_icon = "✅" if status else "❌"
                    status_color = "#2ecc71" if status else "#e74c3c"

                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid {status_color}; padding: 10px; margin: 10px 0; background-color: rgba(0,0,0,0.02);">
                            <div style="font-weight: bold;">{status_icon} {agent_name}</div>
                            <div>Similitud: {score:.3f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if status:
                        st.caption(
                            f"{agent_name} reportó datos **fieles** al output de sus herramientas."
                        )
                    else:
                        st.caption(
                            f"{agent_name} tiene **discrepancias** entre el output de herramientas y su respuesta."
                        )
            else:
                st.info("No hay datos de Agent Fidelities.")
    else:
        st.info("No hay detalles disponibles de Layer 2.")

    # ========== DETALLES DE LAYER 3: LLM-JUDGE ==========
    if layer3_used:
        st.markdown("---")
        st.markdown("### Layer 3: LLM-Judge (Análisis Profundo)")

        layer3_details = hybrid_eval.get("layer3_details", {})

        if layer3_details:
            # Módulos evaluados
            modules_evaluated = layer3_details.get("modules_evaluated", [])

            if modules_evaluated:
                st.write(f"**Módulos Analizados:** {', '.join(modules_evaluated)}")

            # Scores por módulo (si existen)
            if "planner_score" in layer3_details:
                col_l3_1, col_l3_2, col_l3_3 = st.columns(3)

                with col_l3_1:
                    planner_score = float(layer3_details.get("planner_score", 0))
                    st.metric("📋 Planner", f"{planner_score:.2f}/4")

                with col_l3_2:
                    supervisor_score = float(layer3_details.get("supervisor_score", 0))
                    st.metric("🎯 Supervisor", f"{supervisor_score:.2f}/4")

                with col_l3_3:
                    agents_score = float(layer3_details.get("agents_avg_score", 0))
                    st.metric("⚙️ Agentes", f"{agents_score:.2f}/4")

            # Análisis textual
            if "analysis" in layer3_details:
                with st.expander("📝 Ver Análisis Detallado del LLM"):
                    st.write(layer3_details["analysis"])

            # Error category
            if (
                "error_category" in layer3_details
                and layer3_details["error_category"] != "None"
            ):
                st.warning(
                    f"**Categoría de Error Detectada:** {layer3_details['error_category']}"
                )
        else:
            st.info("Layer 3 fue usado pero no hay detalles disponibles.")

    # ========== FALLOS CRÍTICOS ==========
    critical_failures = hybrid_eval.get("critical_failures", [])

    if critical_failures:
        st.markdown("---")
        st.markdown("### Fallos Críticos Detectados")

        st.error(f"**{len(critical_failures)} fallo(s) crítico(s) identificado(s):**")

        for idx, failure in enumerate(critical_failures, 1):
            st.markdown(f"{idx}. {failure}")

    # ========== CONFIGURACIÓN DE PONDERACIÓN ==========
    with st.expander("Ver Configuración de Ponderación (Weights)"):
        weights = hybrid_eval.get("weights_used", {})

        if weights:
            st.write("**Pesos aplicados en el cálculo del score final:**")

            for layer, weight in weights.items():
                st.write(f"  • **{layer}:** {float(weight):.0%}")

            st.caption(
                "Estos pesos determinan la contribución de cada capa al score final."
            )
        else:
            st.info("No hay información de pesos disponible.")

    # ========== RESUMEN EJECUTIVO ==========
    st.markdown("---")
    st.markdown("### Resumen Ejecutivo")

    # Generar resumen basado en los datos
    if layer3_used:
        execution_summary = f"""
        El sistema **escaló a Layer 3** debido a: {hybrid_eval.get('escalation_reason', 'problemas detectados en capas previas')}.
        
        **Proceso de evaluación:**
        - Layer 1 (Guardrails): {layer1_score:.3f} - Validadores deterministas
        - Layer 2 (Semantic): {layer2_score:.3f} - Análisis semántico con embeddings
        - Layer 3 (LLM-Judge): {float(layer3_score):.3f} - Evaluación profunda con LLM
        
        **Score Final:** {final_score:.3f} ({quality_label})
        
        **Tiempo Total:** {eval_time:.3f}s (evaluación profunda completa)
        """
    else:
        execution_summary = f"""
        El sistema **no requirió Layer 3**, resolviendo la evaluación con Layers 1 y 2.
        
        **Proceso de evaluación:**
        - Layer 1 (Guardrails): {layer1_score:.3f} - Validadores deterministas
        - Layer 2 (Semantic): {layer2_score:.3f} - Análisis semántico con embeddings
        
        **Score Final:** {final_score:.3f} ({quality_label})
        
        **Tiempo Total:** {eval_time:.3f}s (evaluación rápida y eficiente)
        """

    st.info(execution_summary.strip())

    # ========== RECOMENDACIONES ==========
    if quality_label in ["Mejorable", "Crítico"] or critical_failures:
        st.markdown("---")
        st.markdown("### Recomendaciones")

        recommendations = []

        # Basado en Layer 1
        if layer1_score < 0.7:
            recommendations.append(
                "- **Mejorar validaciones deterministas:** Se detectaron fallos en Guardrails (routing, sintaxis SQL, cobertura de tareas)."
            )

        # Basado en Layer 2
        if layer2_score < 0.7:
            recommendations.append(
                "- **Revisar coherencia semántica:** Las tareas generadas o las respuestas de agentes no están alineadas semánticamente con las entradas."
            )

        # Basado en Layer 3
        if layer3_used and layer3_score and float(layer3_score) < 0.7:
            recommendations.append(
                "- **Atención a la calidad cualitativa:** El LLM-Judge detectó problemas en la descomposición de tareas, routing o fidelidad de outputs."
            )

        # Basado en fallos críticos
        if critical_failures:
            recommendations.append(
                f"- **Resolver fallos críticos:** {len(critical_failures)} problema(s) grave(s) detectado(s) que requieren atención inmediata."
            )

        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success(
                "No se detectaron problemas significativos. El sistema funcionó correctamente."
            )


def evaluate_with_hybrid(trace_data: dict) -> dict:
    """
    Evaluar con sistema híbrido MACE

    Args:
        trace_data: Dict con trazas del sistema

    Returns:
        Dict con evaluación completa (estructura compatible con frontend)
    """
    evaluator = HybridEvaluator()
    result = evaluator.evaluate(trace_data)

    # El resultado ya viene con la estructura correcta del orchestrator
    return result


def render_comparison_table(baseline_eval=None, llm_judge_data=None, hybrid_eval=None):
    """
    Renderiza tabla comparativa dinámica de 2 o 3 métodos

    Args:
        baseline_eval: Resultado de evaluate_baseline (opcional)
        llm_judge_data: Dict con LLM-Judge (opcional)
        hybrid_eval: Dict con MACE (opcional)
    """
    st.markdown("### Comparativa de Sistemas de Auditoría")

    # Contar evaluadores activos
    active_count = sum(
        1 for x in [baseline_eval, llm_judge_data, hybrid_eval] if x is not None
    )
    if active_count < 2:
        st.info(
            "Ejecuta al menos dos sistemas de auditoría para ver la comparativa global."
        )
        return

    data = {"Métrica": ["Score Global (0-1)", "Tiempo", "Metodología"]}

    # 1. Añadir Baseline
    if baseline_eval:
        data["Baseline"] = [
            f"{float(baseline_eval.baseline_score):.3f}",
            f"{float(baseline_eval.evaluation_time):.3f}s",
            "Determinista",
        ]

    # 2. Añadir LLM-Judge
    if llm_judge_data:
        # Normalizar score de 1-4 a 0-1
        llm_score = float(llm_judge_data["comprehensive_eval"].overall_score) / 4
        llm_time = float(llm_judge_data.get("elapsed_time", 3.5))
        data["LLM-Judge"] = [
            f"{llm_score:.3f}",
            f"{llm_time:.2f}s",
            "Cualitativo (LLM)",
        ]

    # 3. Añadir MACE
    if hybrid_eval:
        mace_score = float(hybrid_eval["final_score"])
        mace_time = float(hybrid_eval["evaluation_time"])
        layers_used = "2 capas" if not hybrid_eval["layer3_used"] else "3 capas"

        data["MACE"] = [
            f"{mace_score:.3f}",
            f"{mace_time:.3f}s",
            f"Híbrido ({layers_used})",
        ]

    df_comparison = pd.DataFrame(data)
    st.dataframe(df_comparison, width="stretch", hide_index=True)

    # Gráfico de tiempos comparativos
    if active_count == 3:
        st.markdown("####Comparación de Tiempos")

        col1, col2, col3 = st.columns(3)

        baseline_time = float(baseline_eval.evaluation_time)
        llm_time = float(llm_judge_data.get("elapsed_time", 3.5))
        mace_time = float(hybrid_eval["evaluation_time"])

        with col1:
            st.metric("Baseline", f"{baseline_time:.3f}s", delta="Más rápido")

        with col2:
            delta_llm = f"+{((llm_time - baseline_time) / baseline_time * 100):.0f}%"
            st.metric("LLM-Judge", f"{llm_time:.2f}s", delta=delta_llm)

        with col3:
            if mace_time < llm_time:
                speedup = ((llm_time - mace_time) / llm_time) * 100
                delta_mace = f"-{speedup:.0f}% vs LLM-J"
            else:
                delta_mace = f"+{((mace_time - llm_time) / llm_time * 100):.0f}%"

            st.metric("MACE", f"{mace_time:.3f}s", delta=delta_mace)


# --- SIDEBAR (Panel de Control) ---
with st.sidebar:
    st.title("Comité de Inversión")

    # BOTÓN PARA IR AL DASHBOARD
    st.markdown("---")
    if st.button("Abrir Dashboard de Análisis", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")
    st.markdown("---")
    st.markdown("### Roles Activos")

    st.markdown(
        """
        <div style="border:1px solid #d0d0d0; border-radius:6px; padding:10px; background-color:#fafafa; margin-bottom:10px;">
            <div style="font-weight:600;">🏛️ Supervisor (CIO)</div>
            <div style="font-size:0.85rem; color:#666;">Coordinación y control del flujo</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("📊 **Technical Analyst** \n_Modelos cuantitativos, SQL y ML_")
    st.warning("📄 **Fundamental Analyst** \n_Research, noticias y RAG_")
    st.error("⚠️ **Risk Officer** \n_Volatilidad, drawdown y alertas_")

    st.divider()

    st.markdown("### Sistema de Auditoría")

    # Toggle para LLM-Judge
    audit_llm_j = st.toggle("LLM-Judge (Evaluación Cualitativa)", value=True)
    if audit_llm_j:
        st.caption("Evaluación con LLM activada:")
        st.caption("• Planner (Descomposición)")
        st.caption("• Supervisor (Routing)")
        st.caption("• Agentes (Herramientas)")
        st.caption("• Output Final (Consolidación)")

    st.divider()

    # Toggle para Baseline
    audit_baseline = st.toggle("Baseline Metrics (Métricas Automáticas)", value=True)
    if audit_baseline:
        st.caption("Métricas deterministas activadas:")
        st.caption("• Routing Accuracy (F1-Score)")
        st.caption("• Numeric Fidelity (Hallucinations)")
        st.caption("• Task Coverage (Completitud)")
        st.caption("• SQL Correctness (Validación)")

    st.divider()

    # Toggle para MACE
    audit_hybrid = st.toggle("MACE(Evaluación Híbrida)", value=True)
    if audit_hybrid:
        st.caption("Sistema de 3 capas activo:")
        st.caption("• L1: Guardrails (Determinista)")
        st.caption("• L2: Semántica (Embeddings)")
        st.caption("• L3: LLM-Judge (Selectivo)")

    st.divider()

    # Estadísticas acumuladas
    st.markdown("### Métricas Acumuladas")

    if st.button("Ver Estadísticas"):
        if "metrics_logger" in st.session_state:
            stats = st.session_state.metrics_logger.get_statistics(source="online")

            if stats["total_evaluations"] > 0:
                st.metric("Total Evaluaciones", stats["total_evaluations"])
                st.metric(
                    "Baseline Promedio",
                    (
                        f"{stats['baseline_avg_score']:.3f}"
                        if stats["baseline_avg_score"]
                        else "N/A"
                    ),
                )
                st.metric(
                    "LLM-Judge Promedio",
                    (
                        f"{stats['llm_judge_avg_score']:.1f}/4"
                        if stats["llm_judge_avg_score"]
                        else "N/A"
                    ),
                )
                st.metric("MACE Promedio", f"{stats['hybrid_avg_score']:.2f}/1.0")

                if stats["error_categories"]:
                    st.write("**Categorías de Error:**")
                    for cat, count in stats["error_categories"].items():
                        st.write(f"  • {cat}: {count}")
            else:
                st.info("No hay evaluaciones acumuladas aún.")

    st.divider()

    st.markdown("### Debugging")
    if st.button("Ver Grafo del Sistema"):
        try:
            graph_image = app.get_graph().draw_mermaid_png()
            # graph_image = app.get_graph(xray=1).draw_mermaid_png() # Para ver lógica de cada agente
            st.image(graph_image, caption="Arquitectura Jerárquica")
        except Exception as e:
            st.error(f"No se pudo generar el grafo: {e}")

    st.divider()

    if st.button("Limpiar Sesión"):
        st.session_state.messages = []
        st.session_state.evaluations = []
        if "trace" in st.session_state:
            del st.session_state.trace
        st.rerun()


# --- INTERFAZ PRINCIPAL ---
st.title("Sistema Multi-Agente de Asesoramiento Financiero")
st.markdown(
    """
Esta arquitectura simula una firma de inversión con **roles especializados**:
* **Planner:** Descompone la consulta en tareas específicas
* **Supervisor:** Enruta cada tarea al especialista correcto
* **Analista Técnico:** Predice precios usando ML y consulta SQL
* **Analista Fundamental:** Busca noticias en vivo e investiga la tecnología
* **Gestor de Riesgos:** Calcula la volatilidad y advierte peligros
"""
)

# 1. Historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

# 2. Renderizar historial de evaluaciones pasadas
if st.session_state.evaluations:
    st.divider()
    st.markdown("## 📊 Historial de Evaluaciones")

    # Crear tabs para cada turno
    tab_labels = [
        f"Turno {eval_data['turn']}" for eval_data in st.session_state.evaluations
    ]
    tabs = st.tabs(tab_labels)

    for idx, (tab, eval_data) in enumerate(zip(tabs, st.session_state.evaluations)):
        with tab:
            st.markdown(f"**Query:** _{eval_data['user_query']}_")
            st.divider()

            # Renderizar LLM-Judge si existe
            if eval_data["llm_judge"]:
                render_llm_judge_panel(eval_data["llm_judge"])

            # Renderizar MACE si existe
            if eval_data.get("hybrid"):
                render_hybrid_panel(eval_data["hybrid"])

            # Renderizar Baseline si existe
            if eval_data["baseline"]:
                render_baseline_panel(eval_data["baseline"])

            # Comparación si hay dos evaluaciones activas
            st.divider()
            render_comparison_table(
                baseline_eval=eval_data.get("baseline"),
                llm_judge_data=eval_data.get("llm_judge"),
                hybrid_eval=eval_data.get("hybrid"),
            )

# 3. Input
user_input = st.chat_input(
    "Ej: 'Analiza el riesgo de ETH', 'Predice el precio de BTC'..."
)

if user_input:
    # Inicializar trace collector
    trace = TraceCollector()
    trace.user_question = user_input

    # A. Usuario
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # B. Procesamiento
    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.status(
            "El Comité está deliberando (Flujo Iterativo)...", expanded=True
        )

        run_name = f"Turno_{len(st.session_state.messages)//2 + 1}"

        final_response_text = ""
        tool_outputs_audit = []

        with mlflow.start_run(run_name=run_name) as run:
            try:
                inputs = {"messages": st.session_state.messages}

                # --- BUCLE DE EJECUCIÓN (STREAMING) ---
                for event in app.stream(inputs):
                    for node_name, node_output in event.items():

                        # ==========================================
                        # 0. PLANNER (Captura de tareas)
                        # ==========================================
                        if node_name == "Planner":
                            trace.capture_planner(node_output)
                            tasks_count = len(trace.planner_tasks)
                            status_container.write(
                                f"📋 **Planner:** Identificó {tasks_count} tarea(s)"
                            )

                            if tasks_count > 0:
                                with status_container.expander(
                                    "📝 Ver Tareas Generadas"
                                ):
                                    for i, task in enumerate(trace.planner_tasks, 1):
                                        st.write(f"{i}. {task}")

                        # ==========================================
                        # 1. SUPERVISOR (Routing y Final)
                        # ==========================================
                        elif node_name == "Supervisor":
                            next_agent = node_output.get("next", "FINISH")

                            # Capturar decisión de routing
                            trace.capture_supervisor_decision(node_output)

                            if next_agent == "FINISH":
                                status_container.write(
                                    f"🏁 **CIO (Supervisor):** Recopilando informes y finalizando."
                                )

                                if (
                                    "messages" in node_output
                                    and len(node_output["messages"]) > 0
                                ):
                                    final_msgs = [
                                        m.content
                                        for m in node_output.get("messages", [])
                                        if isinstance(m, AIMessage)
                                        and m.content.strip()
                                    ]

                                    if final_msgs:
                                        final_response_text = final_msgs[-1]
                                        trace.final_answer = final_response_text
                            else:
                                status_container.write(
                                    f"📡 **CIO (Supervisor):** Derivando tarea a `{next_agent}`..."
                                )

                        # ==========================================
                        # 2. AGENTES ESPECIALIZADOS
                        # ==========================================
                        elif "messages" in node_output:
                            # Capturar current_task del state
                            current_task = node_output.get("current_task", "")

                            # Si current_task está vacío, usar la última tarea del routing_trace
                            if not current_task and trace.routing_trace:
                                current_task = trace.routing_trace[-1].get("task", "")

                            # Capturar ejecución del agente (con current_task)
                            trace.capture_agent_execution(
                                node_name, node_output["messages"], current_task
                            )

                            # Configuración visual por rol
                            icon, card_class = "🤖", "agent-card"
                            if node_name == "Technical_Analyst":
                                icon, card_class = "📊", "card-tech"
                            elif node_name == "Fundamental_Analyst":
                                icon, card_class = "📄", "card-fund"
                            elif node_name == "Risk_Officer":
                                icon, card_class = "⚠️", "card-risk"

                            for msg in node_output["messages"]:

                                # A) Tool Calls (Input)
                                if (
                                    hasattr(msg, "tool_calls")
                                    and len(msg.tool_calls) > 0
                                ):
                                    for tool_call in msg.tool_calls:
                                        t_name = tool_call["name"]
                                        t_args = tool_call["args"]

                                        # Capturar SQL si es crypto_history_tool
                                        if t_name == "crypto_history_tool":
                                            # Validar que 'query' sea string
                                            if "query" in t_args:
                                                query_value = t_args["query"]

                                                # Solo capturar si es string (SQL válido)
                                                if (
                                                    isinstance(query_value, str)
                                                    and query_value.strip()
                                                ):
                                                    trace.capture_sql_query(
                                                        current_task, t_args
                                                    )
                                                else:
                                                    # Log del problema
                                                    print(
                                                        f"⚠️ WARNING: SQL query is not a string: {query_value}"
                                                    )

                                        status_container.write(
                                            f"🛠️ **{node_name}** ejecuta: `{t_name}`"
                                        )
                                        with status_container.expander(
                                            f"📥 Ver Input ({t_name})"
                                        ):
                                            st.json(t_args)

                                # B) Tool Messages (Output)
                                elif msg.type == "tool":
                                    tool_name = msg.name
                                    output_content = msg.content
                                    tool_outputs_audit.append(output_content)

                                    # Detección de imágenes
                                    if (
                                        ".png" in output_content
                                        and "plots_temp" in output_content
                                    ):
                                        try:
                                            match = re.search(
                                                r"(plots_temp/[\w\-\.]+\.png)",
                                                output_content,
                                            )
                                            if match:
                                                image_path = match.group(1)
                                                if os.path.exists(image_path):
                                                    status_container.image(
                                                        image_path,
                                                        caption=f"Gráfico generado por {tool_name}",
                                                    )
                                                    status_container.write(
                                                        f"✅ **Gráfico generado:** `{image_path}`"
                                                    )
                                                else:
                                                    status_container.warning(
                                                        f"Imagen no encontrada: {image_path}"
                                                    )
                                        except Exception:
                                            status_container.code(output_content)
                                    else:
                                        with status_container.expander(
                                            f"📤 Ver Output ({tool_name})"
                                        ):
                                            st.code(output_content)

                                # C) Reporte del Agente
                                elif msg.type == "ai":
                                    content = msg.content
                                    if "### REPORTE" in content:
                                        clean_text = (
                                            content.replace("### REPORTE DEL", "")
                                            .replace("###", "")
                                            .strip()
                                        )
                                        clean_text = (
                                            clean_text.replace("TECHNICAL_ANALYST", "")
                                            .replace("RISK_OFFICER", "")
                                            .replace("FUNDAMENTAL_ANALYST", "")
                                            .strip()
                                        )

                                        status_container.markdown(
                                            f"""
                                            <div class="agent-card {card_class}">
                                                <div class="card-title">{icon} Reporte: {node_name}</div>
                                                {clean_text}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                # --- FIN DEL PROCESAMIENTO ---
                status_container.update(
                    label="✅ Proceso Completado", state="complete", expanded=False
                )

                if not final_response_text:
                    final_response_text = "⚠️ El Supervisor finalizó el proceso pero no se capturó un resumen final de texto."

                # MOSTRAR RESPUESTA FINAL
                st.markdown(final_response_text)
                st.session_state.messages.append(AIMessage(content=final_response_text))

                # ================================================================
                # 3. AUDITORÍA COMPREHENSIVA
                # ================================================================
                if audit_llm_j or audit_baseline or audit_hybrid:
                    st.divider()

                    # Crear objeto de evaluación para este turno
                    current_evaluation = {
                        "turn": len(st.session_state.messages) // 2,
                        "user_query": trace.user_question,
                        "llm_judge": None,
                        "baseline": None,
                        "hybrid": None,
                    }

                    # ============================================================
                    # 3A. LLM-JUDGE (Si está activado)
                    # ============================================================
                    if audit_llm_j:
                        # Para medir el tiempo de ejecucción de LLM-J
                        llm_judge_start_time = time.perf_counter()

                        with st.spinner("⚖️ LLM-Judge analizando el sistema..."):
                            with mlflow.start_span(
                                name="LLM_J_Comprehensive_Audit"
                            ) as audit_span:

                                # 3.1 Evaluar Planner
                                planner_eval = evaluate_planner(
                                    user_message=trace.user_question,
                                    generated_tasks=trace.planner_tasks,
                                    expected_behavior="El Planner debe identificar todas las tareas del mensaje y mantener precisión literal.",
                                )

                                planner_score = (
                                    planner_eval.correctness
                                    + planner_eval.completeness
                                    + planner_eval.precision
                                    + planner_eval.task_decomposition
                                ) / 4

                                # 3.2 Evaluar Supervisor
                                supervisor_eval = evaluate_supervisor(
                                    pending_tasks=trace.planner_tasks,
                                    routing_trace=trace.routing_trace,
                                    expected_behavior="El Supervisor debe enrutar correctamente cada tarea al especialista apropiado.",
                                )

                                supervisor_score = (
                                    supervisor_eval.routing_accuracy
                                    + supervisor_eval.task_completion
                                ) / 2

                                # 3.3 Evaluar Agentes
                                agents_eval = []
                                tools_map = {
                                    "Technical_Analyst": [
                                        "crypto_history_tool",
                                        "crypto_prediction_tool",
                                        "crypto_chart_tool",
                                    ],
                                    "Fundamental_Analyst": [
                                        "crypto_rag_tool",
                                        "crypto_news_tool",
                                    ],
                                    "Risk_Officer": ["crypto_volatility_tool"],
                                }

                                for execution in trace.agent_executions:
                                    agent_name = execution["agent"]
                                    agent_eval = evaluate_agent(
                                        agent_name=agent_name,
                                        current_task=execution.get(
                                            "task", "No especificada"
                                        ),
                                        available_tools=tools_map.get(agent_name, []),
                                        tools_used=execution["tools_used"],
                                        tool_outputs=execution["tool_outputs"],
                                        agent_response=execution["agent_response"],
                                        expected_behavior=f"{agent_name} debe usar las herramientas correctamente y reportar datos fielmente.",
                                    )
                                    agents_eval.append(agent_eval)

                                agents_avg_score = 0
                                if agents_eval:
                                    agents_avg_score = sum(
                                        (
                                            a.tool_selection
                                            + a.tool_execution
                                            + a.output_fidelity
                                            + a.output_completeness
                                            + a.hallucination_check
                                        )
                                        / 5
                                        for a in agents_eval
                                    ) / len(agents_eval)

                                # 3.4 Evaluar Output Final
                                final_eval = evaluate_final_output(
                                    original_tasks=trace.planner_tasks,
                                    agent_outputs=[
                                        e["agent_response"]
                                        for e in trace.agent_executions
                                    ],
                                    final_report=trace.final_answer,
                                    expected_behavior="El informe final debe consolidar todos los outputs de forma completa y precisa.",
                                )

                                final_output_score = (
                                    final_eval.completeness
                                    + final_eval.accuracy
                                    + final_eval.structure
                                    + final_eval.chart_attribution
                                ) / 4

                                # 3.5 Evaluación Comprehensiva
                                comprehensive_eval = evaluate_comprehensive(
                                    planner_eval=planner_eval,
                                    supervisor_eval=supervisor_eval,
                                    agents_eval=agents_eval,
                                    final_eval=final_eval,
                                )

                                audit_span.set_attributes(
                                    {
                                        "overall_score": comprehensive_eval.overall_score,
                                        "error_category": comprehensive_eval.error_category,
                                    }
                                )

                            # Guardar en MLflow
                            mlflow.log_metric(
                                "llm_j_overall_score", comprehensive_eval.overall_score
                            )
                            mlflow.log_metric("llm_j_planner_score", planner_score)
                            mlflow.log_metric(
                                "llm_j_supervisor_score", supervisor_score
                            )
                            mlflow.log_metric("llm_j_agents_score", agents_avg_score)
                            mlflow.log_metric(
                                "llm_j_final_output_score", final_output_score
                            )
                            mlflow.log_param(
                                "error_category", comprehensive_eval.error_category
                            )

                            # Tiempo final de ejecucción LLM-J
                            llm_judge_elapsed_time = (
                                time.perf_counter() - llm_judge_start_time
                            )

                            # Guardar en session_state
                            current_evaluation["llm_judge"] = {
                                "comprehensive_eval": comprehensive_eval,
                                "planner_eval": planner_eval,
                                "planner_score": planner_score,
                                "supervisor_eval": supervisor_eval,
                                "supervisor_score": supervisor_score,
                                "agents_eval": agents_eval,
                                "agents_avg_score": agents_avg_score,
                                "final_eval": final_eval,
                                "final_output_score": final_output_score,
                                "elapsed_time": llm_judge_elapsed_time,
                            }

                        # RENDERIZAR INMEDIATAMENTE
                        render_llm_judge_panel(current_evaluation["llm_judge"])

                    # ============================================================
                    # 3B. BASELINE METRICS (Si está activado)
                    # ============================================================
                    if audit_baseline:
                        st.divider()

                        with st.spinner("📐 Calculando métricas baseline..."):

                            # Preparar trace data
                            trace_data = {
                                "user_question": trace.user_question,
                                "planner_tasks": trace.planner_tasks,
                                "routing_trace": trace.routing_trace,
                                "agent_executions": trace.agent_executions,
                                "sql_queries": trace.sql_queries,
                                "final_answer": trace.final_answer,
                            }

                            # Ejecutar evaluación baseline
                            baseline_eval = evaluate_baseline(trace_data)

                            # Guardar en MLflow
                            mlflow.log_metric(
                                "baseline_score", baseline_eval.baseline_score
                            )
                            mlflow.log_metric(
                                "baseline_routing_f1", baseline_eval.routing_metrics.f1
                            )
                            mlflow.log_metric(
                                "baseline_numeric_f1", baseline_eval.numeric_metrics.f1
                            )
                            mlflow.log_metric(
                                "baseline_hallucination_rate",
                                baseline_eval.numeric_metrics.hallucination_rate,
                            )
                            mlflow.log_metric(
                                "baseline_task_coverage",
                                baseline_eval.task_coverage_metrics.coverage,
                            )
                            mlflow.log_metric(
                                "baseline_sql_correctness",
                                baseline_eval.sql_metrics.correctness,
                            )

                            # Guardar en session_state
                            current_evaluation["baseline"] = baseline_eval

                        # RENDERIZAR INMEDIATAMENTE
                        render_baseline_panel(baseline_eval)

                    # ============================================================
                    # 3C. MACE HYBRID (Si está activado)
                    # ============================================================
                    if audit_hybrid:
                        st.divider()

                        with st.spinner("🔬 MACE analizando (3 capas)..."):
                            # Preparar trace data (mismo formato que baseline)
                            trace_data = {
                                "user_question": trace.user_question,
                                "planner_tasks": trace.planner_tasks,
                                "routing_trace": trace.routing_trace,
                                "agent_executions": trace.agent_executions,
                                "sql_queries": trace.sql_queries,
                                "final_answer": trace.final_answer,
                            }

                            # Ejecutar evaluación híbrida
                            hybrid_eval = evaluate_with_hybrid(trace_data)

                            # Guardar en MLflow
                            mlflow.log_metric(
                                "hybrid_final_score", float(hybrid_eval["final_score"])
                            )
                            mlflow.log_metric(
                                "hybrid_layer1_score",
                                float(hybrid_eval["layer1_score"]),
                            )
                            mlflow.log_metric(
                                "hybrid_layer2_score",
                                float(hybrid_eval["layer2_score"]),
                            )
                            if hybrid_eval["layer3_score"] is not None:
                                mlflow.log_metric(
                                    "hybrid_layer3_score",
                                    float(hybrid_eval["layer3_score"]),
                                )
                            mlflow.log_metric(
                                "hybrid_evaluation_time",
                                float(hybrid_eval["evaluation_time"]),
                            )
                            mlflow.log_param(
                                "hybrid_layer3_used", hybrid_eval["layer3_used"]
                            )
                            mlflow.log_param(
                                "hybrid_quality_label", hybrid_eval["quality_label"]
                            )
                            mlflow.log_param(
                                "hybrid_confidence", hybrid_eval["confidence"]
                            )

                            # Guardar en session_state (resultado completo del orchestrator)
                            current_evaluation["hybrid"] = hybrid_eval

                        # RENDERIZAR INMEDIATAMENTE
                        render_hybrid_panel(hybrid_eval)

                    # ============================================================
                    # COMPARACIÓN (si 2 evaluadores están activados)
                    # ============================================================
                    st.divider()
                    render_comparison_table(
                        baseline_eval=current_evaluation.get("baseline"),
                        llm_judge_data=current_evaluation.get("llm_judge"),
                        hybrid_eval=current_evaluation.get("hybrid"),
                    )

                    # ============================================================
                    # GUARDAR EN METRICS ACCUMULATOR
                    # ============================================================
                    st.session_state.metrics_logger.log_online_evaluation(
                        trace_data=trace_data,
                        baseline_eval=current_evaluation.get("baseline"),
                        llm_judge_data=current_evaluation.get("llm_judge"),
                        hybrid_eval=current_evaluation.get("hybrid"),
                    )

                    st.session_state.evaluations.append(current_evaluation)

            except Exception as e:
                status_container.update(label="❌ Error en el sistema", state="error")
                st.error(f"Ocurrió un error crítico: {str(e)}")
                import traceback

                st.code(traceback.format_exc())
