import os
import re
import time
import streamlit as st
import mlflow
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from evaluation.baseline.evaluator import evaluate_baseline

from evaluation.llm_j.judge import (
    evaluate_planner,
    evaluate_supervisor, 
    evaluate_agent,
    evaluate_final_output,
    evaluate_comprehensive
)
from orchestrator.graph import build_graph

# --- CONFIGURACIÓN ML FLOW ---
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("TFG_Financial_Agent") 
mlflow.langchain.autolog(disable=False)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="AI Investment", 
    page_icon="🏛️", 
    layout="wide"
)

# --- ESTILOS CSS ---
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN ---
@st.cache_resource
def load_app():
    return build_graph()

app = load_app()


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
            self.routing_trace.append({
                "task": current_task,
                "agent": next_agent
            })
    
    def capture_agent_execution(self, agent_name: str, messages: list, current_task: str = ""):
        """Captura ejecución del agente"""
        tools_used = []
        tool_outputs_text = []
        agent_response = ""
        
        for msg in messages:
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tools_used.append(tool_call.get('name', 'unknown'))
                
                if msg.content and not msg.tool_calls:
                    agent_response = msg.content
            
            elif isinstance(msg, ToolMessage):
                tool_outputs_text.append(f"[{msg.name}]: {msg.content}")
        
        if tools_used or agent_response:
            self.agent_executions.append({
                "agent": agent_name,
                "task": current_task,
                "tools_used": tools_used,
                "tool_outputs": "\n".join(tool_outputs_text),
                "agent_response": agent_response
            })
    
    def capture_sql_query(self, task: str, tool_call_args: dict):
        """Captura queries SQL ejecutadas"""
        if 'query' in tool_call_args:
            self.sql_queries.append({
                'task': task,
                'sql': tool_call_args['query']
            })


# ============================================================================
# FUNCIONES DE RENDERIZADO DE EVALUACIONES
# ============================================================================
def render_llm_judge_panel(llm_judge_data):
    """Renderiza el panel de LLM-Judge"""
    comprehensive_eval = llm_judge_data["comprehensive_eval"]
    planner_eval = llm_judge_data["planner_eval"]
    planner_score = llm_judge_data["planner_score"]
    supervisor_eval = llm_judge_data["supervisor_eval"]
    supervisor_score = llm_judge_data["supervisor_score"]
    agents_eval = llm_judge_data["agents_eval"]
    agents_avg_score = llm_judge_data["agents_avg_score"]
    final_eval = llm_judge_data["final_eval"]
    final_output_score = llm_judge_data["final_output_score"]
    
    # Panel de LLM-Judge
    overall_score = comprehensive_eval.overall_score
    if overall_score >= 9:
        score_class = "score-excellent"
        score_label = "EXCELENTE"
    elif overall_score >= 7:
        score_class = "score-good"
        score_label = "BUENO"
    elif overall_score >= 5:
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
                <div class="metric-value {score_class}">{overall_score}/10</div>
                <div style="font-size: 1.2rem; margin-top: 10px; opacity: 0.9;">{score_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Métricas por Módulo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="📋 Planner", value=f"{planner_score:.1f}/10")
        with st.expander("📊 Detalles"):
            st.write(f"**Correctitud:** {planner_eval.correctness}/10")
            st.write(f"**Completitud:** {planner_eval.completeness}/10")
            st.write(f"**Precisión:** {planner_eval.precision}/10")
            st.write(f"**Descomposición:** {planner_eval.task_decomposition}/10")
    
    with col2:
        st.metric(label="🎯 Supervisor", value=f"{supervisor_score:.1f}/10")
        with st.expander("📊 Detalles"):
            st.write(f"**Routing Accuracy:** {supervisor_eval.routing_accuracy}/10")
            st.write(f"**Task Completion:** {supervisor_eval.task_completion}/10")
    
    with col3:
        st.metric(label="⚙️ Agentes", value=f"{agents_avg_score:.1f}/10")
        with st.expander("📊 Detalles"):
            for agent_eval in agents_eval:
                st.write(f"**{agent_eval.agent_name}:**")
                st.write(f"  • Tool Selection: {agent_eval.tool_selection}/10")
                st.write(f"  • Output Fidelity: {agent_eval.output_fidelity}/10")
                st.write(f"  • Hallucination: {agent_eval.hallucination_check}/10")
    
    with col4:
        st.metric(label="📄 Output Final", value=f"{final_output_score:.1f}/10")
        with st.expander("📊 Detalles"):
            st.write(f"**Completitud:** {final_eval.completeness}/10")
            st.write(f"**Precisión:** {final_eval.accuracy}/10")
            st.write(f"**Estructura:** {final_eval.structure}/10")
    
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
        unsafe_allow_html=True
    )
    
    # Métricas Detalladas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="🎯 Routing F1", value=f"{baseline_eval.routing_metrics.f1:.3f}")
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
            label="🔢 Numeric F1",
            value=f"{baseline_eval.numeric_metrics.f1:.3f}"
        )
        with st.expander("📊 Detalles"):
            st.write(f"**Precision:** {baseline_eval.numeric_metrics.precision:.3f}")
            st.write(f"**Recall:** {baseline_eval.numeric_metrics.recall:.3f}")
            st.write(f"**Hallucination Rate:** {halluc_rate:.1%}")
            if halluc_rate > 0.1:
                st.warning("⚠️ Alta tasa de alucinaciones (>10%)")
    
    with col3:
        st.metric(label="✅ Task Coverage", value=f"{baseline_eval.task_coverage_metrics.coverage:.1%}")
        with st.expander("📊 Detalles"):
            st.write(f"**Planificadas:** {baseline_eval.task_coverage_metrics.planned_tasks}")
            st.write(f"**Completadas:** {baseline_eval.task_coverage_metrics.completed_tasks}")
            st.write(f"**Omisión:** {baseline_eval.task_coverage_metrics.omission_rate:.1%}")
    
    with col4:
        st.metric(label="💾 SQL Correctness", value=f"{baseline_eval.sql_metrics.correctness:.1%}")
        with st.expander("📊 Detalles"):
            st.write(f"**Queries Evaluadas:** {baseline_eval.sql_metrics.total_queries}")
            st.write(f"**Queries Correctas:** {baseline_eval.sql_metrics.correct_queries}")
            if baseline_eval.sql_metrics.violations:
                st.warning("**Violaciones SQL:**")
                for violation in baseline_eval.sql_metrics.violations[:3]:
                    st.write(f"• {violation}")


def render_comparison_table(baseline_eval, llm_judge_data):
    """Renderiza la tabla comparativa"""
    st.markdown("### Comparación: Baseline vs LLM-Judge")

    # Obtener tiempo real de LLM-Judge
    llm_judge_time = llm_judge_data.get("elapsed_time", None)
    llm_judge_time_str = f"{llm_judge_time:.2f}s" if llm_judge_time else "~3.5s"
    
    comparison_data = {
        "Método": ["Baseline (Automático)", "LLM-Judge (Cualitativo)"],
        "Score Global": [
            f"{baseline_eval.baseline_score:.3f} (escala 0-1)",
            f"{llm_judge_data['comprehensive_eval'].overall_score}/10"
        ],
        "Tiempo": [
            f"{baseline_eval.evaluation_time:.3f}s",
            llm_judge_time_str
        ],
        "Reproducible": ["✅ 100%", "⚠️ ~95%"],
        "Detecta": ["Errores numéricos/SQL", "Errores semánticos"]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)


# --- SIDEBAR (Panel de Control) ---
with st.sidebar:
    st.title("Comité de Inversión")
    st.markdown("### Roles Activos")
    
    st.markdown(
        """
        <div style="border:1px solid #d0d0d0; border-radius:6px; padding:10px; background-color:#fafafa; margin-bottom:10px;">
            <div style="font-weight:600;">🏛️ Supervisor (CIO)</div>
            <div style="font-size:0.85rem; color:#666;">Coordinación y control del flujo</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info("📊 **Technical Analyst** \n_Modelos cuantitativos, SQL y ML_")
    st.warning("📄 **Fundamental Analyst** \n_Research, noticias y RAG_")
    st.error("⚠️ **Risk Officer** \n_Volatilidad, drawdown y alertas_")

    st.divider()
    
    if st.button("Limpiar Sesión"):
        st.session_state.messages = []
        st.session_state.evaluations = []
        if 'trace' in st.session_state:
            del st.session_state.trace
        st.rerun()

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
    
    st.markdown("### Debugging")
    if st.button("Ver Grafo del Sistema"):
        try:
            graph_image = app.get_graph().draw_mermaid_png()
            st.image(graph_image, caption="Arquitectura Jerárquica")
        except Exception as e:
            st.error(f"No se pudo generar el grafo: {e}")


# --- INTERFAZ PRINCIPAL ---
st.title("Sistema Multi-Agente de Asesoramiento Financiero")
st.markdown("""
Esta arquitectura simula una firma de inversión con **roles especializados**:
* **Planner:** Descompone la consulta en tareas específicas
* **Supervisor:** Enruta cada tarea al especialista correcto
* **Analista Técnico:** Predice precios usando ML y consulta SQL
* **Analista Fundamental:** Busca noticias en vivo e investiga la tecnología
* **Gestor de Riesgos:** Calcula la volatilidad y advierte peligros
""")

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
    tab_labels = [f"Turno {eval_data['turn']}" for eval_data in st.session_state.evaluations]
    tabs = st.tabs(tab_labels)
    
    for idx, (tab, eval_data) in enumerate(zip(tabs, st.session_state.evaluations)):
        with tab:
            st.markdown(f"**Query:** _{eval_data['user_query']}_")
            st.divider()
            
            # Renderizar LLM-Judge si existe
            if eval_data["llm_judge"]:
                render_llm_judge_panel(eval_data["llm_judge"])
            
            # Renderizar Baseline si existe
            if eval_data["baseline"]:
                st.divider()
                render_baseline_panel(eval_data["baseline"])
            
            # Comparación si ambos existen
            if eval_data["llm_judge"] and eval_data["baseline"]:
                render_comparison_table(eval_data["baseline"], eval_data["llm_judge"])

# 3. Input
user_input = st.chat_input("Ej: 'Analiza el riesgo de ETH', 'Predice el precio de BTC'...")

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
        status_container = st.status("El Comité está deliberando (Flujo Iterativo)...", expanded=True)
        
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
                            status_container.write(f"📋 **Planner:** Identificó {tasks_count} tarea(s)")
                            
                            if tasks_count > 0:
                                with status_container.expander("📝 Ver Tareas Generadas"):
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
                                status_container.write(f"🏁 **CIO (Supervisor):** Recopilando informes y finalizando.")

                                if "messages" in node_output and len(node_output["messages"]) > 0:
                                    final_msgs = [
                                        m.content for m in node_output.get("messages", [])
                                        if isinstance(m, AIMessage) and m.content.strip()
                                    ]

                                    if final_msgs:
                                        final_response_text = final_msgs[-1]
                                        trace.final_answer = final_response_text
                            else:
                                status_container.write(f"📡 **CIO (Supervisor):** Derivando tarea a `{next_agent}`...")

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
                            trace.capture_agent_execution(node_name, node_output["messages"], current_task)
                            
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
                                if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                                    for tool_call in msg.tool_calls:
                                        t_name = tool_call["name"]
                                        t_args = tool_call["args"]
                                        
                                        # Capturar SQL si es crypto_history_tool
                                        if t_name == "crypto_history_tool":
                                            # Validar que 'query' sea string
                                            if 'query' in t_args:
                                                query_value = t_args['query']
                                                
                                                # Solo capturar si es string (SQL válido)
                                                if isinstance(query_value, str) and query_value.strip():
                                                    trace.capture_sql_query(current_task, t_args)
                                                else:
                                                    # Log del problema
                                                    print(f"⚠️ WARNING: SQL query is not a string: {query_value}")
                                        
                                        status_container.write(f"🛠️ **{node_name}** ejecuta: `{t_name}`")
                                        with status_container.expander(f"📥 Ver Input ({t_name})"):
                                            st.json(t_args)

                                # B) Tool Messages (Output)
                                elif msg.type == "tool":
                                    tool_name = msg.name
                                    output_content = msg.content
                                    tool_outputs_audit.append(output_content)
                                    
                                    # Detección de imágenes
                                    if ".png" in output_content and "plots_temp" in output_content:
                                        try:
                                            match = re.search(r"(plots_temp/[\w\-\.]+\.png)", output_content)
                                            if match:
                                                image_path = match.group(1)
                                                if os.path.exists(image_path):
                                                    status_container.image(image_path, caption=f"Gráfico generado por {tool_name}")
                                                    status_container.write(f"✅ **Gráfico generado:** `{image_path}`")
                                                else:
                                                    status_container.warning(f"Imagen no encontrada: {image_path}")
                                        except Exception:
                                            status_container.code(output_content)
                                    else:
                                        with status_container.expander(f"📤 Ver Output ({tool_name})"):
                                            st.code(output_content)

                                # C) Reporte del Agente
                                elif msg.type == "ai":
                                    content = msg.content
                                    if "### REPORTE" in content:
                                        clean_text = content.replace("### REPORTE DEL", "").replace("###", "").strip()
                                        clean_text = clean_text.replace("TECHNICAL_ANALYST", "").replace("RISK_OFFICER", "").replace("FUNDAMENTAL_ANALYST", "").strip()
                                        
                                        status_container.markdown(
                                            f"""
                                            <div class="agent-card {card_class}">
                                                <div class="card-title">{icon} Reporte: {node_name}</div>
                                                {clean_text}
                                            </div>
                                            """, 
                                            unsafe_allow_html=True
                                        )

                # --- FIN DEL PROCESAMIENTO ---
                status_container.update(label="✅ Proceso Completado", state="complete", expanded=False)
                
                if not final_response_text:
                    final_response_text = "⚠️ El Supervisor finalizó el proceso pero no se capturó un resumen final de texto."

                # MOSTRAR RESPUESTA FINAL
                st.markdown(final_response_text)
                st.session_state.messages.append(AIMessage(content=final_response_text))

                # ================================================================
                # 3. AUDITORÍA COMPREHENSIVA
                # ================================================================
                if audit_llm_j or audit_baseline:
                    st.divider()

                    # Crear objeto de evaluación para este turno
                    current_evaluation = {
                        "turn": len(st.session_state.messages) // 2,
                        "user_query": trace.user_question,
                        "llm_judge": None,
                        "baseline": None
                    }
                    
                    # ============================================================
                    # 3A. LLM-JUDGE (Si está activado)
                    # ============================================================
                    if audit_llm_j:
                        # Para medir el tiempo de ejecucción de LLM-J
                        llm_judge_start_time = time.perf_counter()

                        with st.spinner("⚖️ LLM-Judge analizando el sistema..."):
                            with mlflow.start_span(name="LLM_J_Comprehensive_Audit") as audit_span:
                                
                                # 3.1 Evaluar Planner
                                planner_eval = evaluate_planner(
                                    user_message=trace.user_question,
                                    generated_tasks=trace.planner_tasks,
                                    expected_behavior="El Planner debe identificar todas las tareas del mensaje y mantener precisión literal."
                                )
                                
                                planner_score = (
                                    planner_eval.correctness +
                                    planner_eval.completeness +
                                    planner_eval.precision +
                                    planner_eval.task_decomposition
                                ) / 4
                                
                                # 3.2 Evaluar Supervisor
                                supervisor_eval = evaluate_supervisor(
                                    pending_tasks=trace.planner_tasks,
                                    routing_trace=trace.routing_trace,
                                    expected_behavior="El Supervisor debe enrutar correctamente cada tarea al especialista apropiado."
                                )
                                
                                supervisor_score = (
                                    supervisor_eval.routing_accuracy +
                                    supervisor_eval.task_completion
                                ) / 2
                                
                                # 3.3 Evaluar Agentes
                                agents_eval = []
                                tools_map = {
                                    "Technical_Analyst": ["crypto_history_tool", "crypto_prediction_tool", "crypto_chart_tool"],
                                    "Fundamental_Analyst": ["crypto_rag_tool", "crypto_news_tool"],
                                    "Risk_Officer": ["crypto_volatility_tool"]
                                }
                                
                                for execution in trace.agent_executions:
                                    agent_name = execution["agent"]
                                    agent_eval = evaluate_agent(
                                        agent_name=agent_name,
                                        current_task=execution.get("task", "No especificada"),
                                        available_tools=tools_map.get(agent_name, []),
                                        tools_used=execution["tools_used"],
                                        tool_outputs=execution["tool_outputs"],
                                        agent_response=execution["agent_response"],
                                        expected_behavior=f"{agent_name} debe usar las herramientas correctamente y reportar datos fielmente."
                                    )
                                    agents_eval.append(agent_eval)
                                
                                agents_avg_score = 0
                                if agents_eval:
                                    agents_avg_score = sum(
                                        (a.tool_selection + a.tool_execution + a.output_fidelity + 
                                         a.output_completeness + a.hallucination_check) / 5
                                        for a in agents_eval
                                    ) / len(agents_eval)
                                
                                # 3.4 Evaluar Output Final
                                final_eval = evaluate_final_output(
                                    original_tasks=trace.planner_tasks,
                                    agent_outputs=[e["agent_response"] for e in trace.agent_executions],
                                    final_report=trace.final_answer,
                                    expected_behavior="El informe final debe consolidar todos los outputs de forma completa y precisa."
                                )
                                
                                final_output_score = (
                                    final_eval.completeness +
                                    final_eval.accuracy +
                                    final_eval.structure +
                                    final_eval.chart_attribution
                                ) / 4
                                
                                # 3.5 Evaluación Comprehensiva
                                comprehensive_eval = evaluate_comprehensive(
                                    planner_eval=planner_eval,
                                    supervisor_eval=supervisor_eval,
                                    agents_eval=agents_eval,
                                    final_eval=final_eval
                                )
                                
                                audit_span.set_attributes({
                                    "overall_score": comprehensive_eval.overall_score,
                                    "error_category": comprehensive_eval.error_category
                                })
                            
                            # Guardar en MLflow
                            mlflow.log_metric("llm_j_overall_score", comprehensive_eval.overall_score)
                            mlflow.log_metric("llm_j_planner_score", planner_score)
                            mlflow.log_metric("llm_j_supervisor_score", supervisor_score)
                            mlflow.log_metric("llm_j_agents_score", agents_avg_score)
                            mlflow.log_metric("llm_j_final_output_score", final_output_score)
                            mlflow.log_param("error_category", comprehensive_eval.error_category)


                            # Tiempo final de ejecucción LLM-J
                            llm_judge_elapsed_time = time.perf_counter() - llm_judge_start_time

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
                                "elapsed_time": llm_judge_elapsed_time
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
                                'user_question': trace.user_question,
                                'planner_tasks': trace.planner_tasks,
                                'routing_trace': trace.routing_trace,
                                'agent_executions': trace.agent_executions,
                                'sql_queries': trace.sql_queries,
                                'final_answer': trace.final_answer
                            }
                            
                            # Ejecutar evaluación baseline
                            baseline_eval = evaluate_baseline(trace_data)
                            
                            # Guardar en MLflow
                            mlflow.log_metric("baseline_score", baseline_eval.baseline_score)
                            mlflow.log_metric("baseline_routing_f1", baseline_eval.routing_metrics.f1)
                            mlflow.log_metric("baseline_numeric_f1", baseline_eval.numeric_metrics.f1)
                            mlflow.log_metric("baseline_hallucination_rate", baseline_eval.numeric_metrics.hallucination_rate)
                            mlflow.log_metric("baseline_task_coverage", baseline_eval.task_coverage_metrics.coverage)
                            mlflow.log_metric("baseline_sql_correctness", baseline_eval.sql_metrics.correctness)

                            # Guardar en session_state
                            current_evaluation["baseline"] = baseline_eval
                        
                        # RENDERIZAR INMEDIATAMENTE
                        render_baseline_panel(baseline_eval)
                    
                    # ============================================================
                    # COMPARACIÓN (si ambos están activados)
                    # ============================================================
                    if audit_llm_j and audit_baseline:
                        render_comparison_table(
                            current_evaluation["baseline"], 
                            current_evaluation["llm_judge"]
                        )
                    
                    # Guardar evaluación en historial
                    st.session_state.evaluations.append(current_evaluation)

            except Exception as e:
                status_container.update(label="❌ Error en el sistema", state="error")
                st.error(f"Ocurrió un error crítico: {str(e)}")
                import traceback
                st.code(traceback.format_exc())