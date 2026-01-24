import os
import re
import streamlit as st
import mlflow
from langchain_core.messages import HumanMessage, AIMessage
from evaluation.llm_j.judge import evaluate_response 
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
    .role-supervisor {
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        padding: 12px;
        background-color: #fafafa;
        margin-bottom: 0.5rem;
    }
    .role-supervisor-title {
        font-weight: 600;
        color: #333333;
    }
    .role-supervisor-desc {
        font-size: 0.85rem;
        color: #666666;
    }
    /* Colores para los roles */
    .role-tech {color: #0068c9; font-weight: bold;}
    .role-fund {color: #00cc96; font-weight: bold;}
    .role-risk {color: #ff2b2b; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN ---
@st.cache_resource
def load_app():
    return build_graph()

app = load_app()

# --- SIDEBAR (Panel de Control) ---
with st.sidebar:
    st.title("Comité de Inversión")
    st.markdown("### Roles Activos")
    
    st.markdown(
        """
        <div class="role-supervisor">
            <div class="role-supervisor-title">🏛️ Supervisor (CIO)</div>
            <div class="role-supervisor-desc">Coordinación y control del flujo</div>
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
        st.rerun()

    audit_mode = st.toggle("Activar Auditor (LLM-J)", value=True)
    if audit_mode:
        st.caption("Un Juez IA evaluará la fidelidad y el enrutamiento.")

    st.divider()
    
    st.markdown("### Debugging")
    if st.button("Ver Grafo del Sistema"):
        try:
            # graph_image = app.get_graph(xray=1).draw_mermaid_png() # Para ver lógica de cada agente
            graph_image = app.get_graph().draw_mermaid_png()
            st.image(graph_image, caption="Arquitectura Jerárquica")
        except Exception as e:
            st.error(f"No se pudo generar el grafo: {e}")


# --- INTERFAZ PRINCIPAL ---
st.title("Sistema Multi-Agente de Asesoramiento Financiero")
st.markdown("""
Esta arquitectura simula una firma de inversión con **roles especializados**:
* **Analista Técnico:** Predice precios usando ML y consulta SQL.
* **Analista Fundamental:** Busca noticias en vivo e investiga la tecnología.
* **Gestor de Riesgos:** Calcula la volatilidad y advierte peligros.
""")

# 1. Historial
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

# 2. Input
user_input = st.chat_input("Ej: 'Analiza el riesgo de ETH', 'Predice el precio de BTC'...")

if user_input:
    # A. Usuario
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # B. Procesamiento
    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.status("El Comité está deliberando...", expanded=True)
        
        run_name = f"Turno_{len(st.session_state.messages)//2 + 1}"

        with mlflow.start_run(run_name=run_name) as run:
            try:
                inputs = {"messages": st.session_state.messages}
                current_turn_tools = []
                final_response = "Sin respuesta."
                
                agent_selected = "None" 

                # --- BUCLE DE EJECUCIÓN (STREAMING) ---
                for event in app.stream(inputs):
                    for node_name, node_output in event.items():
                        
                        # 1. LOGICA SUPERVISOR
                        if node_name == "Supervisor":
                            next_agent = node_output.get("next", "FINISH")
                            agent_selected = next_agent
                            status_container.write(f"📡 **CIO (Supervisor):** Derivando a `{next_agent}`")

                        # 2. LOGICA DE LOS ESPECIALISTAS
                        elif "messages" in node_output:
                            # Asignar iconos según el agente
                            icon = "🏛️"
                            if node_name == "Technical_Analyst": icon = "📊"
                            elif node_name == "Fundamental_Analyst": icon = "📄"
                            elif node_name == "Risk_Officer": icon = "⚠️"

                            for msg in node_output["messages"]:
                                # A) USO DE HERRAMIENTAS (Tool Call)
                                if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                                    for tool_call in msg.tool_calls:
                                        t_name = tool_call["name"]
                                        t_args = tool_call["args"]
                                        status_container.write(f"**{icon} {node_name}** usa herramienta: `{t_name}`")
                                        with status_container.expander(f"Inputs: {t_name}"):
                                            st.json(t_args)

                                # B) RESULTADO DE HERRAMIENTA (Tool Output)
                                if msg.type == "tool":
                                    tool_name = msg.name
                                    output_content = msg.content
                                    
                                    # --- DETECCIÓN DE IMÁGENES ---
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
                                        # Output normal
                                        status_container.write(f"✅ **Dato obtenido ({tool_name}):**")
                                        with status_container.expander("Ver detalle"):
                                            st.code(output_content)
                                    
                                    current_turn_tools.append(output_content)

                                # C) RESPUESTA TEXTUAL (AI Message)
                                if msg.type == "ai" and not msg.tool_calls:
                                    final_response = msg.content

                status_container.update(label="✅ Informe Generado", state="complete", expanded=False)
                st.markdown(final_response)
                st.session_state.messages.append(AIMessage(content=final_response))

                # --- 3. AUDITORÍA (LLM-as-a-Judge) ---
                if audit_mode:
                    if not current_turn_tools:
                        context_text = "NO_DATA_RETRIEVED"
                    else:
                        context_text = "\n---\n".join([str(c) for c in current_turn_tools])

                    with st.spinner("⚖️ El Auditor está revisando la calidad y el enrutamiento..."):
                        
                        generic_expectation = (
                            "El Agente debe ser fiel a los datos financieros recuperados. "
                            "El Supervisor debió elegir al especialista correcto. "
                        )

                        verdict = evaluate_response(
                            question=user_input,
                            agent_selected=agent_selected,
                            context=context_text,
                            answer=final_response,
                            expected_behavior=generic_expectation 
                        )
                        
                        # Guardar métricas en MLflow
                        mlflow.log_metric("faithfulness_score", verdict.score)
                        mlflow.log_param("error_type", verdict.error_type)
                        mlflow.log_param("routing_decision", agent_selected)

                    # Tarjeta de Resultados
                    if verdict.score == 10:
                        color, icon_j = "green", "✅"
                    elif verdict.score >= 5:
                        color, icon_j = "orange", "⚠️"
                    else:
                        color, icon_j = "red", "🚨"

                    st.divider()
                    st.markdown(f"### {icon_j} Auditoría de Calidad: :{color}[{verdict.score}/10]")
                    st.info(f"**Análisis:** {verdict.step_by_step_analysis}")
                    st.caption(f"Tipo de Error: {verdict.error_type}")

            except Exception as e:
                status_container.update(label="❌ Error en el sistema", state="error")
                st.error(f"Ocurrió un error crítico: {str(e)}")