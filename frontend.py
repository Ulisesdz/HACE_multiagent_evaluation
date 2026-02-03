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
        st.rerun()

    audit_mode = st.toggle("Activar Auditor (LLM-J)", value=True)
    if audit_mode:
        st.caption("Un Juez IA evaluará la fidelidad y el enrutamiento.")

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
        # La caja contenedora de todo el proceso técnico
        status_container = st.status("El Comité está deliberando (Flujo Iterativo)...", expanded=True)
        
        run_name = f"Turno_{len(st.session_state.messages)//2 + 1}"
        
        final_response_text = "" 
        tool_outputs_audit = []
        agent_selected = "Supervisor"

        with mlflow.start_run(run_name=run_name) as run:
            try:
                inputs = {"messages": st.session_state.messages}
                
                # --- BUCLE DE EJECUCIÓN (STREAMING) ---
                for event in app.stream(inputs):
                    for node_name, node_output in event.items():
                        
                        # ==========================================
                        # 1. LOGICA SUPERVISOR (Routing y Final)
                        # ==========================================
                        if node_name == "Supervisor":
                            next_agent = node_output.get("next", "FINISH")
                            agent_selected = next_agent
                            
                            if next_agent == "FINISH":
                                status_container.write(f"🏁 **CIO (Supervisor):** Recopilando informes y finalizando.")

                                if "messages" in node_output and len(node_output["messages"]) > 0:
                                    final_msgs = [
                                        m.content for m in node_output.get("messages", [])
                                        if isinstance(m, AIMessage) and m.content.strip()
                                    ]

                                    if final_msgs:
                                        final_response_text = final_msgs[-1]
                            else:
                                status_container.write(f"📡 **CIO (Supervisor):** Derivando tarea a `{next_agent}`...")

                        # ==========================================
                        # 2. LOGICA DE LOS ESPECIALISTAS
                        # ==========================================
                        elif "messages" in node_output:
                            # Configuración visual por rol
                            icon, card_class = "🤖", "agent-card"
                            if node_name == "Technical_Analyst": 
                                icon, card_class = "📊", "card-tech"
                            elif node_name == "Fundamental_Analyst": 
                                icon, card_class = "📄", "card-fund"
                            elif node_name == "Risk_Officer": 
                                icon, card_class = "⚠️", "card-risk"

                            # Iteramos sobre TODOS los mensajes que devolvió el agente (Tools + Respuesta)
                            for msg in node_output["messages"]:
                                
                                # A) USO DE HERRAMIENTAS (Tool Call - Input)
                                if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                                    for tool_call in msg.tool_calls:
                                        t_name = tool_call["name"]
                                        t_args = tool_call["args"]
                                        
                                        # Input en expander independiente
                                        status_container.write(f"🛠️ **{node_name}** ejecuta: `{t_name}`")
                                        with status_container.expander(f"📥 Ver Input ({t_name})"):
                                            st.json(t_args)

                                # B) RESULTADO DE HERRAMIENTA (Tool Message - Output)
                                elif msg.type == "tool":
                                    tool_name = msg.name
                                    output_content = msg.content
                                    tool_outputs_audit.append(output_content) # Guardar para auditoría
                                    
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
                                        # Output de texto normal en expander independiente
                                        with status_container.expander(f"📤 Ver Output ({tool_name})"):
                                            st.code(output_content)

                                # C) REPORTE FINAL DEL AGENTE (AIMessage)
                                elif msg.type == "ai":
                                    content = msg.content
                                    # Verifica si es un reporte firmado
                                    if "### REPORTE" in content:
                                        # Limpieza del texto para que se vea limpio
                                        clean_text = content.replace("### REPORTE DEL", "").replace("###", "").strip()
                                        clean_text = clean_text.replace("TECHNICAL_ANALYST", "").replace("RISK_OFFICER", "").replace("FUNDAMENTAL_ANALYST", "").strip()
                                        
                                        # Renderizamos TARJETA HTML con la respuesta del agente
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
                
                # --- CORRECCIÓN AQUÍ: Usamos la misma variable ---
                if not final_response_text:
                    final_response_text = "⚠️ El Supervisor finalizó el proceso pero no se capturó un resumen final de texto."

                # MOSTRAR RESPUESTA FINAL
                st.markdown(final_response_text)
                st.session_state.messages.append(AIMessage(content=final_response_text))

                # --- 3. AUDITORÍA (LLM-as-a-Judge) ---
                if audit_mode:
                    if not tool_outputs_audit:
                        context_text = "NO_DATA_RETRIEVED"
                    else:
                        context_text = "\n---\n".join([str(c) for c in tool_outputs_audit])

                    with st.spinner("⚖️ El Auditor está evaluando el caso completo..."):
                        
                        generic_expectation = (
                            "El sistema debe haber descompuesto la pregunta compleja en pasos lógicos. "
                            "Debe haber llamado a los agentes correspondientes (Técnico/Riesgo/Fundamental) "
                            "y la respuesta final debe integrar los datos obtenidos sin alucinar."
                        )

                        verdict = evaluate_response(
                            question=user_input,
                            agent_selected="Iterative_System_Flow",
                            context=context_text,
                            answer=final_response_text,
                            expected_behavior=generic_expectation 
                        )
                        
                        # Guardar métricas en MLflow
                        mlflow.log_metric("faithfulness_score", verdict.score)
                        mlflow.log_param("error_type", verdict.error_type)

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