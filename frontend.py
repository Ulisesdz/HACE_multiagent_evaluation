import os
import streamlit as st
import mlflow
from langchain_core.messages import HumanMessage, AIMessage
from orchestrator.graph import build_graph


# --- CONFIGURACIÓN ML FLOW ---
# Carpeta local mlruns
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")

# Nombre experimento
mlflow.set_experiment("TFG_MultiAgent_Orchestrator")

# Autolog para Langchain
mlflow.langchain.autolog()


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Sistema Multi-Agente",
    page_icon="🤖",
    layout="wide"
)

# --- ESTILOS CSS PERSONALIZADOS (OPCIONAL) ---
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; background-color: #f0f2f6;}
    .stStatus {border: 1px solid #e0e0e0; border-radius: 5px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# --- INICIALIZACIÓN DEL ESTADO ---
# Carga del grafo una sola vez usando caché para no recargar el modelo en cada interacción
@st.cache_resource
def load_app():
    return build_graph()

app = load_app()

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("Panel de Control")
    st.markdown("### Agentes Activos")
    st.success("🟢 **Supervisor**: Enrutamiento inteligente")
    st.warning("🟠 **Crypto Agent**: Predicción y Análisis")
    st.info("🔵 **Weather Agent**: Meteorología y Datos")

    if st.button("Limpiar Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    
    # BOTÓN PARA MOSTRAR GRAFO
    st.markdown("### Arquitectura")
    if st.button("Ver Grafo de Agentes"):
        try:
            # Imagen binaria del grafo
            # graph_image = app.get_graph(xray=1).draw_mermaid_png() # Para ver lógica de cada agente
            graph_image = app.get_graph().draw_mermaid_png()
            st.image(graph_image, caption="Arquitectura Jerárquica LangGraph")
        except Exception as e:
            st.error(f"No se pudo generar el grafo visual: {e}") 



# --- INTERFAZ PRINCIPAL ---
st.title("Sistema MultiAgente: Crypto & Weather")
st.markdown("Sistema jerárquico de agentes para predicción de Criptomonedas y Clima utilizando Modelos Locales (Llama 3.1) y RAG")
st.markdown("Pregunta predicciones futuras o datos históricos sobre *Bitcoin, Solana o Ethereum* para criptomonedas, o clima en *Tokio, New York, Paris o Madrid* para información meteorológica.")

# 1. Mostrar historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message.content)

# 2. Input del Usuario
user_input = st.chat_input("Escribe tu pregunta aquí...")

if user_input:
    # A. Mostrar mensaje del usuario
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # B. Procesamiento del Agente
    with st.chat_message("assistant", avatar="🤖"):
        # Contenedor para mostrar el "pensamiento" o proceso interno
        status_container = st.status("🧠 Procesando solicitud...", expanded=True)
        
        try:
            # Ejecutamos el grafo con el historial completo
            inputs = {"messages": st.session_state.messages}
            final_response = "Lo siento, no pude obtener una respuesta."
            
            # Iteramos sobre los eventos del grafo (streaming de pasos)
            for event in app.stream(inputs):
                for node_name, node_output in event.items():
                    
                    # 1. LOGICA SUPERVISOR
                    if node_name == "Supervisor":
                        next_agent = node_output.get("next", "FINISH")
                        status_container.write(f"📡 **Supervisor:** Redirigiendo a `{next_agent}`")
                    
                    # 2. LOGICA DE AGENTES (Crypto y Weather)
                    elif "messages" in node_output:
                        # Analizamos los mensajes nuevos generados en este paso
                        for msg in node_output["messages"]:
                            
                            # A) El Agente QUIERE usar una herramienta (Tool Call)
                            if hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call["name"]
                                    tool_args = tool_call["args"]
                                    status_container.write(f"**{node_name}** decide usar: `{tool_name}`")
                                    with status_container.expander(f"Inputs para {tool_name}"):
                                        st.json(tool_args)

                            # B) La herramienta YA SE EJECUTÓ (Tool Output)
                            # Nota: LangGraph a veces emite el ToolMessage en un nodo separado o junto al agente
                            # dependiendo de cómo esté configurado 'create_react_agent'.
                            # Si detectamos un mensaje de tipo ToolMessage:
                            if msg.type == "tool":
                                tool_name = msg.name
                                tool_output = msg.content
                                status_container.write(f"✅ **Resultado de {tool_name}:**")
                                with status_container.expander(f"Ver Output de {tool_name}"):
                                    st.code(tool_output) # Uso de code para que se vea bien si es CSV o tabla

                            # C) Respuesta final del Agente (AIMessage sin tools)
                            if msg.type == "ai" and not msg.tool_calls:
                                final_response = msg.content

            # Cierre del estado de proceso
            status_container.update(label="✅ Respuesta Generada", state="complete", expanded=False)
            
            # C. Mostrar respuesta final
            st.markdown(final_response)
            
            # Guardar en historial
            st.session_state.messages.append(AIMessage(content=final_response))

        except Exception as e:
            status_container.update(label="❌ Error", state="error")
            st.error(f"Ocurrió un error inesperado: {str(e)}")