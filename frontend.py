import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from orchestrator.graph import build_graph

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="IA Multi-Agente TFG",
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

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("🎛️ Panel de Control")
    st.markdown("### 🎓 Proyecto TFG")
    st.info(
        "Sistema jerárquico de agentes para predicción de Criptomonedas y Clima "
        "utilizando Modelos Locales (Llama 3.1) y RAG."
    )
    
    st.divider()
    
    st.markdown("### 🛠️ Agentes Activos")
    st.success("🟢 **Supervisor**: Enrutamiento inteligente")
    st.warning("🟠 **Crypto Agent**: Predicción y Análisis")
    st.info("🔵 **Weather Agent**: Meteorología y Datos")
    
    st.divider()
    if st.button("🧹 Limpiar Chat"):
        st.session_state.messages = []
        st.rerun()

# --- INICIALIZACIÓN DEL ESTADO ---
# Cargamos el grafo una sola vez usando caché para no recargar el modelo en cada interacción
@st.cache_resource
def load_app():
    return build_graph()

app = load_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- INTERFAZ PRINCIPAL ---
st.title("🤖 Asistente IA: Crypto & Weather")
st.markdown("Pregunta sobre *Bitcoin*, *predicciones futuras*, *clima en Madrid* o *datos históricos*.")

# 1. Mostrar historial de mensajes
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
                    
                    # LOGICA DE VISUALIZACIÓN DE PASOS
                    if node_name == "Supervisor":
                        next_agent = node_output.get("next", "FINISH")
                        status_container.write(f"📡 **Supervisor:** Redirigiendo a `{next_agent}`")
                    
                    elif node_name == "Crypto_Agent":
                        status_container.write("💰 **Crypto Agent:** Analizando mercado y herramientas...")
                        # Capturamos la respuesta si existe
                        if "messages" in node_output:
                            final_response = node_output["messages"][-1].content
                            
                    elif node_name == "Weather_Agent":
                        status_container.write("🌤️ **Weather Agent:** Consultando datos meteorológicos...")
                        if "messages" in node_output:
                            final_response = node_output["messages"][-1].content

            # Cierre del estado de proceso
            status_container.update(label="✅ Respuesta Generada", state="complete", expanded=False)
            
            # C. Mostrar respuesta final
            st.markdown(final_response)
            
            # Guardar en historial
            st.session_state.messages.append(AIMessage(content=final_response))

        except Exception as e:
            status_container.update(label="❌ Error", state="error")
            st.error(f"Ocurrió un error inesperado: {str(e)}")