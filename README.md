# Sistema Multi-Agente Jerárquico: Crypto & Weather
Este proyecto implementa una arquitectura de Inteligencia Artificial Agéntica Jerárquica para la consulta y predicción de mercados financieros (Criptomonedas) y meteorología.

El sistema combina Modelos de Machine Learning Tradicional (Random Forest para series temporales) con Modelos de Lenguaje (LLMs) locales gestionados por LangGraph. Un agente supervisor orquesta las peticiones del usuario, derivándolas a sub-agentes especializados que cuentan con herramientas de memoria (RAG), consulta de datos históricos (SQL) y predicción futura (ML).

## Estructura del Proyecto

```plaintext
multiagent_evalutation/
├── orchestrator/          # [NUEVO] Cerebro del sistema (LangGraph)
    ├── agents.py          # Definición de Supervisor y Sub-agentes
    ├── config.py          # Configuración de LLM (Ollama) y rutas
    ├── graph.py           # Construcción del grafo de estados
    ├── main.py            # Ejecución por consola
    ├── state.py           # Definición del estado compartido
    └── tools.py           # Conexión entre Agentes y herramientas (ML/SQL/RAG)
├── crypto/
    ├── crypto_data.db     # Base de datos SQLite
    ├── RAG_KNOWLEDGE.txt  # Contexto técnico e histórico para el agente RAG
    ├── models/            # Modelos .joblib entrenados
    └── src/            # Modelos .joblib entrenados    
        ├── crypto_data.db     # Base de datos SQLite
        ├── data_manager.py    # Descarga de datos y gestión de DB
        ├── trainer.py         # Entrenamiento y validación cronológica
        └── predictor.py       # Inferencia y predicciones futuras
└── weather/
    ├── weather_data.db    # Base de datos SQLite
    ├── RAG_KNOWLEDGE.txt  # Conocimiento  de ciudades y climatología para el agente RAG
    ├── models/            # Modelos .joblib entrenados
    └── src/            # Modelos .joblib entrenados    
        ├── data_manager.py    # Descarga de Kaggle y gestión de DB
        ├── trainer.py         # Entrenamiento y cálculo de métricas (MAE, R2)
        └── predictor.py       # Inferencia de temperatura por ciudad
```
## Prerrequisitos
Este sistema funciona 100% en local para garantizar privacidad y coste cero.
- Python 3.10+
- Ollama: Debes tener Ollama instalado y corriendo.
[Descargar Ollama](https://ollama.com/download)
```bash
ollama run llama3.1
```
## Guía de Ejecución

Para poner en marcha el sistema completo, debes seguir este orden lógico: Datos -> Entrenamiento -> Conocimiento -> Aplicación.

### Fase 1: Ingeniería de Datos y Entrenamiento (Backend ML)
#### 1. Módulo de Criptomonedas (Crypto)

| Paso | Comando                         | Descripción                                                                 |
|-----:|---------------------------------|-----------------------------------------------------------------------------|
| 1    | `python -m crypto.src.data_manager` | Descarga datos de BTC, ETH y SOL y los guarda en la DB.          |
| 2    | `python -m crypto.src.trainer`      | Crea la carpeta `models/` y entrena un modelo por cada moneda.             |
| 3    | `python -m crypto.src.predictor`    | Carga el modelo entrenado y predice el próximo precio de cierre.           |

#### 2. Módulo de Clima (Weather)

| Paso | Comando                          | Descripción                                                                 |
|-----:|----------------------------------|-----------------------------------------------------------------------------|
| 1    | `python -m weather.src.data_manager` | Descarga el dataset de Kaggle, y guarda los datos en °C  de Madrid, NY, Paris y Tokio en la SB|
| 2    | `python -m weather.src.trainer`      | Entrena modelos por ciudad y muestra el MAE y la precisión.                |
| 3    | `python -m weather.src.predictor`    | Predice la temperatura de mañana para una ciudad específica.               |

### Fase 2: Configuración del RAG (Base de Conocimiento)
Vectorizamos los archivos de texto (RAG_KNOWLEDGE.txt) para que los agentes puedan consultar teoría o datos cualitativos.
Asegúrate de que existan los archivos .txt en las carpetas crypto/ y weather/.

Ejecuta:
```bash
python setup_rag.py
```

### Fase 3: Ejecución del Sistema Agéntico
Tienes dos formas de interactuar con el sistema:

Opción A: Interfaz Gráfica (Recomendado) Visualiza el proceso de pensamiento del Supervisor y los Agentes en tiempo real.
```bash
streamlit run frontend.py
```
Opción B: Consola (CLI) Interacción rápida por terminal.
```bash
python -m orchestrator.main
```


## Arquitectura del Sistema

### 1. El Supervisor (Router)
Analiza la intención del usuario y decide a qué experto derivar la consulta.
- Si la pregunta es sobre Bitcoin/Ethereum -> Crypto Agent.
- Si la pregunta es sobre clima/temperatura -> Weather Agent.

### 2. Los Sub-Agentes (ReAct)
Cada agente especializado (Crypto y Weather) tiene autonomía para decidir qué herramienta usar según el contexto temporal de la pregunta:

| Tipo de Pregunta | Herramienta Seleccionada                          | Tecnología                                                                 |
|-----:|----------------------------------|-----------------------------------------------------------------------------|
| "¿Qué pasó ayer?"    | *_history_tool | SQL (Consulta a .db)|
| "¿Qué pasará mañana?"    | *_prediction_tool      | ML (Inferencia con .joblib)                |
| "¿Qué es / Por qué?"    | *_rag_tool    | RAG (Búsqueda Vectorial en ChromaDB)           |         

### 3. Componentes de ML (Backend)
- **data_manager.py**  
  Se encarga del proceso ETL (Extraer, Transformar y Limpiar).  
  Crea las bases de datos SQLite y gestiona las tablas para cada activo o ciudad.

- **trainer.py**  
  Implementa un `RandomForestRegressor`.  
  Realiza un *split* cronológico (80/20) para garantizar que el modelo no “prediga el pasado” y guarda los modelos optimizados en la subcarpeta `models/`.

- **predictor.py**  
  Punto de entrada para el usuario.  
  Recibe los últimos datos conocidos (*lags*) y devuelve la predicción numérica utilizando los archivos `.joblib`.