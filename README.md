# Sistema Multi-Agente Jerárquico: Crypto & Weather
Este proyecto implementa una arquitectura de Inteligencia Artificial Agéntica Jerárquica para la consulta y predicción de mercados financieros (Criptomonedas) y meteorología.

El sistema combina Modelos de Machine Learning Tradicional (Random Forest para series temporales) con Modelos de Lenguaje (LLMs) locales gestionados por LangGraph. Un agente supervisor orquesta las peticiones del usuario, derivándolas a sub-agentes especializados que cuentan con herramientas de memoria (RAG), consulta de datos históricos (SQL) y predicción futura (ML).

## Estructura del Proyecto

```plaintext
multiagent_evalutation/
├── orchestrator/          # Cerebro del sistema (LangGraph)
│   ├── agents.py          # Definición de Supervisor y Sub-agentes
│   ├── config.py          # Configuración de LLM (Ollama), rutas y MLflow
│   ├── graph.py           # Construcción del grafo de estados
│   ├── main.py            # Ejecución por consola
│   ├── prompts.py         # Gestión centralizada de Prompts y Grounding
│   ├── tools.py           # Herramientas con Introspección de Esquema (Schema Awareness)
│   └── utils.py           # Decoradores de logs y utilidades
├── crypto/
│   ├── crypto_data.db     # Base de datos SQLite con precios históricos
│   ├── RAG_KNOWLEDGE.txt  # Contexto técnico para el agente RAG
│   ├── models/            # Modelos .joblib entrenados
│   ├── plots/             # Gráficas de validación de los modelos
│   └── src/               # Scripts de ML
│       ├── data_manager.py    # ETL: Descarga de Yahoo Finance
│       ├── trainer.py         # Entrenamiento + Validación + Gráficas
│       └── predictor.py       # Inferencia para el Agente
└── weather/
│   ├── weather_data.db    # Base de datos SQLite
│   ├── RAG_KNOWLEDGE.txt  # Conocimiento geográfico para RAG
│   ├── models/            # Modelos .joblib entrenados
│   ├── plots/             # [NUEVO] Gráficas de validación de los modelos
│   └── src/               
│       ├── data_manager.py    # ETL: Gestión de Datasets Climáticos
│       ├── trainer.py         # Entrenamiento + Validación + Gráficas
│       └── predictor.py       # Inferencia para el Agente
└── evaluation/
│   └── llm_j/  # Módulo para evaluación LLM-as-a-judge
│       ├── dataset.json        # Dataset Adversario: Casos diseñados para provocar fallos (Logic, Overflow, etc.)
│       ├── judge.py            # Lógica del Juez: Prompt de Chain-of-Thought (CoT) y esquemas Pydantic
│       ├── run_eval.py         # Script de ejecución offline (Batch Testing)
│       └── results_report.csv  # Reporte generado con métricas y veredictos
├── frontend.py            # Interfaz de usuario con Streamlit
└── setup_rag.py           # Script para vectorizar conocimiento
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
| 2    | `python -m crypto.src.trainer`      | Entrena modelos Random Forest por moneda y muestra el MAE y la precisión.                |
| 3    | `python -m crypto.src.predictor`    | Carga el modelo entrenado y predice el próximo precio de cierre.           |

#### 2. Módulo de Clima (Weather)

| Paso | Comando                          | Descripción                                                                 |
|-----:|----------------------------------|-----------------------------------------------------------------------------|
| 1    | `python -m weather.src.data_manager` | Descarga el dataset de Kaggle, y guarda los datos en °C  de Madrid, NY, Paris y Tokio en la SB|
| 2    | `python -m weather.src.trainer`      | Entrena modelos Random Forest por ciudad y muestra el MAE y la precisión.                |
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

## Características Avanzadas
### Observabilidad con MLflow
El proyecto integra MLflow para la trazabilidad completa de la IA.
1. Interactúa con el chat en Streamlit.
2. Ejecuta en otra terminal: mlflow ui.
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
3. Accede a http://127.0.0.1:5000 para ver:
- Traces: Diagramas de cascada (Waterfall) de cada interacción Agente-Herramienta.
- Prompts: Inspección de qué texto exacto se envía al LLM.
- Latencia: Tiempos de respuesta de cada nodo.

### Guardrails y Grounding Dinámico
El sistema implementa mecanismos de seguridad robustos:
- Introspección de Esquema: Los agentes leen la base de datos al inicio para saber qué tablas (Monedas/Ciudades) existen realmente.
- Anclaje (Grounding): Si preguntas por una ciudad que no está en la DB, el agente rechazará la pregunta en lugar de alucinar datos.
- Prompting Estricto: Reglas explícitas para diferenciar entre un dato histórico (SQL) y una predicción (ML).

### Tipos de Auditoría Implementados
#### LLM As A Judge
El LLM Evaluador analiza la triada: Pregunta Usuario -> Contexto Técnico (SQL/Tool) -> Respuesta Agente y evalúa (Score 0-10) en base a la Fidelidad del Dato y Validación de Procedimiento

1. Auditoría en Tiempo Real (Online) Integrada en la interfaz de Streamlit (frontend.py).
- El usuario puede activar el modo "Juez" mediante un toggle.
- Tras cada respuesta del agente, el sistema inyecta una subtarea de evaluación.
- Visualización: Muestra una tarjeta con Puntuación (0-10), Razonamiento del Juez y Tipo de Error detectado.
2. Evaluación Adversaria (Offline) Ejecuta un banco de pruebas (dataset.json) diseñado para estresar el sistema.
```bash
python -m evaluation.llm_j.run_eval
```

## Arquitectura del Sistema
### 1. El Supervisor (Router)
Analiza la intención del usuario y decide a qué experto derivar la consulta.
- Si la pregunta es sobre Bitcoin/Ethereum -> Crypto Agent.
- Si la pregunta es sobre clima/temperatura -> Weather Agent.
- Si es charla trivial -> Fin.

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