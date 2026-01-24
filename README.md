# AI Investment Committee: Sistema Multi-Agente Financiero
Este proyecto implementa una **Firma de Inversión basada en Agentes de IA**. Simula un comité de expertos financieros donde un **Supervisor (CIO)** orquesta a tres especialistas con roles definidos para asesorar sobre una cartera diversificada del mercado de criptomonedas, cubriendo activos de distintos perfiles de riesgo (**Bitcoin, Ethereum, Solana, BNB, XRP, Cardano y Dogecoin**).

El sistema combina Modelos de Machine Learning (Random Forest para predicción numérica), Búsqueda en Internet (Datos en vivo), RAG (Conocimiento técnico) y Análisis de Riesgos (Cálculo de volatilidad).

## Estructura del Proyecto

```plaintext
multiagent_evalutation/
├── orchestrator/          # Cerebro del sistema (LangGraph)
│   ├── agents.py          # Definición de Roles: Technical, Fundamental, Risk
│   ├── config.py          # Configuración de LLM (Ollama), rutas y MLflow
│   ├── graph.py           # Grafo de estados: Supervisor -> [Agentes]
│   ├── main.py            # Ejecución CLI con colores por rol
│   ├── prompts.py         # Gestión centralizada de Personalidades y Grounding
│   ├── tools.py           # Herramientas: SQL, ML, Gráficos, Volatilidad, WebSearch
│   └── utils.py           # Decoradores de logs y utilidades
├── crypto/
│   ├── crypto_data.db     # Base de datos SQLite con precios históricos
│   ├── RAG_KNOWLEDGE.txt  # Conocimiento técnico (Halving, Consensus, Macro)
│   ├── models/            # Modelos .joblib entrenados (Random Forest)
│   ├── plots/             # Gráficas de validación de los modelos ML
│   └── src/               # Backend de ML
│       ├── data_manager.py    # ETL: Descarga de Yahoo Finance
│       ├── trainer.py         # Entrenamiento + Validación
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
#### Módulo de Criptomonedas (Crypto)

| Paso | Comando                         | Descripción                                                                 |
|-----:|---------------------------------|-----------------------------------------------------------------------------|
| 1    | `python -m crypto.src.data_manager` | Descarga datos de las criptomonedas y los guarda en la DB.          |
| 2    | `python -m crypto.src.trainer`      | Entrena modelos Random Forest por moneda y muestra el MAE y la precisión.                |
| 3    | `python -m crypto.src.predictor`    | Carga el modelo entrenado y predice el próximo precio de cierre.           |

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
streamlit run frontend.py --server.port 8086
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
Rol: Enrutamiento inteligente. Lógica: Analiza el intent del usuario.
- ¿Pide precio/gráfico? -> Technical Analyst.
- ¿Pide noticias/conceptos? -> Fundamental Analyst.
- ¿Pide seguridad/riesgo? -> Risk Officer.
- ¿Charla/Saludo? -> FINISH.

### 2. Los Sub-Agentes (ReAct)
Cada sub-agente dispone de autonomía para razonar sobre la consulta del usuario (Reason) y decidir qué herramienta ejecutar (Act) en función del contexto temporal y semántico de la pregunta.

| Agente              | Rol (Prompt)                        | Herramientas Principales                                                    | Input / Output                               |
| ------------------- | ----------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------- |
| Technical Analyst   | Quant. Frío, numérico, objetivo.    | crypto_history (SQL)  · crypto_prediction (ML)  · crypto_chart (Matplotlib) | Genera Gráficos .png y tablas de precios.    |
| Fundamental Analyst | Researcher. Educativo y contextual. | crypto_news (Web Search)  · crypto_rag (Vectores)                           | Busca Noticias en vivo y explica tecnología. |
| Risk Officer        | Skeptic. Pesimista y cauteloso.     | crypto_volatility (Pandas)                                                  | Calcula Desviación Estándar y emite alertas. |


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