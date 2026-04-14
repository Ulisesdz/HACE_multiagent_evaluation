# MACE - Multi-layered Agent Consensus Evaluator

## Descripción

Sistema de evaluación híbrida de 3 capas para sistemas multi-agente. Combina validadores deterministas, evaluación semántica con embeddings, y LLM-Judge selectivo.

**Ventajas:**
- ~46% más rápido que LLM-Judge puro
- Cobertura semántica completa
- Menor costo (menos llamadas a LLM)

---

## Arquitectura
```
┌─────────────────────────────────────────┐
│         TRACE DEL SISTEMA               │
└──────────────┬──────────────────────────┘
               │
   ┌───────────▼───────────┐
   │  CAPA 1: GUARDRAILS   │  (~0.05s)
   │  (Deterministic)      │
   └───────────┬───────────┘
               │
        ┌──────▼──────┐
        │ ¿PASS?      │
        └──────┬──────┘
          Yes  │  No
   ┌───────────┴───────────┐
   │                       │
┌──▼─────┐          ┌─────▼──────┐
│ CAPA 2 │          │ CRITICAL   │
│ SEMANT │          │ FAILURE    │
│ (~0.5s)│          │ → CAPA 3   │
└──┬─────┘          └─────┬──────┘
   │                      │
┌──▼──────┐         ┌─────▼──────┐
│ ¿PASS?  │         │ CAPA 3:    │
└──┬──────┘         │ LLM-JUDGE  │
   │                │ (~3.5s)    │
   │                └─────┬──────┘
   │                      │
   └──────────┬───────────┘
              │
        ┌─────▼──────┐
        │ SCORE      │
        │ FINAL      │
        └────────────┘
```

---

## Componentes

### Layer 1: Guardrails (`layer1_guardrails.py`)

Validadores deterministas (100% reproducibles):
- Completitud estructural
- Sintaxis de routing
- Rangos numéricos razonables
- Menciones de archivos (gráficos)
- Mapeo tarea-agente

### Layer 2: Semantic Evaluators (`layer2_semantic.py`)

Evaluación semántica con ML:
- Task Fidelity (BERTScore-inspired)
- Agent Fidelity (embedding similarity)
- Routing Quality (keyword heuristics)
- Report Completeness

**Modelos usados:**
- `all-MiniLM-L6-v2` (SentenceTransformer, ~80MB)

### Layer 3: LLM-Judge Selectivo (`layer3_llm.py`)

Análisis profundo solo de módulos problemáticos:
- 📋 Planner (si task fidelity < threshold)
- 🎯 Supervisor (si routing quality < threshold)
- ⚙️ Agents (si agent fidelity < threshold)
- 📄 Final Output (si completeness < threshold)

### Scorer (`scorer.py`)

Fusión inteligente de scores:
- Ponderación adaptativa según capas usadas
- Evaluación de confianza
- Normalización a escala 0-1

### Orchestrator (`orchestrator.py`)

Coordinador del pipeline:
- Ejecución secuencial de capas
- Decisión de escalación
- Consolidación de resultados

---

## Métricas de Rendimiento

### Latencia

| Escenario | Capas | Tiempo | % Casos |
|-----------|-------|--------|---------|
| Perfecto | 1+2 | ~0.55s | ~60% |
| Ambiguo | 1+2+3 | ~4.0s | ~30% |
| Crítico | 1+3 | ~3.55s | ~10% |

**Reducción vs LLM-Judge puro:** ~46%

### Reproducibilidad

- Layer 1: 100%
- Layer 2: determinista dado mismo input
- Layer 3: igual que LLM-Judge

### Distribución de Carga

- ~60% casos resueltos en Capas 1-2 (rápido)
- ~40% requieren Capa 3 (profundo)

---

## Comparación con Otros Métodos

| Método | Latencia | Reproducibilidad | Semántico | Numérico | Estructural |
|--------|----------|------------------|-----------|----------|-------------|
| Baseline | 0.02s | Total | ❌ | ✅ | ✅ |
| LLM-Judge | 3.5s | Parcial | ✅ | ✅ | ✅ |
| **MACE** | **~1.9s** | Equilibrio | **✅** | **✅** | **✅** |

---

## Papers Base

1. **BERTScore** (Zhang et al., 2019) - Similitud semántica via embeddings
2. **ARES** (Saad-Falcon et al., 2023) - Clasificadores binarios para RAG
3. **Cascading LLMs** (Chen et al., 2023) - Routing adaptativo
4. **Prometheus** (Kim et al., 2023) - Rubric-guided evaluation

---

## Troubleshooting

### Error: `sentence-transformers` no instalado
```bash
pip install sentence-transformers
```

Si no quieres instalar, Layer 2 usará solo heurísticas (ligeramente menos preciso pero funcional).

### Latencia alta

Si >90% de casos van a Capa 3, ajustar thresholds en `orchestrator.py`:
```python
# Hacer escalación más conservadora
if 0.4 <= layer2_score <= 0.7:  # Antes: 0.5-0.75
    return True, "Ambiguous score"
```

### Baja correlación con LLM-Judge

Verificar que Layer 2 (embeddings) esté funcionando:
```python
evaluator = HybridEvaluator()
print(f"Embeddings available: {evaluator.layer2.available}")
```

---