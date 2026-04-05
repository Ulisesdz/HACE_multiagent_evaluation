"""
MACE - Multi-layered Agent Consensus Evaluator
Cascading Hybrid Evaluation System

Arquitectura de 3 capas:
- Layer 1: Guardrails (Deterministic) - ~0.05s
- Layer 2: Semantic Evaluators (ML) - ~0.5s  
- Layer 3: LLM-Judge Selectivo - ~3.5s

Reducción de latencia: ~46% vs LLM-Judge puro
"""

from evaluation.hybrid.orchestrator import HybridEvaluator
from evaluation.hybrid.layer1_guardrails import GuardrailsValidator
from evaluation.hybrid.layer2_semantic import SemanticEvaluator
from evaluation.hybrid.layer3_llm import SelectiveLLMJudge
from evaluation.hybrid.scorer import HybridScorer

__all__ = [
    'HybridEvaluator',
    'GuardrailsValidator',
    'SemanticEvaluator',
    'SelectiveLLMJudge',
    'HybridScorer'
]

__version__ = '2.0.0'