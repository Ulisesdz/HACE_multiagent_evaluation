"""
Hybrid Scorer - Fusión de scores de múltiples capas
Combina resultados deterministas, semánticos y generativos
"""

import numpy as np
from typing import Dict, Any, Optional


class HybridScorer:
    """
    Fusión inteligente de scores de las 3 capas
    Ponderación adaptativa según disponibilidad
    """
    
    def __init__(self):
        # Pesos por defecto (si todas las capas se usan)
        self.default_weights = {
            'layer1': 0.3,  # Deterministic (alta confianza)
            'layer2': 0.4,  # Semantic (confianza media)
            'layer3': 0.3   # LLM (confianza media, puede alucinar)
        }
    
    def calculate_final_score(
        self,
        layer1_results: Dict[str, Dict],
        layer2_results: Dict[str, Any],
        layer3_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fusión multi-capa con ponderación adaptativa
        
        Args:
            layer1_results: Resultados de guardrails (dict de validators)
            layer2_results: Resultados semánticos (con avg_score)
            layer3_results: Resultados LLM-Judge (opcional)
        
        Returns:
            Dict con final_score y desglose por capa
        """
        # Score de Capa 1 (convertir pass/fail a 0-1)
        layer1_score = self._compute_layer1_score(layer1_results)
        
        # Score de Capa 2 (ya viene normalizado)
        layer2_score = layer2_results.get('avg_score', 0.5)
        
        # Determinar si se usó Capa 3
        layer3_used = layer3_results is not None and layer3_results.get('evaluation_complete', False)
        
        if layer3_used:
            # Usar pesos por defecto (3 capas)
            layer3_score = layer3_results.get('overall_score', 2.0) / 4  # Normalizar a 0-1
            
            final_score = (
                layer1_score * self.default_weights['layer1'] +
                layer2_score * self.default_weights['layer2'] +
                layer3_score * self.default_weights['layer3']
            )
        else:
            # Solo 2 capas, redistribuir pesos
            # Layer1: 40%, Layer2: 60%
            final_score = layer1_score * 0.4 + layer2_score * 0.6
            layer3_score = None
        
        # Identificar confianza
        confidence = self._assess_confidence(
            layer1_results,
            layer2_results,
            layer3_results,
            layer3_used
        )
        
        return {
            'final_score': final_score,
            'layer1_score': layer1_score,
            'layer2_score': layer2_score,
            'layer3_score': layer3_score,
            'layer3_used': layer3_used,
            'confidence': confidence,
            'weights_used': self._get_weights_used(layer3_used)
        }
    
    def _compute_layer1_score(self, layer1_results: Dict[str, Dict]) -> float:
        """
        Convertir resultados pass/fail de Layer 1 a score 0-1
        
        Args:
            layer1_results: Dict de validadores con 'pass' boolean
        
        Returns:
            Score normalizado 0-1
        """
        if not layer1_results:
            return 1.0
        
        # Contar passes
        passes = sum(1 for v in layer1_results.values() if v.get('pass', False))
        total = len(layer1_results)
        
        return passes / total if total > 0 else 1.0
    
    def _assess_confidence(
        self,
        layer1_results: Dict,
        layer2_results: Dict,
        layer3_results: Optional[Dict],
        layer3_used: bool
    ) -> str:
        """
        Evaluar nivel de confianza en la evaluación
        Basado en desviación estándar entre capas
        
        Args:
            layer1_results: Resultados Layer 1
            layer2_results: Resultados Layer 2
            layer3_results: Resultados Layer 3 (opcional)
            layer3_used: Si se usó Layer 3
        
        Returns:
            'high', 'medium', o 'low'
        """
        layer1_score = self._compute_layer1_score(layer1_results)
        layer2_score = layer2_results.get('avg_score', 0.5)
        
        if not layer3_used:
            # Sin Capa 3: confianza alta si ambas capas coinciden
            if abs(layer1_score - layer2_score) < 0.2:
                return 'high'
            else:
                return 'medium'
        else:
            # Con Capa 3: confianza alta si las 3 capas coinciden
            layer3_score = layer3_results.get('overall_score', 2.0) / 4
            
            scores = [layer1_score, layer2_score, layer3_score]
            std_dev = np.std(scores)
            
            if std_dev < 0.15:
                return 'high'
            elif std_dev < 0.25:
                return 'medium'
            else:
                return 'low'
    
    def _get_weights_used(self, layer3_used: bool) -> Dict[str, float]:
        """
        Obtener pesos aplicados según capas usadas
        
        Args:
            layer3_used: Si se ejecutó Layer 3
        
        Returns:
            Dict con pesos por capa
        """
        if layer3_used:
            return self.default_weights.copy()
        else:
            return {
                'layer1': 0.4,
                'layer2': 0.6,
                'layer3': 0.0
            }
    
    def get_quality_label(self, final_score: float) -> str:
        """
        Convertir score numérico a label cualitativo
        
        Args:
            final_score: Score 0-1
        
        Returns:
            'Excelente', 'Bueno', 'Mejorable', o 'Crítico'
        """
        if final_score >= 0.875:  # 3.5/4
            return 'Excelente'
        elif final_score >= 0.625:  # 2.5/4
            return 'Bueno'
        elif final_score >= 0.375:  # 1.5/4
            return 'Mejorable'
        else:
            return 'Crítico'