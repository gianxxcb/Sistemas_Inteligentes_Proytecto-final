"""
QuantLab ML — Módulo 05: Specialist Models (Modelos Especialistas).
===================================================================
Escuelas de trading especializadas con playbooks coherentes (setup, trigger,
confirmación, invalidación y objetivo).

Implementación de V1 (reglas) y V2 (meta-labeling). V3 temporal queda pendiente.
Las salidas exponen el contrato previsto para la Red de Compuertas (Módulo 06).
"""

from .base_expert import BaseExpert
from .trend_expert import TrendExpert
from .reversal_expert import ReversalExpert
from .breakout_expert import BreakoutExpert
from .volatility_expert import VolatilityExpert
from .momentum_expert import MomentumExpert

EXPERT_CLASSES = (TrendExpert, ReversalExpert, BreakoutExpert, VolatilityExpert, MomentumExpert)

__all__ = [
    "BaseExpert",
    "EXPERT_CLASSES",
    "TrendExpert",
    "ReversalExpert",
    "BreakoutExpert",
    "VolatilityExpert",
    "MomentumExpert",
]
