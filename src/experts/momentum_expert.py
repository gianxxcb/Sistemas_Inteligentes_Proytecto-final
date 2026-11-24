"""
QuantLab ML — Módulo 05: Specialist Models.
Momentum Expert (Experto en Impulso, Aceleración y Persistencia).

Técnicas: Tasa de cambio (ROC), MACD, RSI dinámico, flujo taker y balance de volumen (OBV).
Lógica del playbook:
- Setup: Aceleración consistente de la tasa de cambio multitemporal (ROC 5, 10, 20).
- Disparador: Alineación del impulso con predominancia de volumen agresivo (taker buy ratio).
- Invalidación: Desaceleración súbita o divergencia inversa en osciladores.
- Objetivo: Persistencia direccional del movimiento en el horizonte objetivo.
"""

from typing import Dict, Any, List
import pandas as pd
from .base_expert import BaseExpert


class MomentumExpert(BaseExpert):
    """
    Experto especialista en aceleración y persistencia del momentum.
    """

    def __init__(self, horizon_days: int = 14, version: str = "V2", random_state: int = 42):
        super().__init__(name="MomentumExpert", horizon_days=horizon_days, version=version, random_state=random_state)

    def get_confirmation_feature_names(self) -> List[str]:
        return [
            "roc_5",
            "roc_10",
            "roc_20",
            "macd_line",
            "macd_histogram",
            "taker_buy_ratio",
            "obv_zscore_60",
            "rsi_14",
            "atr_relative",
        ]

    def evaluate_playbook(self, row: pd.Series) -> Dict[str, Any]:
        roc5 = row.get("roc_5", 0.0)
        roc10 = row.get("roc_10", 0.0)
        roc20 = row.get("roc_20", 0.0)
        taker_ratio = row.get("taker_buy_ratio", 0.5)
        obv_z = row.get("obv_zscore_60", 0.0)
        atr_rel = row.get("atr_relative", 0.04)

        rules = []

        # 1. Setup: Aceleración del ROC
        is_bullish_mom = (roc10 > 0.03) and (roc5 > 0.01)
        is_bearish_mom = (roc10 < -0.03) and (roc5 < -0.01)

        if not (is_bullish_mom or is_bearish_mom):
            return {
                "estado": "inactivo",
                "aplicabilidad": 0.1,
                "direccion": "neutral",
                "invalidacion": 0.0,
                "objetivo": 0.0,
                "traza_reglas": ["Sin aceleración clara de momentum (ROC neutral)"],
            }

        direccion = "largo" if is_bullish_mom else "corto"
        rules.append(f"Setup de impulso detectado: {direccion.upper()} (ROC10={roc10:.2f}%)")

        # 2. Disparador: Flujo de compras/ventas taker confirma la aceleración
        if direccion == "largo":
            trigger_fired = (taker_ratio >= 0.51) or (obv_z > 0.5)
        else:
            trigger_fired = (taker_ratio <= 0.49) or (obv_z < -0.5)

        if trigger_fired:
            estado = "disparado"
            aplicabilidad = 0.85
            rules.append(f"Disparador: Confirmación de flujo taker ({taker_ratio:.2f}) y OBV z-score ({obv_z:.2f})")
        else:
            estado = "setup"
            aplicabilidad = 0.55
            rules.append("Setup activo en espera de confirmación de flujo agresivo")

        # 3. Invalidación y objetivo
        if direccion == "largo":
            invalidacion = -max(0.03, atr_rel * 1.4)
            objetivo = max(0.06, atr_rel * 2.5)
        else:
            invalidacion = max(0.03, atr_rel * 1.4)
            objetivo = -max(0.06, atr_rel * 2.5)

        return {
            "estado": estado,
            "aplicabilidad": aplicabilidad,
            "direccion": direccion,
            "invalidacion": invalidacion,
            "objetivo": objetivo,
            "traza_reglas": rules,
        }
