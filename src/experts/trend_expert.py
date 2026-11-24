"""
QuantLab ML — Módulo 05: Specialist Models.
Trend Expert (Experto en Seguimiento y Agotamiento de Tendencia).

Técnicas: SMA, EMA, MACD, ADX, estructura de máximos y mínimos.
Lógica del playbook:
- Setup: Tendencia definida (ADX >= 22, medias alineadas).
- Disparador: Retroceso controlado a la media de 20 periodos con continuación confirmada.
- Invalidación: Cierre estructural opuesto a la SMA 50.
- Objetivo: Proporcional a ATR en la dirección de la tendencia.
"""

from typing import Dict, Any, List
import pandas as pd
from .base_expert import BaseExpert


class TrendExpert(BaseExpert):
    """
    Experto especialista en seguimiento y agotamiento de tendencias.
    """

    def __init__(self, horizon_days: int = 14, version: str = "V2", random_state: int = 42):
        super().__init__(name="TrendExpert", horizon_days=horizon_days, version=version, random_state=random_state)

    def get_confirmation_feature_names(self) -> List[str]:
        return [
            "adx_14",
            "sma_slope_20",
            "dist_to_sma_20",
            "dist_to_sma_50",
            "dist_to_sma_200",
            "macd_histogram",
            "volume_relative_20",
            "higher_highs_ratio_20",
            "lower_lows_ratio_20",
        ]

    def evaluate_playbook(self, row: pd.Series) -> Dict[str, Any]:
        adx = row.get("adx_14", 20.0)
        dist_sma20 = row.get("dist_to_sma_20", 0.0)
        dist_sma50 = row.get("dist_to_sma_50", 0.0)
        dist_sma200 = row.get("dist_to_sma_200", 0.0)
        slope = row.get("sma_slope_20", 0.0)
        macd_hist = row.get("macd_histogram", 0.0)
        vol_rel = row.get("volume_relative_20", 1.0)
        atr_rel = row.get("atr_relative", 0.04)

        rules = []

        # 1. Setup
        is_bullish_trend = (adx >= 22.0) and (dist_sma50 > 0.01) and (slope > 0.0)
        is_bearish_trend = (adx >= 22.0) and (dist_sma50 < -0.01) and (slope < 0.0)

        if not (is_bullish_trend or is_bearish_trend):
            return {
                "estado": "inactivo",
                "aplicabilidad": 0.1,
                "direccion": "neutral",
                "invalidacion": 0.0,
                "objetivo": 0.0,
                "traza_reglas": ["Mercado lateral o ADX < 22"],
            }

        direccion = "largo" if is_bullish_trend else "corto"
        rules.append(f"Setup tendencial detectado: {direccion.upper()} (ADX={adx:.1f})")

        # 2. Disparador (Pullback a la media SMA 20 con continuación)
        is_pullback = abs(dist_sma20) < 0.035
        has_momentum_continuation = (macd_hist > 0) if direccion == "largo" else (macd_hist < 0)

        if is_pullback and has_momentum_continuation:
            estado = "disparado"
            aplicabilidad = 0.85
            rules.append("Disparador: Retroceso a SMA 20 con histograma MACD a favor")
        else:
            estado = "setup"
            aplicabilidad = 0.55
            rules.append("Setup activo en espera de retroceso a la media")

        # 3. Invalidación y objetivo
        if direccion == "largo":
            invalidacion = -max(0.03, atr_rel * 1.5)
            objetivo = max(0.06, atr_rel * 2.5)
        else:
            invalidacion = max(0.03, atr_rel * 1.5)
            objetivo = -max(0.06, atr_rel * 2.5)

        return {
            "estado": estado,
            "aplicabilidad": aplicabilidad,
            "direccion": direccion,
            "invalidacion": invalidacion,
            "objetivo": objetivo,
            "traza_reglas": rules,
        }
