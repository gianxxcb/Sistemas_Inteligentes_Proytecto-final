"""
QuantLab ML — Módulo 05: Specialist Models.
Breakout Expert (Experto en Rupturas y Falsas Rupturas).

Técnicas: Canales de Donchian, Soportes/Resistencias, volumen relativo y expansión de volatilidad.
Lógica del playbook:
- Setup: Compresión previa de rango (Squeeze o canal Donchian estrecho).
- Disparador: Cierre en el límite del canal Donchian de 20 periodos o ruptura de nivel S/R.
- Confirmación: Expansión de volumen relativo (> 1.2x) y volatilidad en aumento.
- Invalidación: Regreso al interior de la zona del canal (falsa ruptura).
- Objetivo: Recorrido proporcional a la altura de la compresión o ATR.
"""

from typing import Dict, Any, List
import pandas as pd
from .base_expert import BaseExpert


class BreakoutExpert(BaseExpert):
    """
    Experto especialista en rupturas y discriminación de falsas rupturas.
    """

    def __init__(self, horizon_days: int = 14, version: str = "V2", random_state: int = 42):
        super().__init__(name="BreakoutExpert", horizon_days=horizon_days, version=version, random_state=random_state)

    def get_confirmation_feature_names(self) -> List[str]:
        return [
            "donchian_width_20",
            "donchian_dist_upper_20",
            "donchian_dist_lower_20",
            "volume_relative_20",
            "volatility_ratio_20",
            "taker_buy_ratio",
            "range_compression_20",
            "atr_relative",
        ]

    def evaluate_playbook(self, row: pd.Series) -> Dict[str, Any]:
        dist_up = row.get("donchian_dist_upper_20", 0.05)
        dist_dn = row.get("donchian_dist_lower_20", 0.05)
        d_width = row.get("donchian_width_20", 0.25)
        vol_rel = row.get("volume_relative_20", 1.0)
        compression = row.get("range_compression_20", 0.4)
        atr_rel = row.get("atr_relative", 0.04)

        rules = []

        # 1. Setup: Compresión previa del rango
        is_compression_setup = (d_width < 0.28) or (compression > 0.42)

        # 2. Disparador: Ruptura del máximo o mínimo de 20 periodos
        is_break_up = dist_up < 0.012
        is_break_dn = dist_dn < 0.012

        if not (is_break_up or is_break_dn):
            if is_compression_setup:
                return {
                    "estado": "setup",
                    "aplicabilidad": 0.45,
                    "direccion": "neutral",
                    "invalidacion": 0.0,
                    "objetivo": 0.0,
                    "traza_reglas": ["Compresión de rango activa; esperando ruptura Donchian"],
                }
            return {
                "estado": "inactivo",
                "aplicabilidad": 0.05,
                "direccion": "neutral",
                "invalidacion": 0.0,
                "objetivo": 0.0,
                "traza_reglas": ["Sin compresión ni ruptura de canal"],
            }

        direccion = "largo" if is_break_up else "corto"
        rules.append(f"Disparador de ruptura Donchian 20d: {direccion.upper()}")

        # 3. Confirmación: Volumen superior a la media
        has_volume = vol_rel >= 1.15
        if has_volume:
            rules.append(f"Confirmación: Expansión de volumen relativo ({vol_rel:.2f}x)")
            aplicabilidad = 0.85
        else:
            rules.append(f"Advertencia: Ruptura con volumen débil ({vol_rel:.2f}x)")
            aplicabilidad = 0.60

        # 4. Invalidación y objetivo
        if direccion == "largo":
            invalidacion = -max(0.035, atr_rel * 1.5)
            objetivo = max(0.06, atr_rel * 2.8)
        else:
            invalidacion = max(0.035, atr_rel * 1.5)
            objetivo = -max(0.06, atr_rel * 2.8)

        return {
            "estado": "disparado",
            "aplicabilidad": aplicabilidad,
            "direccion": direccion,
            "invalidacion": invalidacion,
            "objetivo": objetivo,
            "traza_reglas": rules,
        }
