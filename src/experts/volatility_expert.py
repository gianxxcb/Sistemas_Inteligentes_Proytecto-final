"""
QuantLab ML — Módulo 05: Specialist Models.
Volatility Expert (Experto en Compresión y Expansión de Volatilidad).

Técnicas: ATR, Bollinger Squeeze, Canales de Keltner, volatilidad realizada.
Lógica del playbook:
- Setup: Compresión severa de volatilidad (Bandas de Bollinger dentro de Canales de Keltner).
- Disparador: Salida de la compresión (liberación del squeeze) o expansión de rango.
- Enfoque metodológico: Su estimación primordial es la MAGNITUD esperada del movimiento,
  permitiendo que otros expertos definan el sesgo direccional.
"""

from typing import Dict, Any, List
import pandas as pd
from .base_expert import BaseExpert


class VolatilityExpert(BaseExpert):
    """
    Experto especialista en dinámica de compresión y expansión de volatilidad.
    """

    def __init__(self, horizon_days: int = 14, version: str = "V2", random_state: int = 42):
        super().__init__(name="VolatilityExpert", horizon_days=horizon_days, version=version, random_state=random_state)

    def get_confirmation_feature_names(self) -> List[str]:
        return [
            "bollinger_squeeze",
            "bollinger_bandwidth",
            "vol_realized_10d",
            "vol_realized_20d",
            "atr_relative",
            "volatility_ratio_20",
            "volume_change",
        ]

    def evaluate_playbook(self, row: pd.Series) -> Dict[str, Any]:
        squeeze = row.get("bollinger_squeeze", 0.0)
        bandwidth = row.get("bollinger_bandwidth", 0.3)
        vol_ratio = row.get("volatility_ratio_20", 1.0)
        atr_rel = row.get("atr_relative", 0.04)
        dist_sma20 = row.get("dist_to_sma_20", 0.0)

        rules = []

        # 1. Setup: Squeeze activo o compresión estrecha
        is_squeeze = (squeeze > 0.5) or (bandwidth < 0.22)

        if not is_squeeze and vol_ratio < 1.2:
            return {
                "estado": "inactivo",
                "aplicabilidad": 0.1,
                "direccion": "neutral",
                "invalidacion": 0.0,
                "objetivo": 0.0,
                "traza_reglas": ["Volatilidad en rango medio histórico (sin squeeze)"],
            }

        # 2. Disparador: Liberación de la compresión y expansión
        if is_squeeze:
            if vol_ratio >= 1.10:
                estado = "disparado"
                aplicabilidad = 0.85
                rules.append("Disparador: Liberación del squeeze con expansión de rango (volatility_ratio > 1.1)")
            else:
                estado = "setup"
                aplicabilidad = 0.60
                rules.append("Setup: Bollinger Squeeze activo en compresión")
        else:
            estado = "disparado"
            aplicabilidad = 0.70
            rules.append("Disparador: Expansión súbita de volatilidad realizada")

        # Sesgo direccional leve si el precio ya se desvió de SMA 20, o neutral
        if dist_sma20 > 0.02:
            direccion = "largo"
        elif dist_sma20 < -0.02:
            direccion = "corto"
        else:
            direccion = "neutral"

        # Objetivo de magnitud proporcional a la expansión esperada
        objetivo = max(0.08, atr_rel * 3.0)
        invalidacion = -max(0.04, atr_rel * 1.5)

        return {
            "estado": estado,
            "aplicabilidad": aplicabilidad,
            "direccion": direccion,
            "invalidacion": invalidacion,
            "objetivo": objetivo,
            "traza_reglas": rules,
        }
