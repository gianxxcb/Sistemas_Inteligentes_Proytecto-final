"""
QuantLab ML — Módulo 05: Specialist Models.
Reversal Expert (Experto en Reversión a la Media y Agotamiento).

Técnicas: RSI, Bandas de Bollinger, osciladores de sobrecompra/sobreventa, reversión a la media.
Lógica del playbook:
- Setup: Extensión extrema del precio fuera de bandas o RSI extremo (<32 o >68).
- Disparador: Reingreso del precio dentro de la Banda de Bollinger y giro del oscilador.
- Invalidación: Marcación de un nuevo extremo estructural más allá del pivote.
- Objetivo: Retorno al valor medio (SMA 20).
"""

from typing import Dict, Any, List
import pandas as pd
from .base_expert import BaseExpert


class ReversalExpert(BaseExpert):
    """
    Experto especialista en reversión a la media y giros contra movimientos extendidos.
    """

    def __init__(self, horizon_days: int = 14, version: str = "V2", random_state: int = 42):
        super().__init__(name="ReversalExpert", horizon_days=horizon_days, version=version, random_state=random_state)

    def get_confirmation_feature_names(self) -> List[str]:
        return [
            "rsi_14",
            "bollinger_pctb",
            "dist_to_sma_20",
            "stoch_k_14",
            "sr_dist_support",
            "sr_dist_resistance",
            "volatility_ratio_20",
            "atr_relative",
        ]

    def evaluate_playbook(self, row: pd.Series) -> Dict[str, Any]:
        rsi = row.get("rsi_14", 50.0)
        pct_b = row.get("bollinger_pctb", 0.5)
        dist_sma20 = row.get("dist_to_sma_20", 0.0)
        stoch_k = row.get("stoch_k_14", 50.0)
        atr_rel = row.get("atr_relative", 0.04)

        rules = []

        # 1. Setup: Sobrecompra o sobreventa extrema
        is_oversold_setup = (rsi <= 33.0) or (pct_b <= 0.05) or (dist_sma20 < -0.07)
        is_overbought_setup = (rsi >= 67.0) or (pct_b >= 0.95) or (dist_sma20 > 0.07)

        if not (is_oversold_setup or is_overbought_setup):
            return {
                "estado": "inactivo",
                "aplicabilidad": 0.1,
                "direccion": "neutral",
                "invalidacion": 0.0,
                "objetivo": 0.0,
                "traza_reglas": ["Precio en zona neutral (sin sobreextensión)"],
            }

        direccion = "largo" if is_oversold_setup else "corto"
        rules.append(f"Setup de reversión por sobreextensión: {direccion.upper()} (RSI={rsi:.1f}, %B={pct_b:.2f})")

        # 2. Disparador: Giro del oscilador y reingreso a la banda
        if direccion == "largo":
            trigger_fired = (pct_b > 0.05) and (stoch_k > 20.0)
        else:
            trigger_fired = (pct_b < 0.95) and (stoch_k < 80.0)

        if trigger_fired:
            estado = "disparado"
            aplicabilidad = 0.80
            rules.append("Disparador: Reingreso a las bandas y giro del estocástico/RSI")
        else:
            estado = "setup"
            aplicabilidad = 0.50
            rules.append("Setup de sobreextensión activo en espera de giro")

        # 3. Invalidación y objetivo (retorno a la media)
        if direccion == "largo":
            invalidacion = -max(0.04, atr_rel * 1.6)
            objetivo = max(0.04, abs(dist_sma20))
        else:
            invalidacion = max(0.04, atr_rel * 1.6)
            objetivo = -max(0.04, abs(dist_sma20))

        return {
            "estado": estado,
            "aplicabilidad": aplicabilidad,
            "direccion": direccion,
            "invalidacion": invalidacion,
            "objetivo": objetivo,
            "traza_reglas": rules,
        }
