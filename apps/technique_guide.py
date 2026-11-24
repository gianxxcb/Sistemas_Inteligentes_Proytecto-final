"""Names and plain-language explanations for the registered signals."""

import streamlit as st
import pandas as pd

from src.feature_engine.technique_registry import load_technique_registry


PLAYBOOKS = {
    "trend": ("Continuación de tendencia", "moving_averages"),
    "reversal": ("Regreso hacia el promedio", "rsi"),
    "breakout": ("Ruptura del canal", "donchian"),
    "volatility": ("Expansión de volatilidad", "bollinger_squeeze"),
    "momentum": ("Continuación del impulso", "roc"),
}

TECHNIQUES = {
    "moving_averages": ("Medias móviles", "Suavizan el precio para ver si el movimiento general sube o baja."),
    "macd": ("MACD", "Compara dos promedios de distinta velocidad para ver si el movimiento gana o pierde fuerza."),
    "adx_dmi": ("ADX y dirección", "Mide la fuerza de la tendencia y compara la presión de subidas y bajadas."),
    "market_structure": ("Estructura del precio", "Compara máximos y mínimos recientes para ver si el precio avanza, retrocede o sigue lateral."),
    "stochastic_oscillator": ("Estocástico", "Indica si el cierre está cerca de la parte alta o baja de su rango reciente."),
    "support_resistance": ("Soportes y resistencias", "Busca zonas donde el precio frenó o cambió de rumbo antes."),
    "donchian": ("Canal de Donchian", "Marca los máximos y mínimos recientes para identificar cuándo el precio se acerca a romperlos."),
    "breakout_retest": ("Retesteo de una ruptura", "Comprueba si el precio vuelve a un nivel que rompió y consigue mantenerse del otro lado."),
    "rsi": ("RSI", "Compara subidas y bajadas recientes para detectar cuándo un movimiento se ha extendido demasiado."),
    "rsi_divergence": ("Divergencia del RSI", "Busca ocasiones en que el precio sigue avanzando pero el RSI pierde impulso."),
    "macd_divergence": ("Divergencia del MACD", "Compara los giros del precio con los del MACD para detectar desacuerdos."),
    "obv_divergence": ("Divergencia del volumen acumulado", "Comprueba si el volumen acompaña los nuevos máximos o mínimos del precio."),
    "bollinger": ("Bandas de Bollinger", "Rodean un promedio del precio con bandas que se abren cuando aumenta la variación."),
    "mean_reversion_zscore": ("Distancia al promedio", "Mide qué tan lejos está el precio de lo que suele verse en días recientes."),
    "atr": ("Rango verdadero medio", "Estima cuánto suele moverse el precio por día, incluidos los saltos entre sesiones."),
    "realized_volatility": ("Volatilidad observada", "Mide cuánto variaron los precios en los últimos días."),
    "range_compression": ("Compresión del rango", "Detecta cuándo los movimientos recientes se han estrechado."),
    "bollinger_squeeze": ("Compresión de Bollinger", "Busca periodos de poco movimiento que podrían dar paso a una expansión del rango."),
    "keltner": ("Canales de Keltner", "Muestra la distancia entre el precio y un canal construido con su rango habitual."),
    "roc": ("Velocidad del precio", "Compara el precio de hoy con el de días anteriores para medir la fuerza del cambio."),
    "relative_strength_btc": ("Fuerza frente a Bitcoin", "Compara el movimiento de una moneda con el de Bitcoin en el mismo periodo."),
    "volume": ("Volumen relativo", "Compara cuánto se negoció con lo habitual para saber si un movimiento tuvo mayor participación."),
}


def registered_techniques() -> list[dict]:
    return load_technique_registry()["techniques"]


def playbook_label(technique: str) -> str:
    return PLAYBOOKS.get(technique, (technique, ""))[0]


def guide() -> None:
    st.title("Guía de técnicas")
    st.write("Cada especialista combina varias señales en una estrategia. Los retornos del panel corresponden a esa estrategia completa.")
    techniques = registered_techniques()
    group = st.radio("Familia de señales", list(PLAYBOOKS), format_func=lambda k: PLAYBOOKS[k][0], horizontal=True)
    signals = [TECHNIQUES[t["id"]] for t in techniques if t["expert"] == group]
    st.subheader(PLAYBOOKS[group][0])
    st.table(pd.DataFrame(signals,columns=["Indicador","Lectura del mercado"]).set_index("Indicador"))
