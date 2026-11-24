"""
Volume Features
===============
Funciones puras para características de volumen derivadas de datos OHLCV.

Todas las características son **independientes del precio nominal** (proporciones,
puntajes z o valores relativos) y utilizan únicamente datos hasta el momento t
(sin anticipación).

Funciones
---------
- volume_relative
- obv
- volume_change
- taker_buy_ratio
- quote_volume_relative
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def volume_relative(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volumen relativo (volumen normalizado por su propia SMA).

    Fórmula
    -------
        rvol_t = volume_t / SMA(volume, period)_t

    Parámetros
    ----------
    volume : pd.Series
    period : int, predeterminado 20

    Devuelve
    -------
    pd.Series
        Los valores > 1 indican un volumen superior al promedio.
    """
    vol_sma = volume.rolling(window=period, min_periods=period).mean()
    return volume / vol_sma


def obv(
    close: pd.Series,
    volume: pd.Series,
    zscore_window: int = 60,
) -> pd.Series:
    """Volumen en balance (OBV) normalizado como puntaje z móvil.

    Fórmula
    -------
        OBV_t = OBV_{t−1} + sign(close_t − close_{t−1}) × volume_t

    Como el OBV sin procesar está en unidades nominales y presenta una tendencia
    indefinida, se normaliza de la siguiente manera:

        z_t = (OBV_t − mean(OBV, window)) / std(OBV, window)

    Parámetros
    ----------
    close : pd.Series
    volume : pd.Series
    zscore_window : int, predeterminado 60
        Ventana móvil para la normalización del puntaje z.

    Devuelve
    -------
    pd.Series
        Puntaje z del OBV.
    """
    direction = np.sign(close.diff())
    raw_obv = (direction * volume).cumsum()

    obv_mean = raw_obv.rolling(window=zscore_window, min_periods=zscore_window).mean()
    obv_std = raw_obv.rolling(window=zscore_window, min_periods=zscore_window).std(ddof=1)
    obv_std = obv_std.replace(0, np.nan)

    return (raw_obv - obv_mean) / obv_std


def volume_change(volume: pd.Series) -> pd.Series:
    """Cambio porcentual del volumen entre velas consecutivas.

    Fórmula
    -------
        Δvol_t = (volume_t − volume_{t−1}) / volume_{t−1}

    Parámetros
    ----------
    volume : pd.Series

    Devuelve
    -------
    pd.Series
        El primer valor es NaN.
    """
    prev = volume.shift(1)
    return (volume - prev) / prev


def taker_buy_ratio(
    taker_buy_base_volume: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Proporción del volumen de compra taker respecto al volumen total.

    Fórmula
    -------
        ratio_t = taker_buy_base_volume_t / volume_t

    Los valores cercanos a 0.5 indican un equilibrio entre compras y ventas;
    los valores superiores a 0.5 indican compras netas taker.

    Parámetros
    ----------
    taker_buy_base_volume : pd.Series
    volume : pd.Series

    Devuelve
    -------
    pd.Series
        Valores en [0, 1].
    """
    vol_safe = volume.replace(0, np.nan)
    return taker_buy_base_volume / vol_safe


def quote_volume_relative(quote_volume: pd.Series, period: int = 20) -> pd.Series:
    """Volumen cotizado relativo (volumen cotizado normalizado por su propia SMA).

    Fórmula
    -------
        rqvol_t = quote_volume_t / SMA(quote_volume, period)_t

    Parámetros
    ----------
    quote_volume : pd.Series
    period : int, predeterminado 20

    Devuelve
    -------
    pd.Series
        Los valores > 1 indican un volumen cotizado superior al promedio.
    """
    qv_sma = quote_volume.rolling(window=period, min_periods=period).mean()
    return quote_volume / qv_sma
