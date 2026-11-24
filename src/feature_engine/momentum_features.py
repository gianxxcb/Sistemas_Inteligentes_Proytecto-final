"""
Características de momentum
=================
Funciones puras para características de momentum derivadas de datos OHLCV.

Todas las características son **independientes del precio nominal** y utilizan
únicamente datos hasta el momento t (sin anticipación).

Funciones
---------
- rsi
- roc
- macd_histogram
- stochastic_k
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Índice de fuerza relativa (RSI) con suavizado de Wilder.

    Fórmula
    -------
        Δ_t    = close_t − close_{t−1}
        gain_t = max(Δ_t, 0)
        loss_t = max(−Δ_t, 0)

        Primer avg_gain = mean(gain[1..period])
        Primer avg_loss = mean(loss[1..period])

        avg_gain_t = (avg_gain_{t−1} · (period−1) + gain_t) / period
        avg_loss_t = (avg_loss_{t−1} · (period−1) + loss_t) / period

        RS  = avg_gain / avg_loss
        RSI = 100 − 100 / (1 + RS)

    Ya está en una escala de 0 a 100 (no requiere normalización adicional).

    Parámetros
    ----------
    close : pd.Series
    period : int, predeterminado 14

    Devuelve
    -------
    pd.Series
        Valores de RSI (0-100).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    result = np.full(len(close), np.nan, dtype=float)

    # Primer promedio válido (se requieren *period* diferencias; comienza en el índice period)
    if len(close) < period + 1:
        return pd.Series(result, index=close.index)

    gain_values = gain.to_numpy(dtype=float)
    loss_values = loss.to_numpy(dtype=float)
    avg_gain = np.nanmean(gain_values[1 : period + 1])
    avg_loss = np.nanmean(loss_values[1 : period + 1])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period + 1, len(close)):
        avg_gain = (avg_gain * (period - 1) + gain_values[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss_values[i]) / period
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)

    return pd.Series(result, index=close.index)


def roc(close: pd.Series, periods: List[int] | None = None) -> pd.DataFrame:
    """Tasa de cambio porcentual.

    Fórmula
    -------
        ROC_N(t) = (close_t − close_{t−N}) / close_{t−N} × 100

    Parámetros
    ----------
    close : pd.Series
    periods : list[int], predeterminado [5, 10, 20]

    Devuelve
    -------
    pd.DataFrame
        Columnas con nombre ``roc_{period}`` para cada período.
    """
    if periods is None:
        periods = [5, 10, 20]
    result = pd.DataFrame(index=close.index)
    for p in periods:
        prev = close.shift(p)
        result[f"roc_{p}"] = (close - prev) / prev * 100
    return result


def macd_histogram(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """Histograma MACD normalizado por el precio de cierre.

    Fórmula
    -------
        EMA_fast  = EMA(close, fast)
        EMA_slow  = EMA(close, slow)
        MACD_line = (EMA_fast − EMA_slow) / close
        Signal    = EMA(MACD_line, signal)
        Histogram = MACD_line − Signal

    Reutiliza la misma lógica que ``trend_features.macd``, pero devuelve
    únicamente la serie del histograma como función auxiliar.

    Parámetros
    ----------
    close : pd.Series
    fast, slow, signal : int

    Devuelve
    -------
    pd.Series
        Histograma MACD normalizado.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = (ema_fast - ema_slow) / close
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - signal_line


def stochastic_k(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Oscilador estocástico %K.

    Fórmula
    -------
        %K_t = 100 × (close_t − lowest_low_N) / (highest_high_N − lowest_low_N)

    donde ``lowest_low_N`` y ``highest_high_N`` son el mínimo y el máximo de
    las últimas *period* velas (incluida t).

    Ya está en una escala de 0 a 100.

    Parámetros
    ----------
    close, high, low : pd.Series
    period : int, predeterminado 14

    Devuelve
    -------
    pd.Series
        Valores de %K (0-100). NaN durante las velas iniciales de calentamiento.
    """
    lowest = low.rolling(window=period, min_periods=period).min()
    highest = high.rolling(window=period, min_periods=period).max()
    denom = highest - lowest
    denom = denom.replace(0, np.nan)
    return 100 * (close - lowest) / denom
