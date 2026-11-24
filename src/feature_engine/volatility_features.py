"""
Volatility Features
===================
Funciones puras para características de volatilidad derivadas de datos OHLCV.

Todas las características son **independientes del precio nominal** (proporciones,
percentiles o valores relativos) y solo utilizan datos hasta el momento t (sin anticipación).

Funciones
---------
- atr
- atr_relative
- bollinger_bands
- bollinger_squeeze
- keltner_channels
- volatility_ratio
- rolling_std_return
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Rango verdadero promedio (ATR) con suavizado de Wilder.

    Fórmula
    -------
        TR_t   = max(high_t − low_t,
                     |high_t − close_{t−1}|,
                     |low_t  − close_{t−1}|)

        ATR_1  = mean(TR[1..period])
        ATR_t  = (ATR_{t−1} · (period−1) + TR_t) / period

    Parámetros
    ----------
    high, low, close : pd.Series
    period : int, predeterminado 14

    Devuelve
    -------
    pd.Series
        Valores ATR sin procesar (en las mismas unidades que el precio; use
        ``atr_relative`` para normalizarlos).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    result = np.full(len(close), np.nan, dtype=float)

    if len(tr) < period + 1:
        return pd.Series(result, index=close.index)

    # El primer ATR es el promedio de los primeros *period* valores TR (se omite el índice 0 → NaN por el desplazamiento)
    tr_values = tr.to_numpy(dtype=float)
    result[period] = np.nanmean(tr_values[1 : period + 1])
    for i in range(period + 1, len(tr_values)):
        result[i] = (result[i - 1] * (period - 1) + tr_values[i]) / period

    return pd.Series(result, index=close.index)


def atr_relative(atr_val: pd.Series, close: pd.Series) -> pd.Series:
    """ATR normalizado (relativo al cierre).

    Fórmula
    -------
        ATR_rel_t = ATR_t / close_t

    Parámetros
    ----------
    atr_val : pd.Series
        ATR calculado previamente.
    close : pd.Series

    Devuelve
    -------
    pd.Series
    """
    return atr_val / close


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std: float = 2.0,
) -> pd.DataFrame:
    """Bandas de Bollinger con %B y ancho derivados.

    Fórmula
    -------
        Middle = SMA(close, period)
        Upper  = Middle + std × σ(close, period)
        Lower  = Middle − std × σ(close, period)
        %B     = (close − Lower) / (Upper − Lower)
        BW     = (Upper − Lower) / Middle

    Parámetros
    ----------
    close : pd.Series
    period : int, predeterminado 20
    std : float, predeterminado 2.0

    Devuelve
    -------
    pd.DataFrame
        Columnas: ``bb_upper``, ``bb_lower``, ``bb_pct_b``, ``bb_bandwidth``.
    """
    rolling_mean = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std(ddof=0)

    upper = rolling_mean + std * rolling_std
    lower = rolling_mean - std * rolling_std

    band_diff = upper - lower
    band_diff_safe = band_diff.replace(0, np.nan)

    pct_b = (close - lower) / band_diff_safe
    bandwidth = band_diff / rolling_mean

    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": pct_b,
            "bb_bandwidth": bandwidth,
        },
        index=close.index,
    )


def bollinger_squeeze(
    bandwidth: pd.Series,
    lookback: int = 120,
) -> pd.Series:
    """Compresión de Bollinger: rango percentil del ancho en una ventana retrospectiva.

    Fórmula
    -------
        squeeze_t = percentile_rank(bandwidth_t, bandwidth[t−lookback+1 .. t])

    Un percentil bajo indica volatilidad comprimida (squeeze).

    Parámetros
    ----------
    bandwidth : pd.Series
        Ancho de las bandas de Bollinger (de ``bollinger_bands``).
    lookback : int, predeterminado 120

    Devuelve
    -------
    pd.Series
        Percentil del ancho actual dentro de la ventana retrospectiva (0-1).
    """
    return rolling_percentile_rank(bandwidth, lookback=lookback)


def rolling_percentile_rank(
    values: pd.Series,
    lookback: int = 120,
) -> pd.Series:
    """Rank the current value against its own trailing history without look-ahead."""
    if lookback < 1:
        raise ValueError("lookback debe ser al menos 1")

    raw_values = values.to_numpy(dtype=float)
    result = np.full(len(raw_values), np.nan, dtype=float)
    if len(raw_values) >= lookback:
        windows = np.lib.stride_tricks.sliding_window_view(raw_values, lookback)
        valid = ~np.isnan(windows)
        valid_counts = valid.sum(axis=1)
        ranks = ((windows <= windows[:, -1, None]) & valid).sum(axis=1)
        complete_windows = valid_counts == lookback
        ranked_windows = np.full(len(windows), np.nan, dtype=float)
        ranked_windows[complete_windows] = (
            ranks[complete_windows] / valid_counts[complete_windows]
        )
        result[lookback - 1:] = ranked_windows
    return pd.Series(result, index=values.index, name=values.name)


# ---------------------------------------------------------------------------
# Canales Keltner
# ---------------------------------------------------------------------------

def keltner_channels(
    close: pd.Series,
    atr_val: pd.Series,
    ema_val: pd.Series,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Canales Keltner.

    Fórmula
    -------
        Upper = EMA + multiplier × ATR
        Lower = EMA − multiplier × ATR

    Parámetros
    ----------
    close : pd.Series
        (se usa únicamente para el índice)
    atr_val : pd.Series
        ATR calculado previamente.
    ema_val : pd.Series
        EMA calculada previamente (línea central).
    multiplier : float, predeterminado 1.5

    Devuelve
    -------
    pd.DataFrame
        Columnas: ``kc_upper``, ``kc_lower``.
    """
    upper = ema_val + multiplier * atr_val
    lower = ema_val - multiplier * atr_val
    return pd.DataFrame(
        {"kc_upper": upper, "kc_lower": lower},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Razón de volatilidad
# ---------------------------------------------------------------------------

def volatility_ratio(atr_val: pd.Series, period: int = 20) -> pd.Series:
    """Razón de expansión/contracción de la volatilidad.

    Fórmula
    -------
        vr_t = ATR_t / SMA(ATR, period)_t

    Los valores > 1 indican que el ATR supera su promedio reciente (expansión);
    los valores < 1 indican contracción.

    Parámetros
    ----------
    atr_val : pd.Series
        ATR calculado previamente.
    period : int, predeterminado 20

    Devuelve
    -------
    pd.Series
    """
    atr_sma = atr_val.rolling(window=period, min_periods=period).mean()
    return atr_val / atr_sma


# ---------------------------------------------------------------------------
# Volatilidad realizada móvil
# ---------------------------------------------------------------------------

def rolling_std_return(
    log_returns: pd.Series,
    periods: List[int] | None = None,
) -> pd.DataFrame:
    """Desviación estándar móvil de los rendimientos logarítmicos (volatilidad realizada).

    Fórmula
    -------
        σ_N(t) = std(log_return[t−N+1 .. t])

    Parámetros
    ----------
    log_returns : pd.Series
        Rendimientos logarítmicos (de ``price_features.log_return``).
    periods : list[int], predeterminado [10, 20, 60]

    Devuelve
    -------
    pd.DataFrame
        Columnas con nombre ``rvol_{period}`` para cada período.
    """
    if periods is None:
        periods = [10, 20, 60]
    result = pd.DataFrame(index=log_returns.index)
    for p in periods:
        result[f"rvol_{p}"] = log_returns.rolling(window=p, min_periods=p).std(ddof=1)
    return result
