"""
Características de tendencia
==============
Funciones puras para características de tendencia derivadas de datos OHLCV.

Cada característica es **independiente del precio nominal** (utiliza proporciones
o valores normalizados) y solo usa datos hasta el momento t (sin anticipación).

Funciones
---------
- sma
- ema
- dist_to_sma
- dist_to_ema
- macd
- adx
- sma_slope
- higher_highs_lower_lows
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Promedios móviles
# ---------------------------------------------------------------------------

def sma(close: pd.Series, periods: List[int] | None = None) -> pd.DataFrame:
    """Promedio móvil simple para varias ventanas retrospectivas.

    Fórmula
    -------
        SMA_N(t) = (1/N) * Σ_{i=0}^{N-1} close_{t-i}

    Parámetros
    ----------
    close : pd.Series
    periods : list[int], predeterminado [20, 50, 200]

    Devuelve
    -------
    pd.DataFrame
        Columnas con nombre ``sma_{period}`` para cada período.
    """
    if periods is None:
        periods = [20, 50, 200]
    result = pd.DataFrame(index=close.index)
    for p in periods:
        result[f"sma_{p}"] = close.rolling(window=p, min_periods=p).mean()
    return result


def ema(close: pd.Series, periods: List[int] | None = None) -> pd.DataFrame:
    """Promedio móvil exponencial para varias ventanas retrospectivas.

    Fórmula
    -------
        EMA_t = α · close_t + (1 − α) · EMA_{t-1},  α = 2/(N+1)

    Parámetros
    ----------
    close : pd.Series
    periods : list[int], predeterminado [12, 26, 50]

    Devuelve
    -------
    pd.DataFrame
        Columnas con nombre ``ema_{period}`` para cada período.
    """
    if periods is None:
        periods = [12, 26, 50]
    result = pd.DataFrame(index=close.index)
    for p in periods:
        result[f"ema_{p}"] = close.ewm(span=p, adjust=False, min_periods=p).mean()
    return result


# ---------------------------------------------------------------------------
# Distancia a promedios móviles
# ---------------------------------------------------------------------------

def dist_to_sma(close: pd.Series, sma_val: pd.Series) -> pd.Series:
    """Distancia relativa entre el cierre y su SMA.

    Fórmula
    -------
        d_t = (close_t - SMA_t) / SMA_t

    Parámetros
    ----------
    close : pd.Series
    sma_val : pd.Series
        Valores de SMA calculados previamente.

    Devuelve
    -------
    pd.Series
    """
    return (close - sma_val) / sma_val


def dist_to_ema(close: pd.Series, ema_val: pd.Series) -> pd.Series:
    """Distancia relativa entre el cierre y su EMA.

    Fórmula
    -------
        d_t = (close_t - EMA_t) / EMA_t

    Parámetros
    ----------
    close : pd.Series
    ema_val : pd.Series
        Valores de EMA calculados previamente.

    Devuelve
    -------
    pd.Series
    """
    return (close - ema_val) / ema_val


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Convergencia/divergencia de promedios móviles (MACD), **normalizada por el cierre**.

    Fórmula
    -------
        MACD_line   = (EMA_fast − EMA_slow) / close
        Signal_line = EMA_signal(MACD_line)
        Histogram   = MACD_line − Signal_line

    Dividir por el cierre permite comparar los valores entre activos.

    Parámetros
    ----------
    close : pd.Series
    fast, slow, signal : int
        Períodos de EMA (predeterminados: 12 / 26 / 9).

    Devuelve
    -------
    pd.DataFrame
        Columnas: ``macd_line``, ``macd_signal``, ``macd_histogram``.
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()

    macd_line = (ema_fast - ema_slow) / close
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": signal_line, "macd_histogram": histogram},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# ADX — Índice direccional promedio
# ---------------------------------------------------------------------------

def directional_movement(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Calcula ADX, +DI y -DI usando la misma convención de suavizado Wilder.

    Fórmula
    -------
    1. Rango verdadero (TR) = max(high−low, |high−prev_close|, |low−prev_close|)
    2. +DM = high − prev_high si > 0 y > (prev_low − low); de lo contrario, 0
       −DM = prev_low − low si > 0 y > (high − prev_high); de lo contrario, 0
    3. TR, +DM y −DM suavizados con el método de Wilder (period)
    4. +DI = 100 · +DM suavizado / TR suavizado
       −DI = 100 · −DM suavizado / TR suavizado
    5. DX  = 100 · |+DI − −DI| / (+DI + −DI)
    6. ADX = DX suavizado con el método de Wilder durante *period* velas.

    Ya está en una escala de 0 a 100 (no requiere normalización adicional).

    Parámetros
    ----------
    high, low, close : pd.Series
    period : int, predeterminado 14

    Devuelve
    -------
    pd.DataFrame
        Columnas ``adx``, ``plus_di`` y ``minus_di`` en escala 0-100.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # Rango verdadero
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Movimiento direccional
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=close.index)

    # Función auxiliar de suavizado de Wilder (suma inicial y luego recursión)
    def _wilder_smooth(series: pd.Series, n: int) -> pd.Series:
        values = series.to_numpy(dtype=float)
        result = values.copy()
        # Se necesitan n valores válidos para el primer valor suavizado
        result[:n] = np.nan
        first_val = np.nansum(values[1 : n + 1])  # skip index 0 (NaN from shift)
        result[n] = first_val
        for i in range(n + 1, len(result)):
            result[i] = result[i - 1] - result[i - 1] / n + values[i]
        return pd.Series(result, index=series.index)

    atr_smooth = _wilder_smooth(tr, period)
    plus_dm_smooth = _wilder_smooth(plus_dm, period)
    minus_dm_smooth = _wilder_smooth(minus_dm, period)

    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth

    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    # ADX es el DX suavizado mediante el método de Wilder
    adx_values = np.full(len(close), np.nan, dtype=float)
    # El primer ADX es el promedio de los primeros *period* valores DX disponibles
    dx_values = dx.to_numpy(dtype=float)
    valid_dx = np.flatnonzero(~np.isnan(dx_values))
    if valid_dx.size:
        loc = int(valid_dx[0])
        if loc + period <= len(dx):
            adx_values[loc + period - 1] = np.nanmean(dx_values[loc : loc + period])
            for i in range(loc + period, len(dx)):
                adx_values[i] = (
                    adx_values[i - 1] * (period - 1) + dx_values[i]
                ) / period

    return pd.DataFrame(
        {
            "adx": pd.Series(adx_values, index=close.index),
            "plus_di": plus_di,
            "minus_di": minus_di,
        },
        index=close.index,
    )


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Índice direccional promedio (ADX) con suavizado Wilder."""
    return directional_movement(high, low, close, period=period)["adx"]


# ---------------------------------------------------------------------------
# Pendiente de SMA
# ---------------------------------------------------------------------------

def sma_slope(sma_val: pd.Series, period: int = 5) -> pd.Series:
    """Pendiente normalizada de una SMA a lo largo de *period* velas.

    Fórmula
    -------
        slope_t = (SMA_t − SMA_{t−period}) / SMA_{t−period}

    Es el cambio porcentual de la SMA durante *period* velas y expresa la
    inclinación de la tendencia en términos relativos.

    Parámetros
    ----------
    sma_val : pd.Series
        SMA calculada previamente.
    period : int, predeterminado 5

    Devuelve
    -------
    pd.Series
    """
    prev = sma_val.shift(period)
    return (sma_val - prev) / prev


# ---------------------------------------------------------------------------
# Máximos crecientes / mínimos decrecientes
# ---------------------------------------------------------------------------

def higher_highs_lower_lows(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """Cuenta máximos crecientes (HH) y mínimos decrecientes (LL) consecutivos en una ventana.

    Algoritmo (sin anticipación)
    -------------------------
    En cada vela *t*, retrocede *window* velas. Recorre esa subventana y cuenta:
    - **higher_highs**: la secuencia más larga de días consecutivos en que el
      máximo diario supera al del día anterior, con término en t o antes.
    - **lower_lows**: la secuencia más larga de días consecutivos en que el
      mínimo diario es inferior al del día anterior, con término en t o antes.

    Parámetros
    ----------
    high, low : pd.Series
    window : int, predeterminado 20

    Devuelve
    -------
    pd.DataFrame
        Columnas: ``higher_highs_count``, ``lower_lows_count``.
    """
    if window < 1:
        raise ValueError("window debe ser al menos 1")
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    n = len(high_values)
    indices = np.arange(n)

    higher = np.zeros(n, dtype=bool)
    lower = np.zeros(n, dtype=bool)
    higher[1:] = high_values[1:] > high_values[:-1]
    lower[1:] = low_values[1:] < low_values[:-1]
    last_higher_break = np.maximum.accumulate(np.where(~higher, indices, -1))
    last_lower_break = np.maximum.accumulate(np.where(~lower, indices, -1))
    higher_counts = np.minimum(indices - last_higher_break, window).astype(float)
    lower_counts = np.minimum(indices - last_lower_break, window).astype(float)
    higher_counts[:window] = np.nan
    lower_counts[:window] = np.nan

    return pd.DataFrame(
        {
            "higher_highs_count": higher_counts,
            "lower_lows_count": lower_counts,
        },
        index=high.index,
    )
