"""
Características de precio
==============
Funciones puras que derivan características de precios a partir de columnas OHLCV.

Cada característica es **independiente del precio nominal** (proporciones,
rendimientos logarítmicos o distancias relativas) y utiliza únicamente datos
hasta el momento t (sin anticipación).

Funciones
---------
- log_return
- simple_return
- intraday_range
- body_ratio
- gap
- high_low_position
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(close: pd.Series) -> pd.Series:
    """Rendimiento logarítmico.

    Fórmula
    -------
        r_t = ln(close_t / close_{t-1})

    Parámetros
    ----------
    close : pd.Series
        Precios de cierre.

    Devuelve
    -------
    pd.Series
        Rendimientos logarítmicos. El primer valor es NaN (no hay una vela anterior).
    """
    return np.log(close / close.shift(1))


def simple_return(close: pd.Series) -> pd.Series:
    """Rendimiento simple (aritmético).

    Fórmula
    -------
        r_t = close_t / close_{t-1} - 1

    Parámetros
    ----------
    close : pd.Series

    Devuelve
    -------
    pd.Series
        Rendimientos simples. El primer valor es NaN.
    """
    return close / close.shift(1) - 1


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Rango intradiario normalizado.

    Fórmula
    -------
        range_t = (high_t - low_t) / close_t

    Dividir por el cierre permite comparar el rango entre activos con distintos
    precios nominales.

    Parámetros
    ----------
    high, low, close : pd.Series

    Devuelve
    -------
    pd.Series
    """
    return (high - low) / close


def body_ratio(
    open_: pd.Series, close: pd.Series, high: pd.Series, low: pd.Series
) -> pd.Series:
    """Proporción del cuerpo de la vela.

    Fórmula
    -------
        body_ratio_t = |close_t - open_t| / (high_t - low_t)

    Mide qué fracción del rango total de la vela corresponde al cuerpo.
    Devuelve NaN cuando high == low (vela con rango cero).

    Parámetros
    ----------
    open_, close, high, low : pd.Series

    Devuelve
    -------
    pd.Series
        Valores en [0, 1]. NaN para velas con rango cero.
    """
    full_range = high - low
    # Reemplazar el rango cero por NaN para evitar la división por cero
    full_range = full_range.replace(0, np.nan)
    return (close - open_).abs() / full_range


def gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Brecha nocturna / entre velas.

    Fórmula
    -------
        gap_t = (open_t - close_{t-1}) / close_{t-1}

    Parámetros
    ----------
    open_ : pd.Series
        Precios de apertura de la vela actual.
    close : pd.Series
        Precios de cierre (se desplazan internamente para obtener el cierre anterior).

    Devuelve
    -------
    pd.Series
        Brecha relativa. El primer valor es NaN.
    """
    prev_close = close.shift(1)
    return (open_ - prev_close) / prev_close


def high_low_position(
    close: pd.Series, high: pd.Series, low: pd.Series
) -> pd.Series:
    """Posición del cierre dentro del rango máximo-mínimo.

    Fórmula
    -------
        hlp_t = (close_t - low_t) / (high_t - low_t)

    Devuelve 0 cuando close == low y 1 cuando close == high.
    Devuelve NaN cuando high == low (vela con rango cero).

    Parámetros
    ----------
    close, high, low : pd.Series

    Devuelve
    -------
    pd.Series
        Valores en [0, 1]. NaN para velas con rango cero.
    """
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    return (close - low) / hl_range
