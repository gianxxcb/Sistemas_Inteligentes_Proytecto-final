"""
Etiquetas objetivo: cálculo y clasificación del rendimiento futuro.

Estas son las variables objetivo de predicción, **no** características. Por definición,
utilizan datos futuros (anticipación), por lo que NUNCA deben aparecer como entradas del modelo.

Horizontes de pronóstico: 7, 14, 30 y 90 días calendario.
Los periodos de indicadores técnicos son independientes de estos horizontes.

Esquema de clasificación
---------------------
+1  UP      : forward_return >  threshold
-1  DOWN    : forward_return < -threshold
 0  LATERAL : |forward_return| ≤ threshold

Umbral adaptativo
------------------
threshold_t = max(cost_threshold, k * realized_vol_horizon_t)

Donde:
  cost_threshold = 0.003   (0.30 % de costos de negociación de ida y vuelta)
  k = 1.0 (factor de escala ajustable)
  realized_vol_horizon = std(log_returns durante los últimos `horizon` días) * sqrt(horizon)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Funciones principales
# ---------------------------------------------------------------------------

def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Rendimiento logarítmico desde t hasta t + horizon.

    Fórmula
    -------
    fr_t = ln(close_{t+horizon} / close_t)

    Por definición, esto SÍ utiliza información futura: es la variable objetivo.

    Parámetros
    ----------
    close : pd.Series
        Serie de precios de cierre.
    horizon : int
        Cantidad de días calendario hacia adelante.

    Devuelve
    -------
    pd.Series
        Rendimiento logarítmico futuro. Las últimas ``horizon`` filas son NaN.
    """
    return np.log(close.shift(-horizon) / close)


def realized_vol_horizon(
    log_returns: pd.Series, horizon: int, min_periods: int | None = None
) -> pd.Series:
    """Volatilidad realizada móvil ajustada al horizonte indicado.

    Fórmula
    -------
    vol_t = std(r_{t-horizon+1} … r_t) * √horizon

    Parámetros
    ----------
    log_returns : pd.Series
        Rendimientos logarítmicos (retrospectivos, sin anticipación).
    horizon : int
        Longitud de la ventana Y factor de anualización.
    min_periods : int or None
        Cantidad mínima de observaciones para una estimación válida. El valor predeterminado es ``horizon``.

    Devuelve
    -------
    pd.Series
        Volatilidad móvil ajustada al horizonte.
    """
    if min_periods is None:
        min_periods = horizon
    return log_returns.rolling(window=horizon, min_periods=min_periods).std() * np.sqrt(
        horizon
    )


def adaptive_threshold(
    log_returns: pd.Series,
    horizon: int,
    k: float = 1.0,
    cost_threshold: float = 0.003,
) -> pd.Series:
    """Umbral de clasificación adaptativo por vela.

    Fórmula
    -------
    threshold_t = max(cost_threshold, k * realized_vol_horizon_t)

    Parámetros
    ----------
    log_returns : pd.Series
        Rendimientos logarítmicos para estimar la volatilidad (retrospectivos).
    horizon : int
        Horizonte en días calendario.
    k : float
        Factor de escala del umbral basado en volatilidad (predeterminado: 1.0).
    cost_threshold : float
        Umbral mínimo basado en los costos de operación de ida y vuelta (predeterminado: 0.003).

    Devuelve
    -------
    pd.Series
        Umbral adaptativo por vela.
    """
    vol = realized_vol_horizon(log_returns, horizon)
    return np.maximum(cost_threshold, k * vol)


def forward_return_class(
    fwd_return: pd.Series, threshold: pd.Series | float
) -> pd.Series:
    """Clasifica el rendimiento futuro como +1 (UP), -1 (DOWN) o 0 (LATERAL).

    Fórmula
    -------
    class_t =  +1  si fwd_return_t >  threshold_t
              -1  si fwd_return_t < -threshold_t
               0  en otro caso

    Parámetros
    ----------
    fwd_return : pd.Series
        Rendimiento logarítmico futuro.
    threshold : pd.Series or float
        Umbral adaptativo o fijo.

    Devuelve
    -------
    pd.Series
        Etiquetas de clase enteras: -1, 0, +1. NaN donde las entradas sean NaN.
    """
    result = pd.Series(np.nan, index=fwd_return.index, dtype="float64")
    valid = fwd_return.notna()
    if isinstance(threshold, pd.Series):
        valid = valid & threshold.notna()

    result[valid & (fwd_return > threshold)] = 1.0
    result[valid & (fwd_return < -threshold)] = -1.0
    # Lateral: válido, pero no es UP ni DOWN
    result[valid & result.isna()] = 0.0

    return result.astype("Int64")  # entero que admite valores nulos


# ---------------------------------------------------------------------------
# Función auxiliar: calcula todos los horizontes de pronóstico a la vez
# ---------------------------------------------------------------------------

HORIZONS = (7, 14, 30, 90)
MAX_HORIZON_DAYS = max(HORIZONS)
HORIZON_WINDOWS = {7: 30, 14: 60, 30: 60, 90: 120}
FEATURE_WARMUP_DAYS = 200
HORIZON_GRID = (3, 5, 7, 10, 14, 21, 30, 45, 60, 90)
HORIZON_LABELS = {
    7: "1 semana",
    14: "2 semanas",
    30: "1 mes",
    90: "3 meses",
}


def compute_labels(
    df: pd.DataFrame,
    close_col: str = "close",
    log_return_col: str = "log_return",
    horizons: list[int] | None = None,
    k: float = 1.0,
    cost_threshold: float = 0.003,
) -> pd.DataFrame:
    """Calcula las etiquetas de rendimiento futuro para todos los horizontes y las añade a *df*.

    Columnas nuevas por horizonte *h*:
        ``fwd_return_{h}d``    — rendimiento logarítmico futuro
        ``fwd_threshold_{h}d`` — umbral adaptativo utilizado
        ``fwd_class_{h}d``     — etiqueta de clase (-1, 0, +1)

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe de un solo activo con ``close_col`` y ``log_return_col``.
    close_col : str
        Columna con los precios de cierre.
    log_return_col : str
        Columna con rendimientos logarítmicos (necesarios para el umbral adaptativo).
    horizons : list[int] or None
        Lista de horizontes futuros en días calendario
        (predeterminado: [7, 14, 30, 90]).
    k : float
        Factor de escala de volatilidad para el umbral adaptativo.
    cost_threshold : float
        Umbral mínimo (límite inferior basado en costos de ida y vuelta).

    Devuelve
    -------
    pd.DataFrame
        Copia de *df* con 3 columnas nuevas por horizonte.
    """
    if horizons is None:
        horizons = HORIZONS

    result = df.copy()
    close = result[close_col]
    log_ret = result[log_return_col]

    for h in horizons:
        fwd_ret = forward_return(close, h)
        thresh = adaptive_threshold(log_ret, h, k=k, cost_threshold=cost_threshold)
        fwd_cls = forward_return_class(fwd_ret, thresh)

        result[f"fwd_return_{h}d"] = fwd_ret
        result[f"fwd_threshold_{h}d"] = thresh
        result[f"fwd_class_{h}d"] = fwd_cls

    return result
