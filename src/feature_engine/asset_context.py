"""
Características de contexto del activo: características macro por activo para el metamodelo.

Estas características miden propiedades estructurales de cada activo (régimen de
volatilidad, correlación de mercado, liquidez y antigüedad) y están diseñadas para
alimentar un metamodelo que selecciona o pondera modelos especializados.

Todas las salidas son independientes del precio (proporciones, rangos y estadísticas móviles).
Sin anticipación: cada valor en t utiliza únicamente datos hasta t.
"""

import numpy as np
import pandas as pd


def rolling_volatility_historical(
    log_returns: pd.Series, window: int = 90
) -> pd.Series:
    """Volatilidad realizada histórica anualizada para calendario cripto 24/7.

    Fórmula
    -------
    σ_t = std(r_{t-window+1} … r_t) * √365

    Parámetros
    ----------
    log_returns : pd.Series
        Serie de rendimientos logarítmicos: ln(close_t / close_{t-1}).
    window : int
        Ventana móvil en días de negociación (predeterminado: 90).

    Devuelve
    -------
    pd.Series
        Volatilidad móvil anualizada. NaN en las primeras ``window - 1`` filas.
    """
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(365)


def rolling_correlation_with_btc(
    asset_returns: pd.Series,
    btc_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Correlación móvil de Pearson entre los rendimientos logarítmicos del activo y los de BTC.

    Fórmula
    -------
    ρ_t = corr(r^asset_{t-w+1…t}, r^BTC_{t-w+1…t})

    Parámetros
    ----------
    asset_returns : pd.Series
        Rendimientos logarítmicos del activo.
    btc_returns : pd.Series
        Rendimientos logarítmicos de BTC, alineados con el mismo índice de fechas.
    window : int
        Ventana móvil en días de negociación (predeterminado: 60).

    Devuelve
    -------
    pd.Series
        Correlación móvil en [-1, 1]. NaN durante el período de inicialización.
    """
    return asset_returns.rolling(window=window, min_periods=window).corr(btc_returns)


def relative_volume_rank(
    cross_asset_volumes: pd.DataFrame,
    asset_symbol: str,
) -> pd.Series:
    """Cross-sectional percentile rank of trailing liquidity by date.

    The input is expected to contain trailing quote-volume means indexed by date.
    Each date is ranked independently, preventing full-history leakage.
    """
    if asset_symbol not in cross_asset_volumes.columns:
        raise KeyError(f"No hay volumen cross-sectional para {asset_symbol}")
    return cross_asset_volumes.rank(axis=1, pct=True)[asset_symbol]


def asset_age_days(first_date: pd.Timestamp, current_date: pd.Timestamp) -> int:
    """Cantidad de días calendario desde la primera fecha observada del activo.

    Fórmula
    -------
    age = (current_date - first_date).days

    Parámetros
    ----------
    first_date : pd.Timestamp
        Primera fecha en los datos del activo.
    current_date : pd.Timestamp
        Fecha para la que se calcula la antigüedad.

    Devuelve
    -------
    int
        Antigüedad en días calendario (≥ 0).
    """
    return max(0, (current_date - first_date).days)


def compute_asset_context(
    df: pd.DataFrame,
    btc_returns: pd.Series,
    all_assets_avg_volumes: pd.DataFrame,
    asset_symbol: str,
    log_return_col: str = "log_return",
    quote_volume_col: str = "quote_volume",
    date_col: str = "date",
    relative_strength_period: int = 20,
) -> pd.DataFrame:
    """Calcula todas las características de contexto del activo y las añade al dataframe.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataframe de un solo activo que incluya al menos ``log_return_col``,
        ``quote_volume_col`` y ``date_col``.
    btc_returns : pd.Series
        Rendimientos logarítmicos de BTC indexados por fecha y alineados con el mismo calendario.
    all_assets_avg_volumes : pd.DataFrame
        Medias retrospectivas del volumen cotizado por fecha y símbolo.
    asset_symbol : str
        Símbolo de la columna cuyo rango temporal se calcula.
    log_return_col : str
        Nombre de la columna de rendimientos logarítmicos en *df*.
    quote_volume_col : str
        Nombre de la columna de volumen cotizado en *df*.
    date_col : str
        Nombre de la columna de fechas en *df*.

    Devuelve
    -------
    pd.DataFrame
        Copia de *df* con las nuevas columnas:
        ``ctx_volatility_90d``, ``ctx_corr_btc_60d``,
        ``ctx_volume_rank``, ``ctx_age_days``.
    """
    result = df.copy()

    # Asegurar que la columna de fechas tenga tipo datetime
    dates = pd.to_datetime(result[date_col])

    # 1. Volatilidad realizada móvil de 90 días
    result["ctx_volatility_90d"] = rolling_volatility_historical(
        result[log_return_col], window=90
    )

    # 2. Correlación móvil de 60 días con BTC
    # Dar formato a las fechas como cadenas YYYY-MM-DD para una coincidencia determinista de claves
    date_strings = pd.to_datetime(result[date_col], utc=True).dt.strftime("%Y-%m-%d")
    btc_series = btc_returns.copy()
    if isinstance(btc_series.index, pd.DatetimeIndex):
        btc_series.index = btc_series.index.strftime("%Y-%m-%d")
    else:
        btc_series.index = btc_series.index.astype(str)

    btc_map = btc_series.to_dict()
    btc_aligned = date_strings.map(btc_map).astype(float)
    
    result["ctx_corr_btc_60d"] = rolling_correlation_with_btc(
        result[log_return_col].reset_index(drop=True),
        btc_aligned.reset_index(drop=True),
        window=60,
    )

    # 3. Ranking de liquidez cross-sectional por fecha, sin promediar el futuro
    volume_ranks = relative_volume_rank(all_assets_avg_volumes, asset_symbol)
    volume_ranks.index = volume_ranks.index.astype(str)
    result["ctx_volume_rank"] = date_strings.map(volume_ranks.to_dict())

    # 4. Fuerza relativa: diferencia de retornos logarítmicos retrospectivos
    result["relative_strength_btc_20d"] = (
        result[log_return_col] - btc_aligned
    ).rolling(relative_strength_period, min_periods=relative_strength_period).sum()

    # 5. Antigüedad del activo en días (por fila)
    first_date = dates.min()
    result["ctx_age_days"] = dates.apply(
        lambda d: asset_age_days(first_date, d)
    )

    return result
