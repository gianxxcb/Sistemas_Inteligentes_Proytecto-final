"""
Características de estructura: soporte / resistencia
==========================================
Funciones puras para características estructurales (canales Donchian, compresión
del rango, puntos pivote y zonas de soporte/resistencia).

Todas las características son **independientes del precio nominal** (distancias
relativas) y solo utilizan datos hasta el momento t (sin anticipación). La detección
de pivotes requiere un retraso de confirmación de *window* velas.

Funciones
---------
- donchian_channels
- range_compression
- pivot_points
- support_resistance_zones
- bars_since_pivot
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canales Donchian
# ---------------------------------------------------------------------------

def donchian_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """Canales Donchian con distancias relativas.

    Fórmula
    -------
        Upper_t = max(high[t-period .. t-1])
        Lower_t = min(low[t-period .. t-1])
        dist_upper = (Upper − close) / close
        dist_lower = (close − Lower) / close

    Parámetros
    ----------
    high, low, close : pd.Series
    period : int, predeterminado 20

    Devuelve
    -------
    pd.DataFrame
        Columnas: ``dc_upper``, ``dc_lower``, ``dc_dist_upper``,
        ``dc_dist_lower``.
    """
    upper = high.shift(1).rolling(window=period, min_periods=period).max()
    lower = low.shift(1).rolling(window=period, min_periods=period).min()
    dist_upper = (upper - close) / close
    dist_lower = (close - lower) / close

    return pd.DataFrame(
        {
            "dc_upper": upper,
            "dc_lower": lower,
            "dc_dist_upper": dist_upper,
            "dc_dist_lower": dist_lower,
        },
        index=close.index,
    )


def breakout_retest_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_value: pd.Series,
    upper_channel: pd.Series,
    lower_channel: pd.Series,
    max_retest_wait: int = 8,
    tolerance_atr: float = 0.5,
) -> pd.DataFrame:
    """Emit causal channel-breakout and break/retest event features.

    Boundaries are calculated from completed bars before the current bar. A
    retest is recorded only on the candle that touches the stored breakout level.
    """
    if max_retest_wait < 1:
        raise ValueError("max_retest_wait debe ser al menos 1")
    if tolerance_atr < 0:
        raise ValueError("tolerance_atr no puede ser negativo")

    n = len(close)
    values = {
        "breakout_up": np.zeros(n, dtype=float),
        "breakout_down": np.zeros(n, dtype=float),
        "retest_up_ok": np.zeros(n, dtype=float),
        "retest_down_ok": np.zeros(n, dtype=float),
        "retest_failed": np.zeros(n, dtype=float),
        "retest_expired": np.zeros(n, dtype=float),
        "retest_age_bars": np.full(n, np.nan),
        "retest_pending": np.zeros(n, dtype=float),
    }

    up_pending = None
    down_pending = None
    close_values = close.to_numpy(dtype=float)
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    atr_values = atr_value.to_numpy(dtype=float)
    upper_values = upper_channel.to_numpy(dtype=float)
    lower_values = lower_channel.to_numpy(dtype=float)

    for i in range(n):
        up_break = (
            i > 0
            and np.isfinite(upper_values[i])
            and np.isfinite(upper_values[i - 1])
            and close_values[i] > upper_values[i]
            and close_values[i - 1] <= upper_values[i - 1]
        )
        down_break = (
            i > 0
            and np.isfinite(lower_values[i])
            and np.isfinite(lower_values[i - 1])
            and close_values[i] < lower_values[i]
            and close_values[i - 1] >= lower_values[i - 1]
        )

        if up_break:
            values["breakout_up"][i] = 1.0
            up_pending = (upper_values[i], i)
        elif up_pending is not None:
            level, breakout_idx = up_pending
            age = i - breakout_idx
            if age > max_retest_wait:
                values["retest_expired"][i] = 1.0
                up_pending = None
            else:
                tolerance = atr_values[i] * tolerance_atr if np.isfinite(atr_values[i]) else 0.0
                if close_values[i] < level:
                    values["retest_failed"][i] = 1.0
                    values["retest_age_bars"][i] = age
                    up_pending = None
                elif low_values[i] <= level + tolerance:
                    values["retest_up_ok"][i] = 1.0
                    values["retest_age_bars"][i] = age
                    up_pending = None

        if down_break:
            values["breakout_down"][i] = 1.0
            down_pending = (lower_values[i], i)
        elif down_pending is not None:
            level, breakout_idx = down_pending
            age = i - breakout_idx
            if age > max_retest_wait:
                values["retest_expired"][i] = 1.0
                down_pending = None
            else:
                tolerance = atr_values[i] * tolerance_atr if np.isfinite(atr_values[i]) else 0.0
                if close_values[i] > level:
                    values["retest_failed"][i] = 1.0
                    values["retest_age_bars"][i] = age
                    down_pending = None
                elif high_values[i] >= level - tolerance:
                    values["retest_down_ok"][i] = 1.0
                    values["retest_age_bars"][i] = age
                    down_pending = None

        values["retest_pending"][i] = float(
            up_pending is not None or down_pending is not None
        )

    return pd.DataFrame(values, index=close.index)


def divergence_features(
    high: pd.Series,
    low: pd.Series,
    oscillator: pd.Series,
    prefix: str,
    window: int = 5,
    min_pivot_separation: int = 5,
    min_price_change_pct: float = 0.005,
    min_oscillator_difference: float = 2.0,
    max_age_bars: int = 40,
) -> pd.DataFrame:
    """Detect regular and hidden divergences at confirmed price-pivot timestamps.

    Oscillator values are sampled at the price-pivot candle, but emitted only
    when that pivot has the required ``window`` bars of confirmation.
    """
    if window < 1 or min_pivot_separation < 1 or max_age_bars < 1:
        raise ValueError("Las ventanas y distancias de pivote deben ser positivas")
    if min_price_change_pct < 0 or min_oscillator_difference < 0:
        raise ValueError("Los filtros mínimos de divergencia no pueden ser negativos")

    n = len(oscillator)
    signal_names = (
        "regular_bullish",
        "regular_bearish",
        "hidden_bullish",
        "hidden_bearish",
        "confirmed",
    )
    result_values = {
        f"{prefix}_divergence_{name}": np.zeros(n, dtype=float)
        for name in signal_names
    }
    result_values[f"{prefix}_divergence_price_delta_pct"] = np.zeros(n, dtype=float)
    result_values[f"{prefix}_divergence_oscillator_delta"] = np.zeros(n, dtype=float)
    result_values[f"{prefix}_divergence_pivot_distance"] = np.zeros(n, dtype=float)
    result_values[f"{prefix}_divergence_age_bars"] = np.full(n, np.nan, dtype=float)

    confirmed_pivots = pivot_points(high, low, window=window)
    confirmed_values = {
        column: confirmed_pivots[column].to_numpy(dtype=float)
        for column in ("pivot_high", "pivot_low")
    }
    oscillator_values = oscillator.to_numpy(dtype=float)
    price_pivots = {"high": [], "low": []}
    event_indices = []

    for confirm_idx in range(n):
        center_idx = confirm_idx - window
        if center_idx < 0 or not np.isfinite(oscillator_values[center_idx]):
            continue

        for side, column in (("high", "pivot_high"), ("low", "pivot_low")):
            price_value = confirmed_values[column][confirm_idx]
            if not np.isfinite(price_value):
                continue

            current = (center_idx, float(price_value), oscillator_values[center_idx])
            previous = next(
                (
                    point for point in reversed(price_pivots[side])
                    if current[0] - point[0] >= min_pivot_separation
                ),
                None,
            )
            price_pivots[side].append(current)
            if previous is None or abs(previous[1]) < 1e-12:
                continue

            price_delta = (current[1] - previous[1]) / abs(previous[1])
            oscillator_delta = current[2] - previous[2]
            if (
                abs(price_delta) < min_price_change_pct
                or abs(oscillator_delta) < min_oscillator_difference
            ):
                continue

            if side == "low":
                signal = (
                    "regular_bullish"
                    if price_delta < 0 and oscillator_delta > 0
                    else "hidden_bullish"
                    if price_delta > 0 and oscillator_delta < 0
                    else None
                )
            else:
                signal = (
                    "regular_bearish"
                    if price_delta > 0 and oscillator_delta < 0
                    else "hidden_bearish"
                    if price_delta < 0 and oscillator_delta > 0
                    else None
                )
            if signal is None:
                continue

            result_values[f"{prefix}_divergence_{signal}"][confirm_idx] = 1.0
            result_values[f"{prefix}_divergence_confirmed"][confirm_idx] = 1.0
            result_values[f"{prefix}_divergence_price_delta_pct"][confirm_idx] = price_delta
            result_values[f"{prefix}_divergence_oscillator_delta"][confirm_idx] = oscillator_delta
            result_values[f"{prefix}_divergence_pivot_distance"][confirm_idx] = current[0] - previous[0]
            event_indices.append(confirm_idx)

    if event_indices:
        indices = np.arange(n)
        event_mask = np.zeros(n, dtype=bool)
        event_mask[event_indices] = True
        last_event_indices = np.maximum.accumulate(
            np.where(event_mask, indices, -1)
        )
        ages = indices - last_event_indices
        valid_ages = (last_event_indices >= 0) & (ages <= max_age_bars)
        result_values[f"{prefix}_divergence_age_bars"][valid_ages] = ages[valid_ages]

    return pd.DataFrame(result_values, index=oscillator.index)


# ---------------------------------------------------------------------------
# Compresión del rango
# ---------------------------------------------------------------------------

def range_compression(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Compresión del rango: rango actual frente al máximo en la ventana retrospectiva.

    Fórmula
    -------
        current_range_t = high_t − low_t
        max_range_t     = max(high − low, period)
        compression_t   = current_range_t / max_range_t

    Los valores cercanos a 0 indican una compresión intensa (squeeze); los cercanos
    a 1 indican el rango más amplio de la ventana.

    Parámetros
    ----------
    high, low : pd.Series
    period : int, predeterminado 20

    Devuelve
    -------
    pd.Series
    """
    bar_range = high - low
    max_range = bar_range.rolling(window=period, min_periods=period).max()
    max_range = max_range.replace(0, np.nan)
    return bar_range / max_range


# ---------------------------------------------------------------------------
# Puntos pivote
# ---------------------------------------------------------------------------

def pivot_points(
    high: pd.Series,
    low: pd.Series,
    window: int = 5,
) -> pd.DataFrame:
    """Detecta máximos locales (pivotes altos) y mínimos (pivotes bajos) con un
    retraso de confirmación.

    Algoritmo (sin anticipación)
    -------------------------
    Una vela en el índice *c* es un **pivote alto** si ``high[c]`` es el máximo de
    ``high[c−window .. c+window]``. Sin embargo, solo puede *confirmarse* cuando se
    observa la vela ``c + window``, por lo que el pivote se registra en el índice
    ``c + window`` (la vela de confirmación).

    Los pivotes bajos se determinan de forma análoga usando el mínimo de ``low``.

    Las series devueltas contienen el **precio** del pivote confirmado en la
    *vela de confirmación* y NaN en las demás posiciones.

    Parámetros
    ----------
    high, low : pd.Series
    window : int, predeterminado 5
        Número de velas requeridas a cada lado para confirmar un pivote.

    Devuelve
    -------
    pd.DataFrame
        Columnas: ``pivot_high``, ``pivot_low``.
        Los valores corresponden al precio del pivote en la vela donde se confirma;
        NaN en las demás posiciones.
    """
    if window < 1:
        raise ValueError("window debe ser al menos 1")
    n = len(high)
    ph = np.full(n, np.nan, dtype=float)
    pl = np.full(n, np.nan, dtype=float)
    window_size = 2 * window + 1
    if n < window_size:
        return pd.DataFrame(
            {"pivot_high": ph, "pivot_low": pl},
            index=high.index,
        )

    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    high_windows = np.lib.stride_tricks.sliding_window_view(high_values, window_size)
    low_windows = np.lib.stride_tricks.sliding_window_view(low_values, window_size)
    high_centers = high_values[window:n - window]
    low_centers = low_values[window:n - window]
    high_max = high_windows.max(axis=1)
    low_min = low_windows.min(axis=1)
    unique_high = np.sum(high_windows == high_max[:, None], axis=1) == 1
    unique_low = np.sum(low_windows == low_min[:, None], axis=1) == 1

    high_pivot_offsets = np.flatnonzero((high_centers == high_max) & unique_high)
    low_pivot_offsets = np.flatnonzero((low_centers == low_min) & unique_low)
    high_confirm_indices = high_pivot_offsets + 2 * window
    low_confirm_indices = low_pivot_offsets + 2 * window
    ph[high_confirm_indices] = high_centers[high_pivot_offsets]
    pl[low_confirm_indices] = low_centers[low_pivot_offsets]

    return pd.DataFrame(
        {
            "pivot_high": pd.Series(ph, index=high.index),
            "pivot_low": pd.Series(pl, index=low.index),
        },
        index=high.index,
    )


# ---------------------------------------------------------------------------
# Zonas de soporte / resistencia
# ---------------------------------------------------------------------------

def support_resistance_zones(
    pivot_highs: pd.Series,
    pivot_lows: pd.Series,
    close: pd.Series,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """Distancia y fuerza de las zonas de soporte/resistencia más cercanas.

    Algoritmo (sin anticipación)
    -------------------------
    En cada vela *t*, reúne todos los pivotes confirmados hasta *t*, inclusive.
    Los agrupa en **zonas**, combinando los pivotes cuya distancia relativa al
    centro de la zona esté dentro de ``tolerance``. Cada zona tiene:
    - un *nivel* = precio promedio de sus pivotes
    - una *fuerza* = cantidad de contactos con pivotes

    Luego calcula:
    - **dist_support**: distancia relativa a la zona de soporte más cercana bajo el cierre
    - **dist_resistance**: distancia relativa a la zona de resistencia más cercana sobre el cierre
    - **support_strength**: cantidad de contactos de la zona de soporte más cercana
    - **resistance_strength**: cantidad de contactos de la zona de resistencia más cercana

    Las zonas se reconstruyen de forma incremental; solo se utilizan pivotes
    confirmados en *t* o antes, para evitar la anticipación.

    Parámetros
    ----------
    pivot_highs : pd.Series
        Proviene de ``pivot_points``; es NaN salvo en las velas de confirmación.
    pivot_lows : pd.Series
    close : pd.Series
    tolerance : float, predeterminado 0.02
        Umbral de distancia relativa para combinar pivotes en una zona.

    Devuelve
    -------
    pd.DataFrame
        Columnas: ``sr_dist_support``, ``sr_dist_resistance``,
        ``sr_support_strength``, ``sr_resistance_strength``.
    """
    n = len(close)
    dist_sup = np.full(n, np.nan)
    dist_res = np.full(n, np.nan)
    str_sup = np.full(n, np.nan)
    str_res = np.full(n, np.nan)

    # zones remain in insertion order; the sorted index makes nearest-zone
    # lookups logarithmic while preserving the original first-match rule.
    zones: list[list[float]] = []
    sorted_zone_levels: list[tuple[float, int]] = []

    def _add_pivot(price: float) -> None:
        """Añade un pivote a la lista de zonas y lo combina si está dentro de la tolerancia."""
        if zones:
            levels = np.fromiter(
                (zone[0] / zone[1] for zone in zones),
                dtype=float,
                count=len(zones),
            )
            matching = np.flatnonzero(np.abs(price - levels) / levels <= tolerance)
            if matching.size:
                zone_idx = int(matching[0])
                old_level = levels[zone_idx]
                zones[zone_idx][0] += price
                zones[zone_idx][1] += 1
                new_level = zones[zone_idx][0] / zones[zone_idx][1]
                old_key = (float(old_level), zone_idx)
                position = bisect_left(sorted_zone_levels, old_key)
                if position == len(sorted_zone_levels) or sorted_zone_levels[position] != old_key:
                    raise RuntimeError("El índice de zonas soporte/resistencia perdió sincronía.")
                del sorted_zone_levels[position]
                insort(sorted_zone_levels, (new_level, zone_idx))
                return

        zone_idx = len(zones)
        zones.append([price, 1.0])
        insort(sorted_zone_levels, (price, zone_idx))

    close_vals = close.values
    ph_vals = pivot_highs.values
    pl_vals = pivot_lows.values

    for t in range(n):
        # Incorporar cualquier pivote confirmado en la vela t
        if not np.isnan(ph_vals[t]):
            _add_pivot(ph_vals[t])
        if not np.isnan(pl_vals[t]):
            _add_pivot(pl_vals[t])

        if len(zones) == 0:
            continue

        c = close_vals[t]
        if np.isnan(c) or c == 0:
            continue

        upper_bound = bisect_right(sorted_zone_levels, (c, np.inf))
        if upper_bound:
            support_level = sorted_zone_levels[upper_bound - 1][0]
            first_support = bisect_left(sorted_zone_levels, (support_level, -1))
            support_idx = sorted_zone_levels[first_support][1]
            dist_sup[t] = (c - support_level) / c
            str_sup[t] = zones[support_idx][1]
        if upper_bound < len(sorted_zone_levels):
            resistance_level, resistance_idx = sorted_zone_levels[upper_bound]
            dist_res[t] = (resistance_level - c) / c
            str_res[t] = zones[resistance_idx][1]

    return pd.DataFrame(
        {
            "sr_dist_support": dist_sup,
            "sr_dist_resistance": dist_res,
            "sr_support_strength": str_sup,
            "sr_resistance_strength": str_res,
        },
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Velas desde el pivote
# ---------------------------------------------------------------------------

def bars_since_pivot(pivots: pd.Series) -> pd.Series:
    """Número de velas desde el último pivote confirmado.

    Parámetros
    ----------
    pivots : pd.Series
        Serie de pivotes de ``pivot_points`` (NaN salvo en las velas de confirmación).

    Devuelve
    -------
    pd.Series
        Cantidad entera de velas desde el último pivote no NaN. NaN hasta que
        se observa el primer pivote.
    """
    n = len(pivots)
    indices = np.arange(n)
    pivot_indices = np.where(~np.isnan(pivots.to_numpy(dtype=float)), indices, -1)
    last_pivot_indices = np.maximum.accumulate(pivot_indices)
    ages = np.where(last_pivot_indices >= 0, indices - last_pivot_indices, np.nan)
    return pd.Series(ages, index=pivots.index, dtype=float)
