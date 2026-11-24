"""Módulo de características de derivados, microestructura y tiempo cíclico para V3.

Implementa:
1. Features cíclicas de hora UTC y día de semana (hour_sin, hour_cos, day_sin, day_cos).
2. Indicador binario de proximidad a liquidación de funding (cerca_funding).
3. Dinámica de funding rate (funding_rate_last, funding_rate_rolling_3).
4. Presión compradora neta (taker_buy_ratio).
5. Retornos rezagados de líderes de mercado (BTC, ETH) para predecir altcoins (H9).
6. Proxy de spread y microestructura.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def compute_cyclic_time_features(df: pd.DataFrame, time_col: str = "date") -> pd.DataFrame:
    """Calcula la codificación armónica continua (seno y coseno) de la hora UTC y día de la semana."""
    out = pd.DataFrame(index=df.index)
    dates = pd.to_datetime(df[time_col], utc=True)

    hours = dates.dt.hour + (dates.dt.minute / 60.0)
    days = dates.dt.dayofweek + (hours / 24.0)

    out["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    out["day_sin"] = np.sin(2.0 * np.pi * days / 7.0)
    out["day_cos"] = np.cos(2.0 * np.pi * days / 7.0)

    return out


def compute_funding_features(
    df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    time_col: str = "date",
) -> pd.DataFrame:
    """Calcula variables asociadas al acarreo de funding y la ventana de liquidación."""
    out = pd.DataFrame(index=df.index)
    dates = pd.to_datetime(df[time_col], utc=True).astype("datetime64[ns, UTC]")

    # 1. Indicador de proximidad a liquidación (00, 08, 16 UTC +- 30 min)
    # En barras horarias: horas 0, 8, 16 o justo antes (23, 7, 15)
    is_settlement_window = dates.dt.hour.isin([0, 8, 16, 23, 7, 15])
    out["cerca_funding"] = is_settlement_window.astype(int)

    # 2. Asignación de tasa de funding
    if funding_df is not None and not funding_df.empty and "funding_rate" in funding_df.columns:
        f_df = funding_df.copy()
        f_df["f_date"] = pd.to_datetime(
            f_df["date"], format="mixed", utc=True
        ).astype("datetime64[ns, UTC]")
        f_df = f_df.sort_values("f_date").reset_index(drop=True)
        left = df[[time_col]].reset_index()
        left[time_col] = pd.to_datetime(
            left[time_col], utc=True
        ).astype("datetime64[ns, UTC]")

        # Merge asof con corte causal (direction='backward')
        merged = pd.merge_asof(
            left,
            f_df[["f_date", "funding_rate"]],
            left_on=time_col,
            right_on="f_date",
            direction="backward",
        ).set_index("index")

        out["funding_rate_last"] = merged["funding_rate"].fillna(0.00010)
    else:
        out["funding_rate_last"] = 0.00010

    # Media móvil de 3 liquidaciones de funding (24h)
    out["funding_rate_rolling_3"] = out["funding_rate_last"].rolling(3, min_periods=1).mean()

    return out


def compute_taker_buy_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la presión compradora de mercado a partir de los datos taker de Binance Futures."""
    out = pd.DataFrame(index=df.index)

    if "taker_buy_base_volume" in df.columns and "volume" in df.columns:
        vol = df["volume"].astype(float)
        taker = df["taker_buy_base_volume"].astype(float)
        ratio = taker / np.where(vol > 0, vol, 1.0)
        out["taker_buy_ratio"] = np.clip(ratio, 0.0, 1.0)
    elif "quote_volume" in df.columns and "taker_buy_quote_volume" in df.columns:
        q_vol = df["quote_volume"].astype(float)
        q_taker = df["taker_buy_quote_volume"].astype(float)
        ratio = q_taker / np.where(q_vol > 0, q_vol, 1.0)
        out["taker_buy_ratio"] = np.clip(ratio, 0.0, 1.0)
    else:
        # Fallback neutro 0.50 si no está en la fuente
        out["taker_buy_ratio"] = 0.50

    return out


def compute_lead_market_returns(
    df: pd.DataFrame,
    btc_df: Optional[pd.DataFrame] = None,
    eth_df: Optional[pd.DataFrame] = None,
    time_col: str = "date",
) -> pd.DataFrame:
    """Calcula retornos rezagados de BTC y ETH para predecir altcoins (Hipótesis H9)."""
    out = pd.DataFrame(index=df.index)
    current_dates = pd.to_datetime(df[time_col], utc=True)

    # 1. Rezagos de BTC
    if btc_df is not None and not btc_df.empty:
        b_df = btc_df.copy()
        b_df["b_date"] = pd.to_datetime(b_df[time_col], utc=True)
        b_df = b_df.sort_values("b_date").reset_index(drop=True)
        b_close = b_df["close"].astype(float)
        b_df["btc_ret_lag1"] = b_close.pct_change(1).shift(1)
        b_df["btc_ret_lag2"] = b_close.pct_change(2).shift(1)

        merged_btc = pd.merge_asof(
            df[[time_col]].reset_index(),
            b_df[["b_date", "btc_ret_lag1", "btc_ret_lag2"]],
            left_on=time_col,
            right_on="b_date",
            direction="backward",
        ).set_index("index")

        out["btc_return_lag1"] = merged_btc["btc_ret_lag1"].fillna(0.0)
        out["btc_return_lag2"] = merged_btc["btc_ret_lag2"].fillna(0.0)
    else:
        out["btc_return_lag1"] = 0.0
        out["btc_return_lag2"] = 0.0

    # 2. Rezagos de ETH
    if eth_df is not None and not eth_df.empty:
        e_df = eth_df.copy()
        e_df["e_date"] = pd.to_datetime(e_df[time_col], utc=True)
        e_df = e_df.sort_values("e_date").reset_index(drop=True)
        e_close = e_df["close"].astype(float)
        e_df["eth_ret_lag1"] = e_close.pct_change(1).shift(1)

        merged_eth = pd.merge_asof(
            df[[time_col]].reset_index(),
            e_df[["e_date", "eth_ret_lag1"]],
            left_on=time_col,
            right_on="e_date",
            direction="backward",
        ).set_index("index")

        out["eth_return_lag1"] = merged_eth["eth_ret_lag1"].fillna(0.0)
    else:
        out["eth_return_lag1"] = 0.0

    return out


def compute_micro_spread_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula un proxy empírico del spread en puntos básicos a partir del micro-rango."""
    out = pd.DataFrame(index=df.index)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # Proxy conservador: 5% del rango relativo convertido a bps
    relative_range = (high - low) / np.where(close > 0, close, 1.0)
    spread_bps = relative_range * 0.05 * 10_000.0
    out["bid_ask_spread_bps"] = np.clip(spread_bps, 1.0, 50.0)

    return out
