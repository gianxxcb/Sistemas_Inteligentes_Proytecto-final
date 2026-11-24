"""Pipeline de extracción y consolidación de características multiescala V3.

Orquesta el cálculo conjunto de:
1. Características técnicas fundamentales (tendencia, momentum, volatilidad, volumen, estructura).
2. Características de microestructura y derivados V3 (tiempo cíclico, funding, taker buy, rezagos BTC/ETH).
3. Particionamiento por escala temporal: 1H (Táctico), 4H (Swing corto), 1D (Posicional).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .pipeline import FeaturePipeline
from .derivatives_features import (
    compute_cyclic_time_features,
    compute_funding_features,
    compute_taker_buy_pressure,
    compute_lead_market_returns,
    compute_micro_spread_proxy,
)


class MultiscaleFeaturePipeline:
    """Calculador y consolidador de matrices de features para QuantLab ML V3."""

    def __init__(
        self,
        features_output_dir: Optional[Path] = None,
    ):
        self.output_dir = features_output_dir or (Path(__file__).resolve().parents[2] / "data" / "features" / "crypto")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_pipeline = FeaturePipeline()

    def process_asset_features(
        self,
        symbol: str,
        df_raw: pd.DataFrame,
        timeframe: str = "1h",
        funding_df: Optional[pd.DataFrame] = None,
        btc_df: Optional[pd.DataFrame] = None,
        eth_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Calcula el vector de características completo para un par y escala temporal."""
        df = df_raw.copy()
        market_types = (
            set(df["market_type"].dropna().astype(str).str.lower().unique())
            if "market_type" in df.columns
            else set()
        )
        intervals = (
            set(df["interval"].dropna().astype(str).str.lower().unique())
            if "interval" in df.columns
            else set()
        )
        if market_types != {"binance_usdt_m_futures"}:
            raise ValueError(f"{symbol} no tiene procedencia verificada de Binance USDT-M Futures.")
        if intervals != {timeframe.lower()}:
            raise ValueError(
                f"{symbol} contiene temporalidades {sorted(intervals)}; se esperaba {timeframe.lower()}."
            )
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.sort_values("date").reset_index(drop=True)
        else:
            raise KeyError("El DataFrame debe contener la columna 'date'")

        # 1. Extraer features técnicas base usando el pipeline modular
        base_features = self.base_pipeline.extract_features_for_asset(
            df=df,
            asset_symbol=symbol,
        )

        # 2. Extraer features de microestructura y tiempo cíclico V3
        time_feats = compute_cyclic_time_features(df, time_col="date")
        fund_feats = compute_funding_features(df, funding_df=funding_df, time_col="date")
        taker_feats = compute_taker_buy_pressure(df)
        spread_feats = compute_micro_spread_proxy(df)

        lead_btc = btc_df if btc_df is not None else (df if symbol == "BTCUSDT" else None)
        lead_eth = eth_df if eth_df is not None else (df if symbol == "ETHUSDT" else None)
        lead_feats = compute_lead_market_returns(df, btc_df=lead_btc, eth_df=lead_eth, time_col="date")

        # 3. Concatenar todas las familias de variables
        all_dfs = [base_features, time_feats, fund_feats, taker_feats, spread_feats, lead_feats]

        consolidated = pd.concat(all_dfs, axis=1)

        # Eliminar columnas duplicadas si las hubiera
        consolidated = consolidated.loc[:, ~consolidated.columns.duplicated()]
        consolidated["symbol"] = symbol
        consolidated["timeframe"] = timeframe.lower()
        consolidated["market_type"] = "binance_usdt_m_futures"

        return consolidated

    def build_and_save_dataset(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        timeframe: str = "1h",
        funding_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Procesa una colección de símbolos y guarda el archivo consolidado."""
        btc_df = symbols_data.get("BTCUSDT")
        eth_df = symbols_data.get("ETHUSDT")

        futures_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "futures" / timeframe.lower()
        if btc_df is None:
            btc_file = futures_root / "BTCUSDT.parquet"
            if btc_file.is_file():
                btc_df = pd.read_parquet(btc_file)
        if eth_df is None:
            eth_file = futures_root / "ETHUSDT.parquet"
            if eth_file.is_file():
                eth_df = pd.read_parquet(eth_file)

        frames = []
        for sym, df_raw in symbols_data.items():
            f_df = funding_data.get(sym) if funding_data else None
            processed = self.process_asset_features(
                symbol=sym,
                df_raw=df_raw,
                timeframe=timeframe,
                funding_df=f_df,
                btc_df=btc_df,
                eth_df=eth_df,
            )
            frames.append(processed)

        if not frames:
            return pd.DataFrame()

        final_df = pd.concat(frames, ignore_index=True)
        dest_file = self.output_dir / f"features_{timeframe.lower()}.parquet"
        final_df.to_parquet(dest_file, index=False)

        return final_df
