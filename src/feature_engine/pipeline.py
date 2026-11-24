"""
QuantLab ML — Flujo principal del motor de características
============================================
Orquesta el cálculo de todas las familias de características, el contexto del activo
y las etiquetas objetivo para todo el universo de criptomonedas.

Reglas:
- Todas las características deben ser independientes del precio (proporciones, rendimientos, puntajes z y distancias relativas).
- Sin anticipación en las características: cada valor en t usa solo información hasta t.
- Los objetivos (etiquetas) se calculan explícitamente con horizontes futuros y se mantienen separados de las columnas de características.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from . import (
    price_features as pf,
    trend_features as tf,
    momentum_features as mf,
    volatility_features as vf,
    volume_features as vlf,
    structure_features as sf,
)
from .asset_context import compute_asset_context
from .labels import HORIZONS, compute_labels
from .technique_registry import load_technique_registry, technique_defaults

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Flujo principal para calcular y consolidar características cripto."""

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        features_output_dir: Optional[Path] = None,
    ):
        self.processed_dir = processed_dir or (Path(__file__).resolve().parents[2] / "data" / "processed" / "crypto")
        self.output_dir = features_output_dir or (Path(__file__).resolve().parents[2] / "data" / "features" / "crypto")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.technique_registry = load_technique_registry()

    def _params(self, technique_id: str) -> dict:
        return technique_defaults(self.technique_registry, technique_id)

    def extract_features_for_asset(
        self,
        df: pd.DataFrame,
        btc_returns: Optional[pd.Series] = None,
        all_assets_avg_volumes: Optional[pd.DataFrame] = None,
        asset_symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Calcula todas las familias de características, el contexto y las etiquetas para un dataframe de un activo.

        Parámetros
        ----------
        df : pd.DataFrame
            Dataframe OHLCV procesado de un activo, ordenado por fecha.
        btc_returns : pd.Series, optional
            Rendimientos logarítmicos de BTC indexados por fecha para calcular la correlación.
        all_assets_avg_volumes : pd.DataFrame, optional
            Medias retrospectivas de volumen cotizado por fecha y activo.
        asset_symbol : str, optional
            Símbolo de la serie en ``all_assets_avg_volumes``.

        Devuelve
        -------
        pd.DataFrame
            Dataframe enriquecido con identificadores originales, todas las características, contexto y etiquetas.
        """
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], utc=True)
        data = data.sort_values("date").reset_index(drop=True)

        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        vol = data["volume"]
        quote_vol = data["quote_volume"]
        taker_base = data["taker_buy_base_volume"] if "taker_buy_base_volume" in data.columns else vol * 0.5

        # ---------------------------------------------------------------------
        # 1. Familia de precios
        # ---------------------------------------------------------------------
        data["log_return"] = pf.log_return(close)
        data["simple_return"] = pf.simple_return(close)
        data["intraday_range"] = pf.intraday_range(high, low, close)
        data["body_ratio"] = pf.body_ratio(open_, close, high, low)
        data["gap"] = pf.gap(open_, close)
        data["high_low_pos"] = pf.high_low_position(close, high, low)

        # ---------------------------------------------------------------------
        # 2. Familia de tendencias
        # ---------------------------------------------------------------------
        average_params = self._params("moving_averages")
        sma_periods = average_params["sma_periods"]
        ema_periods = average_params["ema_periods"]
        smas = tf.sma(close, periods=sma_periods)
        for p in sma_periods:
            col_sma = f"sma_{p}"
            data[f"dist_to_sma_{p}"] = tf.dist_to_sma(close, smas[col_sma])
        slope_period = average_params["slope_period"]
        data[f"sma_slope_{sma_periods[0]}"] = tf.sma_slope(
            smas[f"sma_{sma_periods[0]}"], period=slope_period
        )

        emas = tf.ema(close, periods=ema_periods)
        for p in ema_periods:
            col_ema = f"ema_{p}"
            data[f"dist_to_ema_{p}"] = tf.dist_to_ema(close, emas[col_ema])

        macd_params = self._params("macd")
        macd_df = tf.macd(close, **macd_params)
        data["macd_line"] = macd_df["macd_line"]
        data["macd_signal"] = macd_df["macd_signal"]
        data["macd_histogram"] = macd_df["macd_histogram"]

        dmi_params = self._params("adx_dmi")
        dmi = tf.directional_movement(high, low, close, period=dmi_params["period"])
        adx_suffix = dmi_params["period"]
        data[f"adx_{adx_suffix}"] = dmi["adx"]
        data[f"plus_di_{adx_suffix}"] = dmi["plus_di"]
        data[f"minus_di_{adx_suffix}"] = dmi["minus_di"]
        data[f"adx_slope_{adx_suffix}"] = dmi["adx"].diff(dmi_params["slope_period"])
        data[f"adx_percentile_{dmi_params['percentile_window']}"] = vf.rolling_percentile_rank(
            dmi["adx"], lookback=dmi_params["percentile_window"]
        )

        structure_params = self._params("market_structure")
        structure_window = structure_params["window"]
        hh_ll = tf.higher_highs_lower_lows(high, low, window=structure_window)
        data[f"higher_highs_ratio_{structure_window}"] = (
            hh_ll["higher_highs_count"] / structure_window
        )
        data[f"lower_lows_ratio_{structure_window}"] = (
            hh_ll["lower_lows_count"] / structure_window
        )

        # ---------------------------------------------------------------------
        # 3. Familia de momentum
        # ---------------------------------------------------------------------
        rsi_params = self._params("rsi")
        rsi_period = rsi_params["period"]
        data[f"rsi_{rsi_period}"] = mf.rsi(close, period=rsi_period)
        data["rsi_percentile_120"] = vf.rolling_percentile_rank(
            data[f"rsi_{rsi_period}"], lookback=rsi_params["percentile_window"]
        )
        roc_periods = self._params("roc")["periods"]
        rocs = mf.roc(close, periods=roc_periods)
        for p in roc_periods:
            data[f"roc_{p}"] = rocs[f"roc_{p}"]
        stochastic_period = self._params("stochastic_oscillator")["period"]
        data[f"stoch_k_{stochastic_period}"] = mf.stochastic_k(
            close, high, low, period=stochastic_period
        )

        # ---------------------------------------------------------------------
        # 4. Familia de volatilidad
        # ---------------------------------------------------------------------
        atr_params = self._params("atr")
        atr_value = vf.atr(high, low, close, period=atr_params["period"])
        data["atr_relative"] = vf.atr_relative(atr_value, close)
        data["atr_percentile_120"] = vf.rolling_percentile_rank(
            data["atr_relative"], lookback=atr_params["percentile_window"]
        )
        volatility_context_params = self._params("realized_volatility")
        atr_mean_period = volatility_context_params["atr_mean_period"]
        data[f"volatility_ratio_{atr_mean_period}"] = vf.volatility_ratio(
            atr_value, period=atr_mean_period
        )

        bb_params = self._params("bollinger")
        bb_df = vf.bollinger_bands(close, period=bb_params["period"], std=bb_params["std"])
        data["bollinger_pctb"] = bb_df["bb_pct_b"]
        data["bollinger_bandwidth"] = bb_df["bb_bandwidth"]
        squeeze_lookback = self._params("bollinger_squeeze")["lookback"]
        data["bollinger_bandwidth_percentile_120"] = vf.rolling_percentile_rank(
            bb_df["bb_bandwidth"], lookback=squeeze_lookback
        )
        data["bollinger_squeeze"] = data["bollinger_bandwidth_percentile_120"]

        keltner_params = self._params("keltner")
        keltner_mid = tf.ema(close, periods=[keltner_params["ema_period"]])[
            f"ema_{keltner_params['ema_period']}"
        ]
        keltner = vf.keltner_channels(
            close,
            atr_value,
            keltner_mid,
            multiplier=keltner_params["multiplier"],
        )
        data["keltner_upper_dist"] = (close - keltner["kc_upper"]) / close
        data["keltner_lower_dist"] = (close - keltner["kc_lower"]) / close

        realized_vol_periods = volatility_context_params["periods"]
        realized_vols = vf.rolling_std_return(data["log_return"], periods=realized_vol_periods)
        for p in realized_vol_periods:
            data[f"vol_realized_{p}d"] = realized_vols[f"rvol_{p}"]

        # ---------------------------------------------------------------------
        # 5. Familia de volumen
        # ---------------------------------------------------------------------
        volume_params = self._params("volume")
        data[f"volume_relative_{volume_params['rvol_period']}"] = vlf.volume_relative(
            vol, period=volume_params["rvol_period"]
        )
        data[f"obv_zscore_{volume_params['obv_zscore_window']}"] = vlf.obv(
            close, vol, zscore_window=volume_params["obv_zscore_window"]
        )
        data["volume_change"] = vlf.volume_change(vol)
        data["taker_buy_ratio"] = vlf.taker_buy_ratio(taker_base, vol)
        data[f"quote_volume_relative_{volume_params['rvol_period']}"] = vlf.quote_volume_relative(
            quote_vol, period=volume_params["rvol_period"]
        )
        data["volume_relative_20_percentile_120"] = vf.rolling_percentile_rank(
            data[f"volume_relative_{volume_params['rvol_period']}"],
            lookback=volume_params["percentile_window"],
        )

        # ---------------------------------------------------------------------
        # 6. Familia de estructura y soporte/resistencia
        # ---------------------------------------------------------------------
        donchian_period = self._params("donchian")["period"]
        donchian = sf.donchian_channels(high, low, close, period=donchian_period)
        data[f"donchian_dist_upper_{donchian_period}"] = donchian["dc_dist_upper"]
        data[f"donchian_dist_lower_{donchian_period}"] = donchian["dc_dist_lower"]
        data[f"donchian_width_{donchian_period}"] = (
            donchian["dc_upper"] - donchian["dc_lower"]
        ) / close
        compression_period = self._params("range_compression")["period"]
        data[f"range_compression_{compression_period}"] = sf.range_compression(
            high, low, period=compression_period
        )

        sr_params = self._params("support_resistance")
        sr_pivots = sf.pivot_points(high, low, window=sr_params["pivot_window"])
        sr = sf.support_resistance_zones(
            sr_pivots["pivot_high"],
            sr_pivots["pivot_low"],
            close,
            tolerance=sr_params["tolerance"],
        )
        data["sr_dist_support"] = sr["sr_dist_support"]
        data["sr_dist_resistance"] = sr["sr_dist_resistance"]
        data["sr_strength_support"] = sr["sr_support_strength"]
        data["sr_strength_resistance"] = sr["sr_resistance_strength"]
        structure_pivots = sf.pivot_points(
            high, low, window=structure_params["pivot_window"]
        )
        data["bars_since_pivot_high"] = sf.bars_since_pivot(
            structure_pivots["pivot_high"]
        )
        data["bars_since_pivot_low"] = sf.bars_since_pivot(
            structure_pivots["pivot_low"]
        )

        retest_params = self._params("breakout_retest")
        retest_channel = sf.donchian_channels(
            high,
            low,
            close,
            period=retest_params["lookback"],
        )
        breakout_features = sf.breakout_retest_features(
            high,
            low,
            close,
            atr_value,
            retest_channel["dc_upper"],
            retest_channel["dc_lower"],
            max_retest_wait=retest_params["max_retest_wait"],
            tolerance_atr=retest_params["tolerance_atr"],
        )
        data[f"breakout_up_{retest_params['lookback']}"] = breakout_features["breakout_up"]
        data[f"breakout_down_{retest_params['lookback']}"] = breakout_features["breakout_down"]
        for column in (
            "retest_up_ok",
            "retest_down_ok",
            "retest_failed",
            "retest_expired",
            "retest_age_bars",
            "retest_pending",
        ):
            data[column] = breakout_features[column]

        divergence_specs = (
            ("rsi_divergence", "rsi", data[f"rsi_{rsi_period}"]),
            ("macd_divergence", "macd", data["macd_histogram"]),
            ("obv_divergence", "obv", data[f"obv_zscore_{volume_params['obv_zscore_window']}"]),
        )
        for technique_id, prefix, oscillator in divergence_specs:
            divergence_params = self._params(technique_id)
            divergence = sf.divergence_features(
                high,
                low,
                oscillator,
                prefix=prefix,
                window=divergence_params["pivot_window"],
                min_pivot_separation=divergence_params["min_pivot_separation"],
                min_price_change_pct=divergence_params["min_price_change_pct"],
                min_oscillator_difference=divergence_params["min_oscillator_difference"],
                max_age_bars=divergence_params["max_age_bars"],
            )
            data = pd.concat([data, divergence], axis=1)

        zscore_period = self._params("mean_reversion_zscore")["period"]
        rolling_mean = close.rolling(zscore_period, min_periods=zscore_period).mean()
        rolling_std = close.rolling(zscore_period, min_periods=zscore_period).std(ddof=0)
        data[f"price_zscore_{zscore_period}"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)

        # ---------------------------------------------------------------------
        # 7. Características de contexto del activo
        # ---------------------------------------------------------------------
        if btc_returns is not None and all_assets_avg_volumes is not None:
            symbol = asset_symbol
            if symbol is None and "symbol" in data.columns and not data.empty:
                symbol = str(data["symbol"].iloc[0])
            if symbol is None:
                raise ValueError("asset_symbol es obligatorio para calcular el contexto cross-sectional")
            relative_strength_period = self._params("relative_strength_btc")["period"]
            data = compute_asset_context(
                data,
                btc_returns=btc_returns,
                all_assets_avg_volumes=all_assets_avg_volumes,
                asset_symbol=symbol,
                log_return_col="log_return",
                quote_volume_col="quote_volume",
                date_col="date",
                relative_strength_period=relative_strength_period,
            )
        else:
            data["ctx_volatility_90d"] = data["log_return"].rolling(90).std() * np.sqrt(365)
            data["ctx_corr_btc_60d"] = np.nan
            data["ctx_volume_rank"] = np.nan
            data["relative_strength_btc_20d"] = np.nan
            first_d = pd.to_datetime(data["date"]).min()
            data["ctx_age_days"] = (pd.to_datetime(data["date"]) - first_d).dt.days

        # ---------------------------------------------------------------------
        # 8. Etiquetas objetivo (la anticipación se limita estrictamente a las columnas de etiquetas)
        # ---------------------------------------------------------------------
        data = compute_labels(
            data,
            close_col="close",
            log_return_col="log_return",
            horizons=HORIZONS,
            k=1.0,
            cost_threshold=0.003,
        )

        return data

    def run_pipeline(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Ejecuta la ingeniería de características para todos los activos cripto procesados.

        Devuelve
        -------
        dict[str, pd.DataFrame]
            Mapeo de símbolo -> dataframe de características.
        pd.DataFrame
            Dataframe consolidado de todos los activos.
        """
        processed_files = list(self.processed_dir.glob("*_processed.csv"))
        if not processed_files:
            raise FileNotFoundError(f"No processed files found in {self.processed_dir}")

        print("=" * 70)
        print("QUANTLAB ML — FLUJO DEL MOTOR DE CARACTERÍSTICAS")
        print(f"Total de archivos procesados encontrados: {len(processed_files)}")
        print(f"Directorio de salida: {self.output_dir}")
        print("=" * 70)

        # 1. Precargar rendimientos de BTC para la correlación
        btc_file = self.processed_dir / "BTCUSDT_processed.csv"
        if not btc_file.exists():
            raise FileNotFoundError(f"BTC processed file not found: {btc_file}")
        btc_df = pd.read_csv(btc_file)
        btc_df["date"] = pd.to_datetime(btc_df["date"], utc=True).dt.strftime("%Y-%m-%d")
        btc_close = btc_df.sort_values("date").set_index("date")["close"]
        btc_returns = np.log(btc_close / btc_close.shift(1)).dropna()

        # 2. Calcular volúmenes cotizados promedio del universo para el contexto del rango de volumen
        symbol_liquidity = {}
        volume_period = self._params("volume")["rvol_period"]
        for f in processed_files:
            sym = f.stem.replace("_processed", "")
            df_temp = pd.read_csv(f, usecols=["date", "quote_volume"])
            df_temp["date"] = pd.to_datetime(df_temp["date"], utc=True).dt.strftime("%Y-%m-%d")
            df_temp = df_temp.sort_values("date").drop_duplicates("date")
            symbol_liquidity[sym] = pd.Series(
                pd.to_numeric(df_temp["quote_volume"], errors="coerce").rolling(
                    volume_period, min_periods=volume_period
                ).mean().to_numpy(),
                index=df_temp["date"],
                name=sym,
            )
        all_avg_vols = pd.concat(symbol_liquidity.values(), axis=1).sort_index()

        # 3. Procesar cada activo
        results = {}
        all_dfs = []
        failures = []

        for f in processed_files:
            sym = f.stem.replace("_processed", "")
            try:
                raw_df = pd.read_csv(f)
                feat_df = self.extract_features_for_asset(
                    raw_df,
                    btc_returns=btc_returns,
                    all_assets_avg_volumes=all_avg_vols,
                    asset_symbol=sym,
                )
                feat_df["symbol"] = sym

                # Guardar archivo de características por activo
                out_file = self.output_dir / f"{sym}_features.csv"
                feat_df.to_csv(out_file, index=False)

                results[sym] = feat_df
                all_dfs.append(feat_df)
                print(f" [✓] {sym}: {feat_df.shape[1]} columnas, {len(feat_df)} filas guardadas.")
            except Exception as e:
                print(f" [X] Error procesando {sym}: {e}")
                failures.append(f"{sym}: {e}")

        # 4. Consolidar el conjunto de datos completo
        if failures:
            raise RuntimeError(
                f"Falló la ingeniería de características para {len(failures)} activos: "
                + "; ".join(failures)
            )
        if not all_dfs:
            raise RuntimeError("No se generaron features para ningún activo")
        consolidated_df = pd.concat(all_dfs, ignore_index=True)
        consolidated_df = consolidated_df.sort_values(["date", "symbol"]).reset_index(drop=True)
        consolidated_path = self.output_dir / "features_consolidated.csv"
        consolidated_df.to_csv(consolidated_path, index=False)

        print("\n" + "=" * 70)
        print("CONSOLIDACIÓN DE CARACTERÍSTICAS COMPLETADA")
        print(f"Dimensiones consolidadas: {consolidated_df.shape[0]:,} filas x {consolidated_df.shape[1]} columnas")
        print(f"Guardado en: {consolidated_path}")
        print("=" * 70)

        return results, consolidated_df


if __name__ == "__main__":
    pipeline = FeaturePipeline()
    pipeline.run_pipeline()
