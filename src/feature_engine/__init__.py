"""
QuantLab ML — Feature Engine
============================
Transforma datos OHLCV sin procesar en características matemáticas
independientes del precio para la investigación cuantitativa de criptomonedas.

Familias
--------
- price_features    : Rendimientos, rangos, brechas y proporciones del cuerpo de vela
- trend_features    : SMA, EMA, MACD, ADX, pendiente, máximos/mínimos crecientes y decrecientes
- momentum_features : RSI, ROC, estocástico %K
- volatility_features: ATR, bandas de Bollinger, canales de Keltner y desviación estándar móvil σ
- volume_features   : Volumen relativo, OBV y proporción de compras taker
- structure_features: Canales Donchian, puntos pivote y zonas de soporte/resistencia
- asset_context     : Régimen de volatilidad, correlación con BTC, rango de liquidez relativa y antigüedad
- labels            : Objetivos de rendimiento futuro en 7, 14, 30 y 90 días
- pipeline          : Orquestador principal del flujo
- baseline          : Modelos de referencia (incondicional, regresión logística, LightGBM)
"""

from . import (
    price_features,
    trend_features,
    momentum_features,
    volatility_features,
    volume_features,
    structure_features,
    asset_context,
    labels,
    pipeline,
    baseline,
)
from .pipeline import FeaturePipeline

__all__ = [
    "price_features",
    "trend_features",
    "momentum_features",
    "volatility_features",
    "volume_features",
    "structure_features",
    "asset_context",
    "labels",
    "pipeline",
    "baseline",
    "FeaturePipeline",
]
