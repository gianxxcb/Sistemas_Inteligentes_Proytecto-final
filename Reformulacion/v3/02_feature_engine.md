# Módulo 02 — Feature Engine (V3)

**Función:** transformar OHLCV multitemporal, datos de funding y libro de órdenes en
características matemáticas normalizadas, causales e independientes del precio nominal.

> **Cambios V3:** se incorporan features específicas de temporalidad horaria e intra-día
> (`hour_sin`, `hour_cos`, `cerca_funding`, `funding_rate_last`), variables derivadas
> del libro de órdenes (spread en bps), retornos rezagados transversales de BTC/ETH (H9),
> y cálculo multiescala para los 3 estilos de operación (15m, 1H, 4H, 1D).

---

## 1. Familias de características

| Familia | Variables |
|---|---|
| **Precio y Retorno** | Retornos logarítmicos ($r_t$), rangos relativos $(H-L)/C$, gaps de apertura |
| **Tendencia** | SMA, EMA, pendientes relativas, estructura de pivotes (HH, HL, LH, LL) |
| **Momentum** | RSI (Wilder RMA), MACD diferencial normalizado, ROC ($N$ barras) |
| **Volatilidad** | ATR relativo ($ATR_{14}/Close$), percentil histórico de ATR, BandWidth de Bollinger |
| **Volumen y Flujo** | RVOL (volumen / SMA_vol), OBV estandarizado, **taker buy ratio** ($V_{\text{taker\_buy}} / V_{\text{total}}$) |
| **Estructura** | Distancia porcentual a S/R clave, compresión de rango, duración de squeezes |
| **Horario y Liquidación** | `hour_sin`, `hour_cos` (ciclo UTC 24h), `cerca_funding` (±30m de 00/08/16 UTC) |
| **Derivados y Costos** | `funding_rate_last`, spread bid-ask estimado/observado en bps |
| **Mercado Líder (H9)** | Retornos rezagados de BTC y ETH en 1H, 4H y 1D |

---

## 2. Variables técnicas requeridas por los expertos

| Experto | Variables técnicas clave |
|---|---|
| **Trend** | SMA/EMA múltiples, MACD normalizado, **ADX / DMI** (Wilder, fuerza de tendencia) |
| **Reversal** | RSI, Bollinger %B, **divergencias RSI/MACD confirmadas**, distancia a la media |
| **Breakout** | Canales Donchian (máx/mín $N$ períodos), distancia a S/R, volumen relativo, retesteos |
| **Volatility** | ATR relativo, **Bollinger Squeeze** (BandWidth vs Keltner), razón de expansión |
| **Momentum** | ROC, RSI, MACD, **fuerza relativa transversal frente a BTC** |

Todas las especificaciones exactas de fórmulas y rangos se detallan en `05b_catalogo_tecnicas_trading.md`.

---

## 3. Adaptación multitemporal por estilo

El Feature Engine calcula matrices de características alineadas por estilo:

1. **Estilo Táctico (1H / 15m):**
   - Features calculadas con barras de 1H.
   - Indicadores de ciclo horario (`hour_sin`, `hour_cos`) y ventana de liquidación funding.
   - Taker buy volume ratio y micro-spread.
   - Retornos rezagados de BTC en 1H y 4H (H9).

2. **Estilo Swing Corto (4H / 1H):**
   - Features calculadas sobre barras de 4H.
   - Promedios móviles de 20 a 50 períodos de 4H.
   - Correlación móvil 4H con BTC.

3. **Estilo Swing/Posicional (1D / 4H):**
   - Features diarias clásicas (medias 20/50/200 días, Donchian 20d).
   - Régimen estructural y contexto macro.

---

## 4. Normalización estricta sin look-ahead

$$ r_t = \ln(P_t / P_{t-1}) \qquad d_{\text{SMA}} = \frac{P_t - \text{SMA}_k}{\text{SMA}_k} \qquad ATR_{\text{rel}} = \frac{ATR_{14}}{P_t} $$

- Scalers ajustados **únicamente** sobre la partición de entrenamiento de cada pliegue.
- Detección de pivotes con retraso causal obligatorio de $k$ barras (no look-ahead).
- Manejo explícito de NaNs mediante ventanas de calentamiento (warm-up).

---

## 5. Historial mínimo (Warm-up)

- **1D:** Requiere ≥ 320 días de historial previo antes del primer vector válido (por SMA 200).
- **4H:** Requiere ≥ 200 barras de 4H (~33 días).
- **1H:** Requiere ≥ 200 barras de 1H (~8.5 días).
- Los períodos iniciales de warm-up se descartan de entrenamiento y evaluación.

---

## 6. Entregable

Archivos en `data/features/` particionados por estilo (`features_1h.parquet`, `features_4h.parquet`, `features_1d.parquet`) con metadata de trazabilidad.
