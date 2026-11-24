# Módulo 01 — Crypto Data Engine (V3)

**Función:** recolectar, validar y normalizar los datos de mercado. Es la base de todo;
errores aquí contaminan todo el sistema.

> **Cambios V3:** se añaden velas de 1H y 15m para los estilos táctico y swing corto,
> datos de funding histórico de perpetuos, datos del libro de órdenes (bid-ask spread)
> para medir slippage, y la hora UTC como dimensión explícita de los datos.

---

## 1. Universo de activos

| Grupo | Activos |
|---|---|
| Principales | BTC, ETH |
| Alta capitalización | BNB, SOL, XRP |
| Alternativos | ADA, AVAX, LINK, LTC |
| Adicionales | DOT, DOGE, BCH, TRX, y otros |

- **Meta inicial:** 20–30 criptomonedas líquidas con historial suficiente.
- **Incluir activos delistados o que perdieron liquidez** para evitar sesgo de
  supervivencia.
- **Monedas reservadas:** ≈5 de 25, estratificadas, fijadas antes de entrenar (módulo 13).

---

## 2. Datos requeridos por estilo

### 2.1 Datos OHLCV — temporalidades por estilo

| Estilo | Temporalidad decisión | Temporalidad gestión | Ambigüedad intravela |
|---|---|---|---|
| Táctico | 1H | 15m | Se resuelve con 15m |
| Swing corto | 4H | 1H | Se resuelve con 1H |
| Swing/posicional | 1D | 4H (1H cerca de niveles) | Se resuelve con 4H |

**Temporalidades a descargar:** 15m, 1H, 4H, 1D (y 1D para features de largo plazo).

Los datos V3 provienen exclusivamente de **Binance USDT-M Futures**. Cada archivo se
mantiene en su temporalidad real y en UTC; no se permite completar una temporalidad
faltante copiando velas spot o renombrando velas 1D como intradía. La descarga pagina
desde la primera vela que el contrato tenga disponible y conserva solo velas cerradas.
Los límites de la API son límites por página, no límites del historial total. Cada
parquet lleva metadatos de mercado y temporalidad; los archivos V3 antiguos sin
procedencia verificable no se mezclan con la nueva descarga.

Binance ofrece: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1mo.
Los archivos incluyen además: **número de operaciones** y **volumen comprador** (taker
buy volume), que son features adicionales para los expertos.

Todos los timestamps en **UTC**. Cripto opera 24/7; no hay sesión que marque el día.

### 2.2 Datos de funding histórico (nuevo en V3)

Para cada contrato perpetuo USDT-M de Binance:

- **Tasa de funding** por cada liquidación (cada 8 h): 00:00, 08:00 y 16:00 UTC.
- Historial disponible en Binance a través de `/fapi/v1/fundingRate`.
- Se almacena junto con los OHLCV en la misma tabla, alineado por timestamp.

**Uso:**
- Feature `funding_rate_last`: tasa de la última liquidación.
- Feature `cerca_funding`: indicador binario (±30 min de cada liquidación).
- Costo de funding acumulado por episodio (módulo 15 §1.3).

### 2.3 Datos del libro de órdenes (nuevo en V3)

Para medir el slippage real (módulo 15 §1.2):

- **Bid-ask spread** en el momento de entrada y salida de operaciones históricas.
- Profundidad del libro (top 5 niveles) en el instante de cada señal.
- Fuente: datos históricos de libro de Binance o proxy desde OHLCV
  (spread estimado = (high - low) × k, con k calibrado).

**Uso:**
- Feature `bid_ask_spread_bps`: spread en puntos básicos.
- Estimación de slippage por activo y por hora UTC.
- Calibración del parámetro de slippage del módulo 15.

---

## 3. Patrón horario de actividad (hora UTC)

La investigación muestra picos de **actividad, volatilidad e iliquidez entre 16:00 y
17:00 UTC** en Binance y otros exchanges, relacionados con horarios de funding y
trading algorítmico.

**Consecuencias para los datos:**
- Añadir `hour_utc` como columna en todos los datasets procesados.
- Calcular features cíclicas: `hour_sin = sin(2π × hour/24)`,
  `hour_cos = cos(2π × hour/24)`.
- Los modelos de los expertos tácticos recibirán estas features (H4).

---

## 4. Validaciones obligatorias

- Sin velas duplicadas ni huecos (o huecos marcados explícitamente).
- Detección de precios/volúmenes anómalos (volumen 0, saltos imposibles, wicks erróneos).
- Consistencia: `low ≤ open, close ≤ high`.
- Historial mínimo por activo (≥2 años en 1D; ≥1 año en 1H; ≥6 meses en 15m).
- Versionado del dataset con fecha de descarga y fuente.
- **Consistencia entre temporalidades:** el OHLCV de 1H debe ser consistente con el
  agregado de los 4 × 15m correspondientes (verificación de sumas de volumen y
  coherencia de high/low).
- Cada temporalidad debe estar alineada a su rejilla UTC (15m, 1h, 4h o medianoche
  diaria), y las agregaciones se comparan solo en intervalos completos.
- El historial empieza en la fecha de listado del contrato perpetuo; ocho años por
  moneda no son posibles si el contrato no existía entonces. No se extenderá Futures
  con Spot para aparentar mayor cobertura.

  La recarga completa de los 25 perpetuos V3 se ejecuta con
  `python scripts/download_v3_market_data.py --online --all-symbols`. El constructor de
  features procesa un activo a la vez para evitar cargar el universo intradía completo
  en memoria: `python scripts/build_v3_features.py --timeframe all --all-symbols`.
  La validación de cada serie es independiente de la comparación entre escalas; las
  discrepancias agregadas se reportan como advertencias y deben revisarse antes de usar
  los resultados como validación final de estrategia.

---

## 5. Almacenamiento

```
data/raw/            # tal como se descargó (por temporalidad y activo)
  15m/               # velas de 15 minutos
  1h/                # velas de 1 hora
  4h/                # velas de 4 horas
  1d/                # velas diarias
  funding/           # tasas de funding históricas
  orderbook/         # snapshots del libro de órdenes
data/processed/      # limpio, calendario UTC uniforme, consistencias verificadas
data/features/       # variables derivadas (módulo 02)
data/scenarios/      # embeddings, clusters, registros de escenarios
```

---

## 6. Admisión de monedas (V2, sin cambios)

Como el usuario elige sus monedas, el Data Engine genera para cada moneda nueva un
**informe de admisión** con historial, calidad y liquidez (módulo 14). Los criterios
de liquidez se evalúan ahora también en 1H (no solo en 1D).

---

## 7. Entregable

`Crypto Dataset v2.0` (multitemporal: 15m/1H/4H/1D) + datos de funding + informe de
calidad de datos por activo y temporalidad.

---

## 8. Riesgos

- Datos de exchanges pequeños poco fiables.
- Tokens con historial corto en temporalidades inferiores (el historial de 15m en
  Binance puede ser más corto que el de 1D para activos listados antes de 2020).
- Cambios de ticker/migraciones de token.
- El libro de órdenes histórico puede no estar disponible para fechas anteriores a 2021
  en algunos activos; en ese caso usar el proxy de spread estimado.
- **Volumen taker buy:** disponible en Binance desde 2019 aproximadamente; verificar
  disponibilidad por activo.

## 9. Estado del dataset V3 (2026-11-20)

Se descargaron 25 contratos perpetuos Binance USDT-M Futures en cuatro temporalidades,
con 7.406.710 velas en total. Las features reconstruidas contienen:

| Temporalidad | Filas |
|---|---:|
| 15m | 5.598.783 |
| 1H | 1.399.692 |
| 4H | 349.926 |
| 1D | 58.309 |

El historial comienza en la primera vela disponible por contrato; no equivale a ocho
años completos para cada moneda. La validación individual de las 100 series pasó, pero
la auditoría agregada entre escalas reportó 225 discrepancias en 1.807.883 comparaciones
y 74/75 verificaciones en `WARNING`. Por tanto, la consistencia entre temporalidades no
debe describirse como plenamente aprobada hasta investigar esas diferencias.

En esta descarga se dispone de OHLCV y funding. No se verificó un histórico de libro de
órdenes; el slippage configurado en los episodios es una hipótesis del modelo de costos,
no una medición histórica del spread real. Véase la [hoja de ruta V3](10_roadmap_repositorio.md)
para los bloqueos restantes.
