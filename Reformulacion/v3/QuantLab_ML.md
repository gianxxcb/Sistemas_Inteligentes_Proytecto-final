# QuantLab ML — V3: Sistema de Trading con ML para Criptomonedas

## Qué hace este sistema

Un sistema de aprendizaje automático que aprende a aplicar **técnicas de trading**
(análisis técnico, gestión de trades) como lo haría un trader experto, pero de forma
sistemática, reproducible y validada estadísticamente.

Opera en **Binance USDT-M Futures** con apalancamiento moderado, en tres estilos de
operación que cubren horizontes de horas a semanas.

---

## Tres estilos de operación

| Estilo | Temporalidad decisión | Horizonte | Ritmo típico |
|---|---|---|---|
| **Táctico** | 1H (gráfico de 1 hora) | 6–24 horas | ~1 operación por día |
| **Swing corto** | 4H | 2–7 días | ~3 operaciones por semana |
| **Swing/posicional** | 1D + disparador 4H | 10–30 días | 2–3 operaciones por mes |

Los tres estilos coexisten: pueden generar señales simultáneas sobre el mismo activo
con distintos horizontes, sin contradicción.

---

## Los cinco módulos núcleo

```
Datos (01) ──► Features (02) ──► Representación (03)
                                        │
                                  Escenarios (04)──► HMM Régimen
                                        │
                               Expertos (05): 5 técnicas × 3 estilos
                                        │
                               Orquestador (06): 3 niveles
                                        │
                               Forecast Engine (07)
                                        │
                               Riesgo / Backtesting (09)
```

---

## Cómo aprende un experto

Cada uno de los cinco expertos (Trend, Reversal, Breakout, Volatility, Momentum)
aprende de **episodios**: unidades de aprendizaje que imitan cómo opera un trader.

```
1. FORMACIÓN        El experto ve la estructura del mercado (ventana de precio)
       │
       ▼
2. PLAN             Emite: entrada, stop (invalidación), objetivo, acciones condicionales
       │             Solo entra si E[R neto] > 0 (filtro de ejecución)
       ▼
3. REPLAY           El simulador revela eventos uno a uno (E1–E4)
   por eventos      El experto actúa solo cuando ocurre algo relevante
       │
       ▼
4. RESULTADO        R neto = ganancia/pérdida en múltiplos de riesgo, tras costos
                    (comisión + slippage + funding acumulado)
```

---

## Modelo de costos (Binance Futures)

| Componente | Valor |
|---|---|
| Comisión maker (limit) | 0.020 % |
| Comisión taker (market) | 0.040–0.050 % |
| Slippage estimado | 0.30 % (supuesto; medir con datos propios) |
| Funding base | 0.01 % cada 8 h (~11 % anual) |
| **Costo total típico** (táctico, ida y vuelta) | **~0.36–0.39 %** |

**Stop mínimo:** ~1–2 % del precio (parámetro del sistema). Con stops menores, los
costos consumen la ventaja.

---

## Orquestador en tres niveles

```
Nivel 0: Promedio simple / Hedge / Fixed Share  ◄── siempre activo
    │  Si supera al incondicional con CPCV:
    ▼
Nivel 1: Contracción jerárquica + Régimen HMM   ◄── activo si mejora al nivel 0
    │  Si supera al nivel 0 con Sharpe deflactado:
    ▼
Nivel 2: Gating neuronal                        ◄── solo si H6 se rechaza
```

---

## Validación rigurosa

| Técnica | Qué protege |
|---|---|
| **Muestreo CUSUM** | Episodios más independientes, menos redundancia |
| **Pesos por unicidad** | Corrige la dependencia entre episodios solapados |
| **Purga y embargo** | Evita que el futuro se filtre al entrenamiento |
| **CPCV** | Distribución del rendimiento, no un único número |
| **Sharpe deflactado** | Corrige el sobreajuste por comparaciones múltiples |
| **Experimento nulo** | El modelo no debe encontrar "ventaja" en datos barajados |
| **Test final 2025** | Usado una sola vez, al final de todo el desarrollo |

---

## 9 hipótesis a probar (módulo 17)

H1: magnitud se predice mejor que dirección · H2: el filtro de costos mejora la
expectativa neta · H3: XGBoost iguala a redes neuronales · H4: hora UTC y funding
mejoran la ejecución táctica · H5: HMM mejora la selección de expertos · H6:
promedio simple iguala al gating neuronal · H7: CUSUM mejora la calidad de etiquetas ·
H8: confirmación multitemporal mejora la expectativa · H9: retornos de BTC/ETH predicen
otras monedas.

---

## Mapa de módulos V3

| Módulo | Contenido |
|---|---|
| `01_data_engine` | OHLCV 15m/1H/4H/1D, funding histórico, libro de órdenes |
| `02_feature_engine` | Features técnicas por experto y por temporalidad |
| `03_representation_engine` | Embedding de ventanas de precio |
| `04_scenario_engine` | Clustering de escenarios + HMM de régimen (3 estados) |
| `05_specialist_models` | 5 expertos × 3 estilos, CUSUM, triple barrera playbook, pesos unicidad |
| `06_meta_model` | Orquestador 3 niveles (simple→jerárquico→neuronal) |
| `07_forecast_engine` | Distribuciones por estilo/horizonte, filtro de ejecución integrado |
| `08_evaluacion_validacion` | CPCV, Sharpe deflactado, protocolo H1–H9 |
| `09_riesgo_backtesting` | Margen, liquidación, funding, stop mínimo, presupuesto |
| `10_roadmap_repositorio` | Hoja de ruta de implementación |
| `11_redes_neuronales` | Arquitecturas temporales (V3 de expertos) |
| `12_episodios_y_simulador` | Episodios por eventos, funding acumulado, resolución intravela |
| `13_division_de_datos` | CPCV, purga por estilo, monedas reservadas |
| `14_diario_de_casos` | Indexación y recuperación de episodios históricos |
| `15_modelo_de_costos` | **[NUEVO]** Comisiones, funding, stop mínimo, filtro de ejecución |
| `16_paso_del_tiempo` | **[NUEVO]** Eventos E1–E4, CUSUM, temporalidades múltiples |
| `17_hipotesis_y_experimentos` | **[NUEVO]** H1–H9 con protocolo explícito |

---

## Decisiones tentativas que esperan datos

- Stop mínimo (~1–2 % del precio).
- Slippage real (el 0.30 % es un supuesto conservador; medir con libro de órdenes).
- Número de estados del HMM (3 candidatos; probar también 4).
- Umbral CUSUM $h$.
- Presupuesto de operaciones semanal (top-3 a top-5).
- Horizonte máximo del estilo táctico (6 h vs 24 h).

---

## Aviso permanente

> Este sistema es una herramienta de investigación. Ningún resultado dentro de muestra
> ni en validación garantiza rendimiento futuro. El test final 2025 se usa una sola vez.
> Las monedas reservadas no se consultan hasta el test final. Los costos y comisiones
> de Binance pueden cambiar; verificar en la página oficial antes de cada uso operativo.
> El slippage de 0.30 % es una suposición que debe medirse con datos reales.
