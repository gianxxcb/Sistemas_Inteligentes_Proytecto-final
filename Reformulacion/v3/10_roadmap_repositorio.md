# Módulo 10 — Hoja de Ruta y Estructura del Repositorio (V3)

## Objetivos de diseño V3

La versión 3 reorienta el desarrollo hacia la operación realista en **Binance USDT-M Futures**
con apalancamiento, tres estilos temporales (Táctico 1H, Swing corto 4H, Swing/posicional 1D),
filtro estricto de costos, muestreo CUSUM y orquestación escalonada (Nivel 0 → 1 → 2).

La tabla siguiente describe los **objetivos** de diseño, no el estado de cierre. El estado
comprobado al 20 de noviembre de 2026 aparece en la sección posterior.

| Etapa V3 | Entregable | Módulos clave | Criterio para avanzar |
|---|---|---|---|
| **E1** | Ingesta multitemporal (15m, 1H, 4H, 1D), funding y book proxy | 01, 15 | Consistencia temporal validada; datasets de funding y spread consolidados |
| **E2** | Features multiescala y variables de derivados / UTC | 02, 15 | Features causales sin look-ahead; variables de funding y hora integradas |
| **E3** | Motor de simulación por eventos y modelo de costos en futuros | 12, 15, 16 | Replay por eventos E1–E4; cálculo de R neto con comisión, slippage y funding |
| **E4** | Muestreo CUSUM y extracción de episodios independientes | 05, 16 | Unicidad media > 0.60; etiquetas netas por triple barrera de playbook |
| **E5** | Scenario Engine dual: Clustering local + HMM de régimen de mercado | 04 | 3 estados HMM reproducibles y alineados; clusters estables en train |
| **E6** | Baseline HGB y comparación TCN/GRU/nulo, ponderados por unicidad | 05, 17 | Comparación por estilo, con purga temporal y evaluación en activos no vistos |
| **E7** | Orquestador Nivel 0 (promedio/Hedge) y Nivel 1 (shrinkage + régimen) | 06, 17 | Nivel 0 y 1 evaluados en CPCV; H5 y H6 comprobadas |
| **E8** | Evaluación rigurosa CPCV + Sharpe deflactado e hipótesis H1–H9 | 08, 17 | Protocolo de hipótesis predefinido y ejecutado en validación sin consultar el holdout final |
| **E9** | Cartera, control de liquidación/margen y visualización interactiva | 09 | Límites de concentración y correlación BTC; cero liquidaciones de margen |
| **E10** | Test confirmatorio final y validación en activos reservados | 08, 13, 14 | Congelar datos, activos y criterios antes de una única ejecución confirmatoria |

## Cierre técnico de E1, E3 y E5 · 22 de noviembre de 2026

| Etapa | Resultado verificado | Alcance |
|---|---|---|
| E1 | 100 series de 25 contratos en cuatro escalas, 225 discrepancias identificadas en cuarentena y 177.058 tasas de financiación consolidadas. | El proxy de spread de 15 minutos se calibró con bid/ask de ADA, LINK y LTC del 1 de junio de 2023. El 8 de junio, en 288 intervalos distintos, el error absoluto medio fue 0,90 pb. Su aplicación a otros contratos es una estimación. Medio spread aproxima el coste de cruce sin impacto de mercado. |
| E3 | Replay íntegro de 533.001 episodios. Se comprobaron las aperturas de ejecución frente a las velas de gestión y la reconciliación del retorno bruto, costes y retorno neto para todas las entradas. | Simulador y Rendimiento leen el replay causal. Los costes de ejecución mantienen las hipótesis configuradas. Hubo 3.424 cargos o abonos simulados de financiación con tasa imputada, contados por episodio. La cartera de 2025 conserva sus artefactos congelados. |
| E5 | Agrupamiento de embeddings con cinco reajustes de entrenamiento por bloques. ARI medio de validación de 0,902, 0,966 y 0,948 para táctico, swing corto y posicional. Los tres superan 0,70 también en la partición posterior. | Selección de proyección dentro de entrenamiento y exclusión de activos reservados para clustering. El HMM de contexto de Mercado usa BTC y rendimientos diarios suavizados a siete días. Dos ejecuciones reprodujeron los mismos estados. En 2024 clasificó 39 días bajistas, 231 neutros y 96 alcistas. |

El [informe verificable](../../reports/validation/closure_e1_e3_e5.json) contiene los conteos por estilo. Para reconstruir las salidas desde la raíz del proyecto, ejecutar en este orden:

```powershell
python scripts/consolidate_v3_microstructure.py
python scripts/build_v3_episodes.py
python scripts/recheck_v3_scenarios.py
python scripts/verify_v3_e1_e3_e5.py
```

El primer comando descarga archivos públicos de Binance. El verificador usa la auditoría de velas vigente y comprueba sus discrepancias contra la cuarentena. Si se actualizan las velas, regenerar primero esa auditoría con `python scripts/audit_v3_sources.py`.

## Cierre técnico de E4 y E7 · 22 de noviembre de 2026

**E4.** El muestreo parte del replay CUSUM causal. Para cada contrato, experto y estilo admite la primera entrada ejecutable y espera al cierre de la posición antes de aceptar otra. Los episodios censurados y liquidados también ocupan tiempo. Conservó 205.267 episodios tácticos, 68.622 de swing corto y 10.841 posicionales. La unicidad temporal recalculada en los episodios resueltos es 1,00 en las quince combinaciones. La dependencia entre contratos o especialistas todavía puede existir. El replay completo permanece disponible en Simulador junto a la muestra E4.

**E7.** El diagnóstico consume las identidades de las señales E4 en contratos no reservados y usa un retorno común de una vela, tras costes y financiación observada. H5 compara contracción jerárquica con y sin HMM en dos pruebas cronológicas, F3 (2023) y F4 (2024), con ajuste previo de expertos, escenarios y régimen. Reporta log loss del experto seleccionado e intervalo bootstrap de diferencias mensuales de expectativa. Ningún estilo superó los dos criterios de H5.

H6 compara en investigación promedio, Hedge, Fixed Share, nivel 1 y red con abstención. Ocho bloques temporales forman 28 pares de prueba y siete trayectorias completas por estilo. Se purgan intervalos y se aplica embargo. Expertos y contexto base se ajustan antes de 2023 para CPCV. Los combinadores se ajustan dentro de cada partición. Las pruebas pareadas usan meses y corrección Benjamini–Hochberg conjunta. Para promover la red se exige primero evidencia favorable al nivel 1. Las seis políticas se abstuvieron en todas las oportunidades CPCV bajo los filtros del ensayo. Por ello, el experimento no demuestra superioridad económica ni equivalencia de políticas. Se conserva el nivel 0. La corrida reservada de 2025 mantiene su protocolo y resultados anteriores.

Resultados y evidencia por pliegue: [muestra E4](../../data/episodes/independent_v3/summary.json) y [evaluación E7](../../reports/validation/e7_orchestration/summary.json). Para regenerar, ejecutar primero `python scripts/build_v3_independent_sample.py` y después `python scripts/evaluate_v3_orchestration.py`.

## Cierre de E8, E9 y dictamen de E10 · 23 de noviembre de 2026

**E8.** Se fijaron el universo, fechas, selección de muestras y modelos en el [protocolo](../../config/validation_e8_v3.json). Los pliegues anuales de 2021 a 2024 ajustan modelos con etiquetas anteriores al inicio del año y excluyen etiquetas que cruzan su final. Se excluyen los cinco contratos reservados como objetivos. En H9 se usan precios pasados de BTC y ETH como contexto. Las pruebas de ejecución se hicieron sobre episodios y las predictivas sobre secuencias de cuatro velas. H2 utiliza una muestra determinista de hasta cuatro decisiones por contrato y mes y reproduce también los planes descartados por el filtro. H7 utiliza la etiqueta común de una vela y su expectativa direccional descuenta costes fijos. Los desgloses de H9 son por contrato. El manifiesto de activos indica que no se dispone de capitalización histórica para esa agrupación.

H1 alcanzó una reducción del error superior al 5 % en los doce pliegues de estilo y año. H7 alcanzó sus dos umbrales descriptivos en los cuatro pliegues tácticos. Ninguna de las nueve comparaciones alcanzó significación al 5 % tras la corrección conjunta. H5 y H6 reutilizan las pruebas registradas en E7, con 28 particiones, siete trayectorias y Sharpe deflactado por política y estilo. La validación se realiza sobre años de desarrollo ya conocidos por el proyecto. El [resumen E8](../../reports/validation/e8/summary.json) distingue los umbrales descriptivos de la evidencia estadística.

**E9.** Se reconstruyeron las posiciones a partir de las señales históricas congeladas de 2025 con precios de apertura, valoración horaria, margen aislado y gestión con velas de 15 minutos. El stop inicial es el mayor entre el 1,5 % de la entrada y 1,5 veces el ATR relativo de la decisión. El objetivo está a dos distancias de stop. Cuando stop y liquidación caen en una misma subvela se toma el desenlace conservador. Se aplican costes en ambos extremos y financiación observada mientras la posición está abierta. La correlación usa veinte retornos diarios cerrados. Cada apertura respeta el 10 % por contrato, el 40 % correlacionado con BTC y el 100 % bruto. Estos límites se evalúan al admitir una posición.

| Política E9 | Operaciones | Capital final sobre 10.000 USDT | Caída máxima horaria | Liquidaciones |
|---|---:|---:|---:|---:|
| Con filtro | 98 | 9.362,29 | 7,80 % | 0 |
| Control exploratorio | 117 | 9.266,98 | 8,89 % | 0 |

Ambos libros conciliaron, con cobertura de financiación durante las posiciones y cero incumplimientos de límites al abrir. El [informe de margen](../../reports/validation/margin_20261124T012928162162/report.json) corresponde a la edición revisada en velas de 15 minutos.

**E10.** El [dictamen sellado](../../reports/validation/e10/decision.json) registra que la versión actual queda descartada para despliegue. El estilo posicional incumple la comparación con la referencia, la cobertura táctica del intervalo queda en 79,73 % frente al 80 % requerido y falta evidencia de ventaja neta. El período reservado de 2025 ya fue consultado. Una prueba confirmatoria independiente sigue pendiente de una política admisible y de una muestra nueva, con criterios y activos congelados antes de abrirla.

Para reproducir los cálculos de investigación con el entorno del proyecto:

```powershell
python scripts/evaluate_v3_execution_hypotheses.py
python scripts/evaluate_v3_model_hypotheses.py
python scripts/evaluate_v3_margin.py
```

Para comprobar los archivos de la edición publicada, ejecutar `python scripts/close_v3_validation.py --verify`. El comando sin opciones resume E8 y sella el dictamen de esta edición. Si se intenta sustituir la evidencia por otra generación, el sello rechaza el cambio.

## Estado histórico de implementación (2026-11-20)

| Etapa | Resultado observado | Pendiente / criterio de cierre |
|---|---|---|
| **E1 — Datos** | Descargados 25 perpetuos Binance USDT-M Futures en 15m, 1H, 4H y 1D; 7.406.710 velas totales. No todos los contratos tienen ocho años de historial. | Revisar la alerta de coherencia entre escalas: se reportaron 225 discrepancias en 1.807.883 comparaciones y 74/75 verificaciones agregadas en `WARNING`. No declarar la consistencia global validada hasta resolverla o documentar excepciones. |
| **E2 — Features** | Reconstruidas las cuatro escalas para los 25 activos: 5.598.783 (15m), 1.399.692 (1H), 349.926 (4H) y 58.309 (1D) filas. | La validación individual de series no sustituye la resolución de la alerta inter-escalas. |
| **E3 — Simulación y costos** | Generados episodios con costos configurados; existe una corrida CPCV táctica de 15 splits. | Completar y auditar la cartera integrada, comprobar PnL/riesgo, cobertura de cuantiles y métricas de Sharpe; no hay balance de cartera aprobado. |
| **E4 — Episodios** | 13.425 posicionales, 105.706 swing corto y 413.870 tácticos, para 25 activos en cada estilo. | Unicidad media de resueltos: 0,871, 0,690 y 0,574. Revisar expertos/estilos bajo el mínimo de 0,60; no reducir el umbral para declarar aprobación. |
| **E5 — Representación y escenarios** | La corrida `identity_invariant` pasa el criterio de fuga de identidad en los tres estilos. La revisión de estabilidad fija reporta ARI 0,730 táctico, 0,301 swing corto y 0,216 posicional. | Solo el escenario táctico supera el umbral ARI 0,70. Swing corto y posicional deben permanecer bloqueados hasta una nueva evidencia elegible. |
| **E6 — Especialistas** | Los 5 baselines HistGradientBoosting por estilo se entrenaron con particiones temporales suficientes (15/15). | La comparación HGB/TCN/GRU/nulo deja 7 experto-estilos en diagnóstico y 8 bloqueados; ningún experto es elegible para despliegue. |
| **E7 — Orquestador** | La existencia de expertos entrenados no demuestra integración del Meta-Model V3 con señales y escenarios vigentes. | Integrar y probar el flujo con abstención; mantener bloqueado mientras dependencias y validación final no estén aprobadas. |
| **E8 — Validación** | El CPCV táctico completó 15/15 splits; la fracción de splits rentables fue 0, el R neto medio por split fue −0,091 y la cobertura P10–P90 media 36,6%. | Resultado no concluyente por mínimos de muestra/bloques y sin evidencia de ventaja; repetir solo tras resolver los bloqueos metodológicos y de datos. |
| **E9 — Cartera y riesgo** | Se reportan métricas por episodio y por split, no un balance de cartera integrado aceptado. | Completar asignación/exposición, costos, funding, posiciones simultáneas y curva de equity; validar drawdown y límites de riesgo. |
| **E10 — Holdout** | Los cinco activos reservados están excluidos del ajuste de especialistas. | La evaluación confirmatoria en esos activos permanece pendiente; no habilitar despliegue hasta completar el protocolo congelado. |

### Fuentes vigentes

- Baselines y particiones por estilo: `experiments/v3/specialists/training_report_{tactical,swing_short,swing_positional}.json`.
- Comparación arquitectónica: `experiments/v3/specialists/architecture_comparison/`.
- Fuga y estabilidad de representación: `experiments/v3/representation/identity_invariant/training_summary.json` y `experiments/v3/representation/stability_review/training_summary.json`.
- CPCV táctico: `reports/validation/validation_v3_cpcv_1h_tactico.json`.
- El archivo `experiments/v3/specialists/training_report_all.json` corresponde a una corrida preliminar pequeña (4.854/4.810/3.113 filas por estilo); usar los reportes por estilo para describir la corrida completa.

---

## Estructura del repositorio V3

```
QuantLab-ML/
├── data/
│   ├── raw/
│   │   ├── 15m/
│   │   ├── 1h/
│   │   ├── 4h/
│   │   ├── 1d/
│   │   └── funding/
│   ├── processed/
│   ├── features/
│   ├── scenarios/
│   └── episodes/
├── src/
│   ├── data_engine/          (collectors, funding_fetcher, validators)
│   ├── feature_engine/       (multiscale, derivatives_features, cyclic_time)
│   ├── representation_engine/(tcn_encoder, temporal_autoencoder)
│   ├── scenario_engine/      (clustering, hmm_regime, ood_detector)
│   ├── experts/              (trend, reversal, breakout, volatility, momentum)
│   │   ├── playbooks/        (reglas, stops, objetivos)
│   │   └── technique_registry/
│   ├── meta_model/           (level0_hedge, level1_shrinkage, level2_gating, arbitration)
│   ├── forecasting/          (probabilistic_engine, execution_filter)
│   ├── simulator/            (event_simulator, cost_model, order_resolution)
│   ├── validation/           (cpcv, deflated_sharpe, purging_embargo, hypotheses)
│   ├── backtesting/          (margin_risk, portfolio_allocation)
│   └── visualization/
├── config/
│   ├── reserved_assets_v3.json
│   ├── fee_schedule_binance.yaml
│   └── style_definitions.yaml
├── experiments/
├── reports/
│   ├── hypotheses/           (H1_magnitude, H2_costs, ..., H9_btc_lead)
│   └── evaluation/
├── apps/                     (dashboard Streamlit)
├── tests/
└── requirements.txt
```

---

## Principio de rigor en la implementación

Ningún módulo superior (como el Nivel 2 de orquestación o las redes V3 de especialistas)
se activará a menos que la hipótesis correspondiente demuestre un beneficio concluyente
fuera de muestra en los pliegues de validación. El test confirmatorio final se mantiene
reservado y requiere un protocolo congelado.

## Corridas de entrenamiento V3 con datos locales

Las corridas V3 deben consumir exclusivamente los parquets de `data/features/crypto/`;
no se deben cargar checkpoints, embeddings ni escenarios bajo `experiments/v2`. Después
de reconstruir las features, la representación puede entrenarse independientemente:

```powershell
.venv\Scripts\python.exe scripts\download_v3_market_data.py --online
.venv\Scripts\python.exe scripts\build_v3_features.py --timeframe all
.venv\Scripts\python.exe scripts\train_representation_v3.py --style all --output-dir experiments\v3\representation\identity_invariant
.venv\Scripts\python.exe scripts\build_v3_episodes.py --style all --output data\episodes\identity_invariant\v3_episodes.parquet
.venv\Scripts\python.exe scripts\train_specialist_models_v3.py --style all --episodes-dir data\episodes\identity_invariant --output-dir experiments\v3\specialists\identity_invariant
.venv\Scripts\python.exe scripts\compare_specialist_architectures_v3.py --style all --episodes-dir data\episodes\identity_invariant --output-dir experiments\v3\specialists\architecture_comparison
```

La descarga V3 pagina el histórico disponible por contrato para el universo de 25
activos, guarda cada temporalidad real de Futures y omite velas aún abiertas. Los
parámetros `--start-date` y `--end-date` limitan el rango si se solicitan; `--page-size`
solo regula el tamaño de página, no la longitud histórica. El manifiesto de ingesta
permite continuar desde el último timestamp descargado; los archivos sin procedencia
USDT-M Futures se reemplazan, no se incorporan al historial. No se debe entrenar hasta
reconstruir las features y revisar el reporte de coherencia entre escalas.

Los especialistas se entrenan solo con episodios resueltos, alineados a sus features
V3 y separados cronológicamente en train/calibración/test. Los artefactos nuevos quedan
aislados bajo `experiments/v3/`. Que los componentes de representación, escenarios o
especialistas estén ajustados no implica por sí mismo que el Meta-Model ni el backtest
los consuman: esa integración y su evaluación CPCV son criterios de cierre separados.
La normalización de la representación es un z-score causal por activo; el probe de
identidad se ajusta en train y se evalúa en val. La unicidad de especialistas se calcula
dentro de activo y experto. Los activos reservados quedan excluidos del ajuste de expertos.
La elegibilidad de escenarios requiere ARI medio >= 0.70 sobre una partición val fija,
comparando refits de train remuestreados por bloques temporales. La comparación de expertos
evalúa HistGradientBoosting, TCN, GRU y una referencia nula con folds temporales agrupados
por activo; los modelos insuficientes quedan bloqueados y todos siguen en modo diagnóstico
hasta completar la validación final.
