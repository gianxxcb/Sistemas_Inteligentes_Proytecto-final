# QuantLab ML — Plataforma de investigación cuantitativa y descubrimiento de escenarios

Plataforma de investigación cuantitativa basada en **descubrimiento automático de escenarios** y **modelos especializados interconectados (Mixture of Experts)**, con salida **probabilística** (no señales binarias) aplicada al mercado de criptomonedas.

---

## Abrir la interfaz V3

Con Python 3.12 en la carpeta del proyecto:

```powershell
python -m pip install -r requirements.txt
python app.py
```

La aplicación lee los datos y reportes locales distribuidos con el proyecto. En **Mercado** se exploran velas de futuros y en **Cartera** se consulta el ensayo reservado de 2025. Para reproducir los informes desde los archivos de datos:

```powershell
python scripts/audit_v3_sources.py
python scripts/evaluate_integrated_v3.py
```

El ensayo usa las direcciones de los playbooks y una vela completa entre apertura de entrada y de salida. Ajusta especialistas en contratos no reservados hasta 2022, estima competencia en 2023, compara niveles y calibra cuantiles en particiones distintas de 2024. Los datos de 2025 de los cinco contratos reservados se usan para la evaluación. Las tasas de financiación ausentes se declaran en el libro de operaciones. Los reportes originales por eventos y su benchmark TCN/GRU siguen disponibles para comparación. Las tablas históricas siguientes reflejan la evaluación original de tres estilos. Las escalas 3D y 7D ya tienen datos, episodios, representaciones, escenarios y especialistas entrenados; su evidencia integrada todavía no permite aprobarlas para despliegue.

Para confirmar una corrida se exige un resumen por cada uno de los cinco estilos y una evaluación por cada una de las veinticinco combinaciones de estilo y especialista. Una combinación repetida o ausente bloquea el veredicto.

Para ejecutar la suite del backend y las pruebas de interacción de Streamlit en el entorno del proyecto:

```powershell
python -m pytest -q
```

## 1. Parámetros oficiales del proyecto

| Parámetro | Configuración |
|---|---|
| **Mercado vigente V3** | Binance USDT-M Futures, contratos perpetuos /USDT |
| **Universo V3** | 25 contratos; 5 se mantienen reservados fuera del ajuste de especialistas |
| **Cobertura histórica V3** | Desde la primera vela disponible por contrato; la antigüedad varía por activo y temporalidad |
| **Temporalidades V3** | 15m, 1H, 4H, 1D, 3D y 1W, en UTC |
| **Estilos** | Táctico (1H), swing corto (4H), posicional (1D), swing 3D y swing 7D (vela semanal UTC) |
| **Fuente primaria V3** | Binance USDT-M Futures API |
| **Salida de investigación** | Episodios con R neto, probabilidades de expertos y cuantiles; ningún experto está aprobado para despliegue |
| **Arquitectura implementada** | Episodios por eventos/CUSUM, especialistas HGB, benchmark TCN/GRU y ensayo de cartera con niveles uniforme, jerárquico y neuronal sujetos a validación. |

## Resultado histórico de la integración (tres estilos)

El ensayo original evaluó 8.692 pronósticos en los cinco activos reservados durante 2025. Sus métricas corresponden solo a táctico, swing corto y posicional; no representan los nuevos especialistas 3D/7D. La interfaz también muestra el control exploratorio, las operaciones y las observaciones de funding faltantes.

Las ventanas de la nueva evaluación usan exclusivamente información disponible al cierre de la vela. El generador de episodios por eventos también espera ahora ese cierre para buscar la primera vela de ejecución. Los resultados por eventos entregados con el proyecto corresponden al generador anterior.

## Estado de la ampliación 3D/7D (25 de noviembre de 2026)

- Se generaron features para los 25 contratos: 19.468 filas en 3D y 8.350 en 1W (la vela nativa semanal de Binance, abierta el lunes UTC).
- El replay produjo 6.319 episodios 3D (5.461 resueltos) y 4.030 episodios 7D (3.356 resueltos). Los cinco especialistas de cada estilo tienen muestras suficientes en el test temporal principal.
- Las representaciones superaron la prueba de fuga de identidad. La revisión de escenarios por bloques fue elegible: ARI medio/test de 0,896/0,932 en 3D y 0,847/0,795 en 7D.
- `run_v3_orchestrator.py --style all` carga los cinco estilos sin modelos faltantes. En la prueba local, 3D aportó 12 salidas de especialistas y se abstuvo en 7 símbolos; 7D aportó 3 salidas y se abstuvo en 22.
- La auditoría abarca 150 series. Mantiene 278 fechas discrepantes en cuarentena (incluidas las velas 3D irregulares); no hay huecos de rejilla y las advertencias quedan contabilizadas. Ver [cierre E1/E3/E5](reports/validation/closure_e1_e3_e5.json).
- **No apto para despliegue:** la evaluación integrada está bloqueada. En 3D y 7D fallan criterios de estabilidad/benchmark frente a la referencia; además, dos especialistas 7D no alcanzan los mínimos del benchmark secuencial por pliegue. Los modelos y las predicciones del orquestador son diagnósticos, no una recomendación operativa. Consultar el [informe integrado vigente](reports/validation/integrated_20261125T002006461907/report.json).

La corrida de evaluación integrada es un protocolo de investigación separado de la inferencia con los artefactos entrenados. Para probar esta última en tiempo de ejecución se usa `python scripts/run_v3_orchestrator.py --style all`.

## Referencia del experimento original V3 (corte: 2026-11-20)

Esta tabla resume los artefactos generados con la arquitectura V3. La presencia de
un modelo o reporte no equivale a validación de despliegue.

| Componente | Evidencia actual | Estado |
|---|---|---|
| Datos y features | 25 activos y cuatro temporalidades; features: 5.598.783 (15m), 1.399.692 (1H), 349.926 (4H) y 58.309 (1D) filas. | Generados; sigue pendiente resolver la alerta de coherencia entre escalas reportada por la auditoría. |
| Episodios | 13.425 posicionales, 105.706 swing corto y 413.870 tácticos; 533.001 en total. | Generados para los 25 activos. Unicidad media de resueltos: 0,871 / 0,690 / 0,574, respectivamente. |
| Representación | La corrida `identity_invariant` pasa el criterio de fuga de identidad en los tres estilos. | Elegible como representación según ese criterio; esto no valida por sí solo los escenarios. |
| Escenarios | La revisión de estabilidad con validación fija obtiene ARI 0,730 táctico, 0,301 swing corto y 0,216 posicional. | Solo táctico supera el mínimo documentado de 0,70; los otros escenarios quedan bloqueados. |
| Expertos baseline | 15/15 modelos HGB entrenados con partición temporal suficiente. | Diagnósticos; ninguno es elegible para despliegue. |
| Benchmark de arquitecturas | HGB, TCN, GRU y nulo comparados por estilo. | 7 experto-estilos evaluados solo como diagnóstico; 8 bloqueados por evidencia insuficiente. |
| Backtest CPCV táctico | 15/15 splits ejecutados; ningún split rentable; cobertura P10–P90 media 36,6%. | No concluyente: cada split queda por debajo de los mínimos de muestra/bloques. No es un balance validado de cartera. |
| Holdout final | Los 5 activos reservados no participaron en el ajuste de los baselines. | La evaluación confirmatoria final en activos reservados sigue pendiente. |

Reportes vigentes: [hoja de ruta V3](Reformulacion/v3/10_roadmap_repositorio.md),
[representación y estabilidad](experiments/v3/representation/stability_review/training_summary.json),
[entrenamiento de especialistas](experiments/v3/specialists/training_report_tactical.json) y
[CPCV táctico](reports/validation/validation_v3_cpcv_1h_tactico.json).

> **Alcance histórico:** desde la sección 2, el universo Spot, el Dataset v1.0 y las
> tablas de avance V1/V2 describen implementaciones e informes anteriores. No deben
> confundirse con los datos Futures ni con el estado vigente V3 de esta sección.

---

## 2. Universo histórico de activos (Crypto Dataset v1.0, Spot)

### A. Universo Base (25 Criptomonedas)
1. **Principales / Alta Capitalización:** BTC, ETH, BNB, SOL, XRP, ADA, AVAX, DOGE
2. **Core / Cap Media:** LTC, NEO, TRX, XLM, ETC, VET, LINK, ATOM, ALGO, BCH, DOT, NEAR, SHIB
3. **DeFi / Ecosistemas:** UNI, AAVE, FIL
4. **Polygon (Unificado):** `POLYGON_USDT` (fusión cronológica de `MATICUSDT` y `POLUSDT` tras migración de token)

### B. Control del sesgo de supervivencia (5 casos de colapso o exclusión de cotización)
- **LUNCUSDT** (Terra Classic / ex-LUNA, colapso mayo 2022)
- **FTTUSDT** (FTX Token, colapso nov 2022)
- **EOSUSDT** (Declive estructural desde top 5 histórico)
- **SRMUSDT** (Serum, proyecto delistado en 2022)
- **WTCUSDT** (Waltonchain, proyecto histórico delistado en 2023)

---

## 3. Arquitectura del repositorio

```text
QuantLab-ML/
├── data/
│   ├── raw/crypto/           # Series crudas descargadas con metadatos de auditoría
│   ├── processed/crypto/     # Calendario continuo 24/7 UTC (Dataset v1.0 congelado)
│   ├── features/crypto/      # Variables técnicas y normalizadas (Fase 2)
│   └── scenarios/            # Embeddings z_t, catálogo y biblioteca de escenarios (Fase 4)
├── src/
│   ├── data_engine/          # Ingesta, validación y normalización temporal (Fase 1)
│   ├── feature_engine/       # Features causales, registro técnico y targets (Fase 2)
│   ├── representation_engine/# Autoencoder temporal CNN1D/TCN y espacio latente z_t (Fase 3)
│   ├── scenario_engine/      # Agrupador PCA/HDBSCAN/GMM, detector OOD y catálogo (Fase 4)
│   ├── experts/              # 5 especialistas de trading, playbooks y meta-labeling (Fase 5)
│   ├── forecasting/          # Contrato tipado y fail-closed de pronósticos V2 (Módulo 07)
│   ├── simulator/            # Replay causal y resultados de episodios en R (Módulo 12)
│   ├── validation/           # Folds globales, purga, test final y admisión (Módulos 13–14)
│   ├── case_journal/         # Diario SQLite y recuperación anti-fuga (Módulo 14)
│   └── evaluation/           # Comparación walk-forward de políticas y backtest mínimo V2
├── reports/
│   ├── data_quality/         # Auditoría OHLC e inventario
│   ├── baseline/             # Métricas y benchmarks de la línea base
│   ├── representation/       # Reconstrucción, pérdida y prueba de fuga
│   ├── scenarios/            # Catálogo formal, validez OOS y diagnóstico OOD
│   └── experts/              # Comparación V1/V2/V3, Holdout y matriz de correlación
├── Reformulacion/            # Documentación maestra y especificaciones de los módulos
├── scripts/
│   ├── build_crypto_dataset_v1.py     # Pipeline reproducible Fase 1
│   ├── build_features_and_baseline.py # Pipeline reproducible Fase 2
│   ├── train_representation_engine.py # Pipeline reproducible Fase 3
│   ├── run_scenario_discovery.py      # Pipeline reproducible Fase 4
│   ├── build_v2_episodes.py           # Replay/candidatos con costos explícitos (Módulo 12)
│   ├── train_specialist_models.py     # Entrada compatible al flujo V2 walk-forward
│   ├── index_v2_cases.py              # Diario por pliegue con embeddings propios
│   ├── review_asset_admission.py      # Decisión de admisión con evidencia OOD
│   ├── preflight_v2_pipeline.py       # Auditoría de prerequisitos V2, solo lectura
│   ├── evaluate_v2_forecast_baseline.py # Baseline probabilístico OOF, sin test 2025
│   └── evaluate_v2_policies.py        # Evaluación OOF de políticas, cartera y baselines simples
├── apps/
│   └── v2_evaluation_dashboard.py     # Dashboard Streamlit mínimo para el informe de evaluación
├── tests/                    # Pruebas unitarias completas
├── requirements.txt
└── README.md
```

---

## 4. Estado: avance V1 y transición a V2

Las cifras de esta sección describen artefactos históricos. La regeneración de la
etapa 6 corrige el replay y la selección temporal del encoder. Consulte la vista
del orquestador para ver el estado y el origen de cada ejecución.

El progreso detallado de las fases 1–5 que sigue corresponde a la implementación y a los artefactos de la **V1**. La existencia de código llamado `V2` o de documentación V2 no significa que el flujo completo ya use la nueva metodología.

La comparación se basa en los [cambios V1→V2](Reformulacion/v2/CAMBIOS_v2.md) y en las especificaciones de [módulos 12](Reformulacion/v2/12_episodios_y_simulador.md), [13](Reformulacion/v2/13_division_de_datos_y_validacion.md) y [14](Reformulacion/v2/14_diario_de_casos_y_admision.md).

### Impacto de la reformulación V2 en los módulos existentes

| Módulo | Avance comprobable en el repositorio | Estado frente a V2 | Falta para alinearlo con V2 |
|---|---|---|---|
| **01 — Data Engine** | El script ejecuta descarga, procesamiento y validación de 1D y 4H; el módulo 14 combina el reporte con reglas explícitas de admisión. | **Capa de datos y decisión preliminar implementadas.** | Configurar liquidez y generar embeddings OOD del pliegue vigente para cada activo; la admisión no concede estado normal si falta evidencia. |
| **02 — Feature Engine** | Features técnicos y pivotes confirmados; escalas 30/60/120 y warm-up `200 + W`. El flujo especialista V2 usa resultados del replay, no targets de forward-return. | **Alineado para construir entradas V2.** | Las etiquetas forward-return siguen disponibles solo para baseline y análisis descriptivo; regenerar features si cambia el dataset congelado. |
| **03 — Representation Engine** | Modelo y pipeline admiten W=30/60/120, scaler/checkpoint/embeddings por escala y directorio por pliegue; selección del checkpoint usa validation. | **Reentrenado en F1–F4 para las tres escalas.** | Revisar métricas por fold: 11/12 pruebas de fuga de identidad quedaron en mínimo; F1/W120 quedó en advertencia moderada. Los artefactos no demuestran rentabilidad. |
| **04 — Scenario Engine** | El pipeline ajusta con train, corta resultados por fin de horizonte y aplica corrección FDR al periodo OOS. | **Generados los 12 catálogos fold/escala: 5 con estabilidad alta, 4 moderada y 3 baja.** | Revisar evidencia OOS; los catálogos moderados/bajos no deben tratarse como validados. Los perfiles siguen usando forward-return descriptivo, no R del replay experto. |
| **05 — Specialist Models** | `fit_episodes` entrena con R del replay; el flujo OOF aplica el punto de equilibrio a operaciones. | **OOF ejecutado:** 80 modelos (4 folds × 5 expertos × 4 horizontes) y 507.590 predicciones; el replay contiene 788.736 filas con costos aprobados de 5 bps comisión, 0 spread y 2 bps deslizamiento por lado. | Revisar resultados por fold, calibración y combinaciones sin operaciones; V3 neuronal sigue pendiente. El test final 2025 requiere confirmación explícita y se consume una sola vez. |
| **05b — Catálogo de técnicas** | Catálogo y registro declarativo existentes. | **Sin cambios sustanciales en V2.** | No requiere migración metodológica según la reformulación; verificar que sus técnicas/variables sigan alineadas con los módulos 2 y 5 al implementarlos. |

### Módulos V2: implementación y pendientes operativos

| Módulo V2 | Estado comprobable | Falta para completarlo |
|---|---|---|
| **12 — Episodios y simulador** | Replay causal diario con entrada en apertura siguiente, regla conservadora intravela, MAE/MFE, censura y contrafactuales; la política de operación no consulta etiquetas futuras. | Generados 788.736 candidatos/resultados con 14 bps round-trip. El replay usa planes fijos y barras diarias; resolución intradía 4H y aprendizaje de espera/revisión siguen pendientes. |
| **13 — División y validación** | Folds globales F1–F4, purga por fin real de replay, embargo, pesos de solapamiento, calibración cronológica, 80 modelos OOF y guardia persistente del test final. | Revisar las métricas: 8 combinaciones fold/experto/horizonte no produjeron operaciones evaluables y 4 modelos no obtuvieron calibrador; no usar esos resultados como evidencia. El test 2025 sigue reservado; el ensayo en vivo y la integración operativa del módulo 06 están pendientes. |
| **14 — Diario de casos y admisión** | Diario SQLite indexado para F1–F4, retrieval restringido por fold/fecha y exclusión de reservas; admisión integra historial, calidad, liquidez y soporte OOD de train. | Se indexaron 1.571.392 casos; 78.695 (5,0 %) no tienen embedding recuperable y quedan excluidos del retrieval. El análisis exploratorio propone 5,46 M USDT como candidato, pero no se activa: el P10 histórico varía ~20× entre folds y falta validarlo contra rendimiento. Integrar retrieval/alertas de deriva con producción. |
| **06 — Orquestador (primer incremento)** | Matriz A con predicciones OOF previas y escenarios `train`; arbitraje C fail-closed con abstención ante falta de evidencia o conflicto. | Matrices F3/F4 generadas: 0/118 y 0/120 celdas factibles. La capa C debe abstenerse con la evidencia actual.<br><br>**Pendiente:** gating neuronal B e integración al Forecast Engine; el comparador no confirmó superioridad consistente del ensemble. Ver [resultados y límites](reports/experts/meta_model_v2_first_increment.md) y [evaluación de políticas](reports/evaluation/v2_policy_comparison.md). |
| **07 — Forecast Engine** | Contrato tipado más proveedor baseline pooled: regresión cuantílica, probabilidades logísticas calibradas temporalmente, filtros de soporte y abstención explícita. Los cinco activos reservados no participan en train/calibración y se evalúan aparte. | **Prototipo OOF exploratorio en 7/14d, F1–F4.** No consulta 2025 y no se integra al orquestador. Pinball P50 mejora en 0/16 cortes; Brier en 9/16; log loss en 7/16; error de calibración de clases en 1/16.<br><br>**Pendiente:** no hay mejora consistente y la calibración de probabilidades es débil. No afinar con estos folds ya consultados ni habilitar uso operativo. Hace falta una hipótesis nueva y validación temporal independiente. Ver [informe](reports/evaluation/v2_forecast_baseline.md) y [métricas](reports/evaluation/v2_forecast_baseline.csv). |
| **08/09 — Evaluación y backtest mínimo** | Comparador OOF para F3/F4 y 7/14d, atribución y ablaciones de salida por operación y a nivel de cartera, con dashboard Streamlit. | **Bloque exploratorio implementado y reproducible.** La [evaluación de políticas](reports/evaluation/v2_policy_comparison.md), [atribución](reports/evaluation/v2_policy_attribution.md), [ablación por operación](reports/evaluation/v2_exit_management_ablation.md) y [ablación de cartera](reports/evaluation/v2_portfolio_exit_ablation.md) documentan resultados; se regeneran con `scripts/evaluate_v2_policies.py`.<br><br>**Pendiente:** los folds ya fueron consultados; la comparación de cartera usa equity realizada, no mark-to-market. Falta sensibilidad a costos y riesgo por correlación. No es backtest de producción ni consume el test final 2025. |

**Lectura del cuadro:** ya se ejecutaron replay, representación, catálogos, entrenamiento OOF e indexación por fold. La [revisión descriptiva OOF](reports/experts/specialist_walk_forward_v2_review.md) documenta métricas, variación por fold, intervalos y combinaciones sin calibración/operaciones. El módulo 06 tiene un primer incremento A/C, pero ninguna celda de F3/F4 resulta factible. El módulo 07 ya genera estimaciones de investigación, pero el baseline no mejora de forma consistente y su calibración no es suficiente para una integración operativa. Persisten advertencias de estabilidad, casos sin embedding y playbooks con planes fijos.

#### Pendientes para cerrar la migración con evidencia

1. Revisar métricas OOF por fold, horizonte y régimen; documentar las combinaciones sin operación o calibración y no rebajar sus criterios.
2. Revisar advertencias de estabilidad de escenarios y fuga de identidad; no promover catálogos débiles a evidencia validada.
3. Decidir qué cobertura de embeddings es aceptable: 5,0 % de los casos indexados no tiene vector recuperable.
4. Validar y aprobar el umbral de liquidez propuesto en el [análisis exploratorio](reports/data_quality/coin_admission/liquidity_threshold_analysis.md); el candidato no está activo.
5. Revisar las ablaciones de salida por operación y cartera junto con la atribución por activo/régimen, sin seleccionar subgrupos post hoc. No relajar la matriz A/C ni activar B o el Forecast Engine. El baseline del módulo 07 ya se evaluó exploratoriamente; no ajustar contra F1–F4 otra vez. Formular una hipótesis nueva y definir un periodo independiente no consultado antes de cualquier afirmación confirmatoria.

La resolución intradía con 4H, la política aprendida de revisión/espera y el ensayo en vivo son extensiones pendientes, fuera del alcance del backtest mínimo. El test final 2025 no se ejecuta por defecto: requiere confirmación explícita y su consumo queda registrado.

Para comprobar los prerequisitos sin escribir artefactos ni consumir el test final:

```powershell
python scripts\preflight_v2_pipeline.py
```

El preflight valida el esquema y cobertura de features, activos reservados, costos, dataset de episodios y artefactos por fold/escala. Devuelve código de salida distinto de cero mientras existan bloqueos; los costos se pueden validar pasando `--fee-bps-per-side`, `--spread-bps-per-side` y `--slippage-bps-per-side`.

### Avance de la línea base V1

Las siguientes listas documentan lo que ya se construyó bajo el flujo anterior; los checks no representan por sí solos criterios de aceptación V2.
Los informes y checkpoints citados aquí son históricos: la migración de código no reentrena automáticamente los modelos ni reescribe los resultados. El nuevo encoder guarda resultados por escala (`window_30`, `window_60`, `window_120`).

### Fase 1 completada: motor de datos de criptomonedas
- [x] Descarga de 31 series históricas completas desde Binance Spot.
- [x] Fusión cronológica de MATIC y POL en `POLYGON_USDT`.
- [x] Normalización temporal en calendario UTC continuo 24/7 delimitado en `2018-01-01` a `2025-12-31`.
- [x] Auditoría de calidad: 68,991 velas validadas con 0 fallas OHLC y 0 duplicados.
- [x] Generación de [dataset_v1_inventory.csv](data/processed/crypto/dataset_v1_inventory.csv) e informe [dataset_v1_quality_report.md](reports/data_quality/dataset_v1_quality_report.md).

### Fase 2 implementada: motor de características y línea base (confirmación de horizontes parcial)
- [x] Módulo en `src/feature_engine/` con variables normalizadas de precio, tendencia, impulso, volatilidad, volumen, estructura, divergencias, retesteos y contexto del activo.
- [x] Registro declarativo validado de 22 técnicas con parámetros por defecto, rangos de búsqueda, roles e invalidaciones en `src/feature_engine/technique_registry.json`.
- [x] Pruebas de temporalidad, ausencia de fuga, invariancia de escala, estructura y rejilla ([tests/test_feature_engine.py](tests/test_feature_engine.py)).
- [x] Generación de variables y objetivos a 7d, 14d, 30d y 90d para 30 activos.
- [x] Consolidación en [features_consolidated.csv](data/features/crypto/features_consolidated.csv) (68,991 filas x 128 columnas).
- [x] Evaluación de la línea base con partición cronológica 60/20/20 y purga de ventanas de 7 días; informe en [reports/baseline/baseline_results.md](reports/baseline/baseline_results.md).
- [x] Evaluación OOS de la rejilla 3/5/7/10/14/21/30/45/60/90d en [horizon_grid_3_90_report.md](reports/baseline/horizon_grid_3_90_report.md).
- [ ] Confirmación empírica completa: la validación retuvo 5/7/10/14d, pero de los horizontes configurados solo 7d y 14d mostraron una ventaja direccional pequeña; 30d y 90d no se retuvieron. La predicción de magnitud no superó su baseline en ningún horizonte. Se mantienen los targets actuales sin afirmar que 30d/90d estén confirmados.

### Fase 3: evidencia histórica V1; código V2 multiescala
- [x] Módulo en `src/representation_engine/` con tensores de ventanas deslizantes; el informe histórico usó 60 velas × 47 variables normalizadas.
- [x] Arquitectura N1 (CNN 1D / TCN) + N2 (autoencoder temporal) en PyTorch con espacio latente $z_t \in \mathbb{R}^{32}$.
- [x] Pruebas unitarias de arquitectura, formas y estabilidad de gradientes ([tests/test_representation_engine.py](tests/test_representation_engine.py)).
- [x] Entrenamiento histórico con varios conjuntos y pérdida Smooth L1 (59,497 ventanas: 25.6k entrenamiento, 13.8k validación, 13.6k prueba y 6.3k reserva); no es un resultado del pipeline V2 multiescala.
- [x] Reconstrucción verificada en activos no vistos (pérdida de reserva = 0.16174 frente a pérdida de validación = 0.15224).
- [x] **Prueba de fuga de identidad del activo superada:** precisión de 11.35% frente a 3.85% de azar teórico (**fuga mínima / aprobado**).
- [ ] Los embeddings y el checkpoint V1 citados en el informe histórico no están presentes en esta copia; no deben considerarse artefactos disponibles.
- [x] Informe oficial en [reports/representation/representation_report.md](reports/representation/representation_report.md).

### Fase 4: evidencia histórica V1; código V2 parcialmente migrado
- [x] Módulo en `src/scenario_engine/` con agrupamiento híbrido (PCA + HDBSCAN + GMM) y probabilidades de pertenencia blanda $P(S_k | z_t)$.
- [x] Detector fuera de distribución (`OODDetector` en `src/scenario_engine/ood.py`): clasifica ventanas atípicas como *estructura desconocida* (3.25% del total histórico) sin forzar clasificaciones espurias.
- [x] Purga y embargo temporal (`purge_overlapping_episodes`): separación mínima de 90 días, acorde con el horizonte máximo, para reducir solapamiento de resultados.
- [x] Perfilado empírico y contraste estadístico para 7d, 14d, 30d y 90d; la referencia de validez OOS del catálogo es 14d.
- [x] Pruebas unitarias completas de clustering, OOD, embargo y catálogo ([tests/test_scenario_engine.py](tests/test_scenario_engine.py)).
- [x] Catálogo histórico persistido en [scenario_registry.json](data/scenarios/scenario_registry.json) y [scenario_registry.yaml](data/scenarios/scenario_registry.yaml); pendiente regenerarlo por escala con el pipeline migrado.
- [x] Dataset histórico de 59,497 ventanas etiquetadas en [market_scenarios_clustered.csv](data/scenarios/market_scenarios_clustered.csv); no corresponde a los artefactos V2 por escala.
- [x] Informe histórico de descubrimiento en [reports/scenarios/scenario_discovery_report.md](reports/scenarios/scenario_discovery_report.md).

### Fase 5 ejecutada con el flujo anterior: modelos especialistas (resultados históricos)
- [x] Módulo en `src/experts/` con las 5 escuelas de trading especializadas: `TrendExpert`, `ReversalExpert`, `BreakoutExpert`, `VolatilityExpert` y `MomentumExpert`.
- [x] Los informes/checkpoints existentes son históricos. El flujo canónico V2 entrena con replay en R, calibración cronológica y folds OOF; falta ejecutarlo con costos aprobados para regenerar resultados.
- [x] **Advertencia:** la columna V3 del informe antiguo usa Gradient Boosting, no una red neuronal temporal; la implementación migrada rechaza V3 hasta que exista esa arquitectura.
- [x] Generalización cross-asset reportada históricamente con SOL, AVAX, NEAR y SHIB; no equivale al nuevo holdout aprobado en `reserved_assets_v2.json`.
- [x] Análisis histórico de correlación entre expertos; es descriptivo y no demuestra por sí solo complementariedad ni valor predictivo incremental.
- [x] Pruebas unitarias de especialistas completadas ([tests/test_specialist_models.py](tests/test_specialist_models.py)).
- [x] Checkpoint de modelos V2 persistido en `experiments/specialists/specialist_models_v2.pkl`.
- [x] Informe oficial de especialistas en [reports/experts/specialist_models_report.md](reports/experts/specialist_models_report.md).

### Próximas fases

- [ ] Fase 6 / módulo 06: primer incremento A/C implementado, pero F3/F4 no identifican celdas factibles; gating neuronal B e integración/evaluación integral siguen pendientes.
- [ ] Fases 7–9 / módulos 07–09: pronósticos probabilísticos, evaluación y gestión de riesgo/backtesting.
- [x] Fase 10: visualización y tablero de investigación.

El cuaderno [notebooks/QuantLab_ML.ipynb](notebooks/QuantLab_ML.ipynb) reúne la exploración reproducible de los datos y los resultados de las fases completadas. La documentación maestra y la hoja de ruta están en [Reformulacion/](Reformulacion/).

### Regenerar y revisar la etapa 6

El dashboard permite elegir entre las matrices históricas y una ejecución corregida.
En **Etapa 6 · Orquestador** se ve el progreso por etapa, la matriz A filtrada por
pliegue y horizonte, los motivos de bloqueo por celda y dos comprobaciones con
datos sintéticos del simulador y la abstención.

Para regenerar episodios, expertos, representaciones, escenarios, diarios, matrices
y evaluación con los costos del experimento anterior:

```powershell
python scripts\rebuild_v2_stage6.py --fee-bps-per-side 5 --spread-bps-per-side 0 --slippage-bps-per-side 2
```

Cada ejecución escribe bajo `experiments/v2_corrected/<ejecución>/`. La interfaz
recoge su progreso y muestra la evaluación tras completar el recálculo. El flujo
usa datos hasta el cierre de 2024. La evidencia de 2025 publicada por experimentos
históricos impide presentarlo como un periodo nunca consultado del proyecto.
La validación confirmatoria requiere definir un periodo independiente. Los cortos
del replay representan exposición lineal sintética sobre el nocional de entrada.

### Ejecutar evaluación V2 de políticas por separado

```powershell
python scripts\evaluate_v2_policies.py
streamlit run apps\v2_evaluation_dashboard.py
```

La evaluación compara políticas en F3/F4 a 7/14 días y acepta las rutas
`--oof`, `--scenario-root`, `--competence-root` y `--report-dir` de una ejecución
corregida. El [informe histórico](reports/evaluation/v2_policy_comparison.md) documenta
el resultado exploratorio `not_confirmed` anterior a las correcciones. El
backtest calcula drawdown sobre equity realizada y la referencia `buy_and_hold`
abre largos sucesivos a horizonte fijo. Consulte los módulos 08 y 09 para los límites.
## Cierre de investigación V3 · 23 de noviembre de 2026

En **Validación y calidad** aparecen las nueve hipótesis, el contraste con la referencia y el dictamen de admisión. En **Cartera simulada** se puede alternar entre el ensayo histórico y la simulación con margen aislado, cambiar de política y seleccionar una operación para ver sus velas de 15 minutos, stop, objetivo y nivel de liquidación.

La cartera con filtro terminó con 9.362,29 USDT a partir de 10.000 en 98 operaciones. Registró cero liquidaciones y una caída máxima del 7,80 % en la curva horaria. Los cinco contratos de 2025 ya habían sido examinados. La decisión de la edición V3 es **descartar el despliegue**. El informe conserva las razones y las huellas de los archivos usados.

Desde la raíz, `python scripts/close_v3_validation.py --verify` verifica el dictamen publicado. La [hoja de ruta](Reformulacion/v3/10_roadmap_repositorio.md) detalla el alcance y la reproducción de los experimentos.

