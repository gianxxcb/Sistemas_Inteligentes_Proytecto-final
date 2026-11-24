# Cambios de v2 a v3

## Motivación

La V2 operaba con horizontes de 7/14/30/90 días en gráfico diario. Los resultados
exploratorios mostraron que solo 7d y 14d superaban el baseline direccional, que el
pinball P50 no mejoraba al incondicional en ningún corte, y que la matriz de
competencia no identificó celdas `factible` en F3/F4. La investigación de bloques
temáticos (comisiones, validación, orquestador, horizontes, entrenamiento de expertos)
derivó en las siguientes conclusiones que estructuran esta versión.

---

## Decisiones clave de V3

### 1. Tres estilos de operación (nuevo)
Se reemplazan los horizontes fijos 7/14/30/90 d por tres estilos que encajan con el
ritmo real del operador (1 op/día, 3/semana, 2–3/mes) y generan muchos más episodios
independientes que los 90 días:

| Estilo | Temporalidad decisión | Gestión | Horizonte máx. | Episodios/moneda/año |
|---|---|---|---|---|
| Táctico | 1H | 15m | 6–24 h | ~4 400 (a 12 h) |
| Swing corto | 4H | 1H | 2–7 días | ~730 (a 3 d) |
| Swing/posicional | 1D + disparador 4H | 4H (1H cerca de niveles) | 10–30 días | ~73–365 |

### 2. Futuros perpetuos con apalancamiento (nuevo)
El sistema operará en **Binance USDT-M Futures**. Esto cambia el modelo de costos
completo: comisión maker/taker diferenciada, funding cada 8 h, gestión de margen y
liquidación, y requisitos de stop mínimo para que el costo no supere la ventaja.

### 3. Modelo de costos explícito dentro del aprendizaje (nuevo)
Las etiquetas se calculan netas de costos reales (comisión + spread + slippage + funding
proporcional a la duración). El filtro de ejecución bloquea operaciones cuyo valor
esperado no supere el costo total.

### 4. Paso del tiempo por eventos (nuevo)
El "reloj" del sistema no es un día fijo: el experto "despierta" solo ante cuatro
eventos definidos. Esto evita sobre-muestrear períodos tranquilos y sub-muestrear
los volátiles.

### 5. Orquestador en tres niveles escalonados (nuevo)
El gating neuronal pasa de ser la arquitectura principal a ser el nivel 2, que solo se
activa si supera estadísticamente al nivel 0 (promedio simple / Hedge) y al nivel 1
(contracción jerárquica con régimen). Esta jerarquía responde a la evidencia de que las
combinaciones simples superan a las ponderaciones sofisticadas con muestras pequeñas.

### 6. Validación con kit López de Prado completo (nuevo)
Se incorporan: muestreo CUSUM, triple barrera ajustada al playbook propio, pesos por
unicidad, purga y embargo explícitos, CPCV (trayectorias múltiples de backtest) y
Sharpe deflactado por comparaciones múltiples.

### 7. Hipótesis explícitas a probar (nuevo)
H1–H9 definidas como experimentos independientes, cada una con su protocolo.

---

## Nuevos módulos en V3

| Módulo | Contenido |
|---|---|
| `15_modelo_de_costos.md` | Comisiones Binance Futures, funding, slippage, stop mínimo, punto de equilibrio neto, filtro de ejecución |
| `16_paso_del_tiempo_por_eventos.md` | Cuatro eventos de despertar, gestión en temporalidad fina, barrera temporal dinámica |
| `17_hipotesis_y_experimentos.md` | H1–H9 con protocolo de cada experimento |

## Módulos con cambios sustanciales

| Módulo | Cambios principales |
|---|---|
| `01_data_engine` | Añade velas 1H y 15m, datos de libro de órdenes (bid-ask), datos de funding histórico, horario UTC (picos 16–17h) |
| `04_scenario_engine` | Regímenes de Markov oculto (3 estados: alcista/neutro/bajista), percentil de volatilidad como segundo eje de contexto |
| `05_specialist_models` | Muestreo CUSUM, triple barrera con exits del playbook propio, pesos por unicidad, curriculum de casos claros→ambiguos |
| `06_meta_model` | Tres niveles (simple→jerárquico→neuronal), Hedge/Fixed Share en nivel 0, contracción jerárquica en nivel 1, gating en nivel 2 solo si gana con CPCV+Sharpe deflactado |
| `07_forecast_engine` | Horizontes en horas y días (no 90d), magnitud y volatilidad como objetivos primarios, filtro de costo integrado |
| `08_evaluacion_validacion` | CPCV, Sharpe deflactado, muestra efectiva informada siempre, H1–H9 como protocolo de evaluación |
| `09_riesgo_backtesting` | Margen, liquidación, funding acumulado, stop mínimo en R, presupuesto de operaciones |
| `12_episodios_y_simulador` | Paso por eventos, gestión en temporalidad fina, resolución intravela con barra inferior, episodios horarios y de 4H |
| `13_division_de_datos` | Adapta purga a horizontes en horas, CPCV en lugar de walk-forward simple |

## Sin cambios sustanciales en V3

`02_feature_engine`, `03_representation_engine`, `05b_catalogo_tecnicas_trading`,
`10_roadmap_repositorio`, `11_redes_neuronales`, `14_diario_de_casos_y_admision`.

## Decisiones tentativas a validar con datos en V3

- Stop mínimo (~1–2 % del precio) como parámetro del sistema.
- Slippage de 0.30 % (supuesto; debe medirse con datos propios de libro de órdenes).
- Número de estados del HMM (3 candidatos; probar también 4).
- Umbral CUSUM *h* (parámetro del sistema).
- Horizonte máximo del estilo Táctico (6 h vs 24 h).
- Presupuesto de operaciones (top-3 por semana por ventaja neta esperada).
