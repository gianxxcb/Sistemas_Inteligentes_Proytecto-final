# Módulo 05 — Specialist Models / Expertos (V3)

**Función:** cada experto es una escuela de trading completa que domina una familia de
técnicas y aprende cuándo y cómo aplicarlas, ahora con muestreo CUSUM, triple barrera
ajustada al playbook propio, pesos por unicidad y curriculum de aprendizaje.

> **Cambios V3:** muestreo de episodios con filtro CUSUM (no todas las barras),
> etiquetado con triple barrera y exits del playbook propio, pesos por unicidad,
> curriculum de casos claros → ambiguos, features de hora UTC y cercanía al funding
> para el estilo táctico.

---

## 1. Principio de diseño (sin cambios)

Ver módulo 05 V2 §1–2. Resumen:

- Separación experto (técnica) / orquestador (cuándo usar al experto).
- Cada experto tiene un **playbook** con 6 bloques: condiciones, disparador,
  confirmación, invalidación, objetivo, modos de fallo.
- Los expertos no ven los resultados del clustering; solo la estructura de la ventana.

---

## 2. Muestreo de episodios con CUSUM (nuevo V3)

En lugar de generar un episodio candidato en cada cierre de barra, se usa el
**filtro CUSUM simétrico** (módulo 16 §6) para muestrear solo cuando ocurre un evento
informativo significativo:

```
h = umbral CUSUM (parámetro del sistema, en unidades de ATR o retorno)
Cuando S_t+ ≥ h o S_t- ≤ -h → generar candidato y resetear el acumulador
```

**Efecto:** los episodios están más separados en el tiempo, aumenta la unicidad media
y se reduce el número de episodios redundantes. Se espera unicidad media >0.6 (H7).

**Parámetro $h$:** valor tentativo $h = 1 \times ATR(20)$. Se calibra en validación.

---

## 3. Etiquetado: triple barrera ajustada al playbook (nuevo V3)

La triple barrera de López de Prado se adapta para usar **los propios niveles del
playbook** en lugar de barreras genéricas:

| Barrera | V2 (genérica) | V3 (playbook) |
|---|---|---|
| Objetivo (superior/inferior) | k₁ × ATR | Nivel de objetivo del playbook (p. ej. resistencia, extensión Fibonacci) |
| Stop (invalidación) | k₂ × ATR | Nivel de invalidación del playbook (mínimo estructural, línea de tendencia) |
| Tiempo | Horizonte fijo | Horizonte máximo del estilo (6h / 7d / 30d) |

**Resultado:** el experto aprende a juzgar **si su propia técnica va a funcionar**,
no si el precio va a subir o bajar en general. Las etiquetas son:

- `+1`: el objetivo se toca antes que el stop y antes del tiempo máximo.
- `-1`: el stop se toca antes que el objetivo y antes del tiempo máximo.
- `0`: ninguno de los dos se toca (el tiempo máximo expira).

**Resultado neto:** se calcula con el R realizado, neto de costos (módulo 15 §6).

**Resolución de ambigüedad intravela:** según el protocolo del módulo 16 §4.

---

## 4. Pesos por unicidad (nuevo V3)

Cuando dos episodios se solapan, se ponderan en el entrenamiento según su unicidad
(módulo 16 §7):

```python
starts, ends = posiciones_en_grid(intervalos)
concurrencia = cumsum(delta_en(starts, +1) + delta_en(ends + 1, -1))
unicidad = promedio_por_intervalo(1 / concurrencia, starts, ends)
pesos = unicidad / unicidad.sum()
```

La implementación usa acumuladores de diferencias y sumas prefijas para evitar recorrer
repetidamente los mismos intervalos. Estos pesos se pasan como `sample_weight` al
entrenamiento del baseline o de la red neuronal.

---

## 5. Curriculum de aprendizaje (nuevo V3)

El experto aprende en dos fases ordenadas:

| Fase | Qué episodios usa | Qué aprende |
|---|---|---|
| **1 — Casos claros** | Episodios con resultado muy positivo (R neto > 1.5) o muy negativo (R neto < -0.8) | El patrón básico de éxito/fracaso de la técnica |
| **2 — Casos ambiguos** | Todos los episodios, incluidos los marginales (R neto entre -0.8 y 1.5) | Matices: cuándo una señal aparentemente buena falla, cuándo una débil funciona |

La fase 1 estabiliza el modelo antes de exponer los casos difíciles. Se valida que el
modelo de fase 1 no sobreajuste los casos extremos antes de pasar a la fase 2.

---

## 6. Features adicionales para el estilo táctico (nuevo V3)

Los expertos del estilo táctico (1H) reciben features de temporalidad horaria:

| Feature | Descripción | Hipótesis |
|---|---|---|
| `hour_sin`, `hour_cos` | Codificación cíclica de la hora UTC (24h) | H4 |
| `cerca_funding` | Binaria: ±30 min de las liquidaciones de funding | H4 |
| `funding_rate_last` | Tasa de funding de la última liquidación | H4 |
| `btc_return_lag1h` | Retorno de BTC en la hora anterior | H9 |
| `btc_return_lag4h` | Retorno de BTC en las 4 horas anteriores | H9 |
| `volume_taker_buy_ratio` | Volumen comprador / volumen total | Señal de presión compradora |

---

## 7. Tres versiones de cada experto (igual que V2)

| Versión | Descripción |
|---|---|
| **V1 — Reglas fijas** | Playbook con parámetros estándar, sin aprendizaje |
| **Baseline V3** | HistGradientBoosting tabular con pesos de unicidad y cuantiles |
| **Candidatos V3** | TCN y GRU sobre secuencias causales; comparar con el baseline y una referencia nula (H3) |

La comparación actual no declara un ganador ni habilita despliegue. Los activos reservados
se mantienen intactos para la prueba confirmatoria final.

---

## 8. Contrato de salida estándar (actualizado V3)

Igual que V2, con estas adiciones:

| Campo nuevo | Contenido |
|---|---|
| `style` | `tactical` / `swing_short` / `swing_positional` |
| `costo_total_estimado_R` | Costo total estimado en R (módulo 15) |
| `p_equilibrio_neto` | $p^*$ con costo total incluido |
| `filtro_ejecucion_ok` | True si todos los criterios del filtro de ejecución se cumplen |
| `funding_estimado` | Costo de funding estimado para el horizonte del plan |

---

## 9. Playbooks por experto (igual que V2)

Los cinco expertos (Trend, Reversal, Breakout, Volatility, Momentum) se mantienen con
sus playbooks de V2. Ver módulo 05 V2 §4 para el detalle.

**Ajuste V3:** cada playbook especifica ahora también:
- La temporalidad de gestión (15m, 1H, 4H) en la que se revisan los planes.
- El stop mínimo en términos de ATR (p. ej. "stop mínimo: 1× ATR(14)").
- La acción específica ante E1, E2, E3 y E4 (módulo 16 §2).

---

## 10. Entrenamiento (actualizado V3)

1. **Datos:** episodios CUSUM de activos no reservados. La comparación de arquitecturas usa
   folds por activo: el bloque de test no participa en train/calibración y solo se evalúa
   después del corte temporal; train y calibración purgan episodios que cruzan el límite.
2. **Pesos:** por unicidad (§4).
3. **Curriculum:** fase 1 (casos claros) → fase 2 (todos).
4. **Calibración:** `p_exito` calibrada en bloque cronológico separado.
5. **Validación:** V1 → V2 → V3 en ese orden; V3 solo si supera a V2 (H3).

---

## 11. Validación individual (igual que V2)

Antes de conectarse al orquestador, cada experto debe demostrar:

1. Rendimiento propio frente a V1, frecuencia incondicional y disparador aleatorio.
2. Calibración de `p_exito` y cobertura de cuantiles.
3. Aplicabilidad válida: score alto → mejores resultados.
4. Generalización a activos no vistos y periodos posteriores.
5. Coherencia del playbook (revisión manual).
6. Experimento nulo con series barajadas.

---

## 12. Estado de implementación

- El flujo CUSUM reemplaza el bucle de todas las barras en el generador de candidatos.
- El etiquetado de triple barrera usa los niveles del playbook del episodio, no barreras
  ATR genéricas.
- Los pesos de unicidad se calculan antes del `fit` y se pasan como `sample_weight`.
- La construcción de episodios recorta barras de gestión con búsqueda temporal y carga
  únicamente las columnas usadas por los playbooks para limitar memoria en el universo completo.
- La unicidad y los pesos se calculan por activo y experto; los activos reservados se
  excluyen de train/calibración/test temporal para mantenerlos fuera del ajuste.
- El curriculum se implementa como un filtro de filas en dos llamadas a `fit`.
- Las features horarias se añaden al `FeatureEngine` (módulo 02) como features opcionales
  activables por `style == "tactical"`.
- El entrenador disponible en el repositorio usa HistGradientBoosting para clasificación
  y cuantiles como baseline del pipeline V3. `scripts/compare_specialist_architectures_v3.py`
  compara ese baseline con TCN, GRU y una referencia nula usando ventanas causales, folds
  agrupados por activo, calibración cronológica y métricas por activo no visto. Los modelos
  con train/calibración/test insuficientes quedan bloqueados. La comparación y sus métricas
  son diagnósticas: no acreditan los seis criterios de §11 ni habilitan despliegue; los
  activos reservados y el test confirmatorio final permanecen intactos.

### Resultados de la corrida completa (2026-11-20)

| Estilo | Episodios totales | Resueltos | Unicidad media de resueltos | Baselines con test temporal suficiente |
|---|---:|---:|---:|---:|
| Posicional | 13.425 | 11.842 | 0,871 | 5/5 |
| Swing corto | 105.706 | 94.771 | 0,690 | 5/5 |
| Táctico | 413.870 | 349.735 | 0,574 | 5/5 |

Los 15 baselines son HistGradientBoosting; los resultados de test son diagnósticos y
ningún modelo tiene `deployment_eligible=true`. En la comparación HGB/TCN/GRU/nulo,
7 combinaciones estilo-experto solo se evaluaron de forma diagnóstica y 8 quedaron
bloqueadas por evidencia insuficiente en entrenamiento/calibración o activos no vistos.
Algunos expertos no alcanzan la unicidad de referencia 0,60, en particular varios del
estilo táctico y Volatility en swing corto. Estos casos no deben reportarse como listos.

Fuentes por estilo: [posicional](../../experiments/v3/specialists/training_report_swing_positional.json),
[swing corto](../../experiments/v3/specialists/training_report_swing_short.json),
[táctico](../../experiments/v3/specialists/training_report_tactical.json) y
[comparaciones arquitectónicas](../../experiments/v3/specialists/architecture_comparison/).
El resumen `training_report_all.json` pertenece a una corrida preliminar pequeña; no
sustituye los reportes por estilo de la corrida completa.
