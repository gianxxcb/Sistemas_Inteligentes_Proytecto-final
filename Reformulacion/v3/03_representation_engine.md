# Módulo 03 — Market Representation Engine (V3)

**Función:** convertir una ventana temporal de mercado en un vector latente compacto ($z_t$)
que sintetice la **estructura geométrica y dinámica** del precio sin codificar la identidad
del activo ni su precio nominal.

> **Cambios V3:** las ventanas se adaptan directamente a las temporalidades de los tres
> estilos de operación (Táctico 1H, Swing corto 4H, Swing/posicional 1D), eliminando
> la escala de 90 días (que arrojaba escasas muestras independientes en V2) y enfocando
> la representación en la extracción de patrones para el clustering y el HMM.

---

## 1. Entrada por estilo de operación

$$ X_t^{(W)} = [x_{t-W+1}, x_{t-W+2}, \dots, x_t], \qquad W \in \{24, 30, 60\} $$

| Estilo | Temporalidad | Longitud de ventana ($W$) | Tiempo físico cubierto |
|---|---|---|---|
| **Táctico** | 1H | 24 velas | 24 horas (1 día completo) |
| **Swing corto** | 4H | 30 velas | 5 días de mercado |
| **Swing/posicional** | 1D | 60 velas | 60 días (~2 meses) |

Cada vector $x_i$ contiene las variables normalizadas generadas por el Feature Engine (módulo 02).

---

## 2. Componentes de la arquitectura

### A. TCN / CNN 1D (Filtros causales y locales)
- Aprende micro-estructuras: impulsos, compresiones (squeezes), patrones de velas,
  divergencias iniciales y agotamientos de momentum.
- Convoluciones causales dilatadas (dilatación 1, 2, 4, 8) para respetar estrictamente
  el corte causal.

### B. Autoencoder temporal (Espacio latente estructural)

$$ X_t \xrightarrow{\text{Encoder}} z_t \xrightarrow{\text{Decoder}} \hat{X}_t \qquad \mathcal{L} = \lVert X_t - \hat{X}_t \rVert^2 $$

- Dimensión latente $z_t$: 16 dimensiones para el estilo Táctico, 32 dimensiones para
  Swing corto y Posicional.
- Entrenamiento completamente auto-supervisado (sin etiquetas de retorno futuro).
- **Control de fuga de activo:** se evalúa que un clasificador lineal no pueda predecir
  el ticker del activo a partir de $z_t$. El probe se ajusta en train y evalúa en val;
  el embedding solo pasa si la precisión es como máximo 1.5× la probabilidad base.

---

## 3. Salida hacia el Scenario Engine (módulo 04)

1. Vector $z_t$ continuo para clustering (HDBSCAN / k-means).
2. Distancia al espacio conocido de entrenamiento (medida de out-of-distribution / OOD).
3. Base de comparación para recuperación de análogos en el diario de casos (módulo 14).

---

## 4. Validación del módulo

- Error de reconstrucción fuera de muestra (OOS) en folds walk-forward.
- Estabilidad de escenarios: se ajustan remuestreos por bloques temporales en train y
  se comparan sus asignaciones sobre la misma partición val. Se requiere ARI medio >= 0.70;
  con datos insuficientes o estabilidad inferior, los escenarios quedan bloqueados.
- Estabilidad de representaciones entre semillas aleatorias.
- Inspección visual de vecinos más cercanos mediante UMAP para corroborar similitud
  morfológica en el gráfico.

## 5. Entrenamiento V3

El entrenamiento se ejecuta sobre `data/features/crypto/features_{1h,4h,1d}.parquet`,
con las ventanas y dimensiones latentes definidas por estilo. El escalado se ajusta
mediante un z-score causal por activo sobre las 60 barras previas (mínimo 20, `shift(1)`)
y luego se ajusta un escalador global solo con barras train de activos no reservados.
La prueba de fuga ajusta el clasificador en embeddings train y lo evalúa en val; no usa
test ni activos reservados para elegir transformaciones. Los activos reservados no
participan en el ajuste del encoder. Hasta que exista el manifiesto V3
descrito en el repositorio, se conserva la lista aprobada de
`config/reserved_assets_v2.json` únicamente como partición de activos; no se reutilizan
modelos, embeddings ni métricas V2. El test confirmatorio final requiere versionar la
lista V3 antes de ejecutarse.

```powershell
.venv\Scripts\python.exe scripts\train_representation_v3.py --style all --output-dir experiments\v3\representation\identity_invariant
```

Los checkpoints, embeddings, asignaciones de escenarios y manifiestos de procedencia
se guardan en `experiments/v3/representation/identity_invariant/`. El encoder V3 usa convoluciones
temporales causales dilatadas; los embeddings y el agrupador se ajustan sin etiquetas
futuras y únicamente con la partición train. El mismo comando ajusta el HMM de régimen
con retornos y volatilidad de BTC hasta el corte train; val/test y las asignaciones
históricas no se usan para estimar sus parámetros.
