# Módulo 04 — Scenario Engine (V3)

**Función:** detectar el tipo de estructura de mercado actual y el régimen de mercado,
para que el orquestador sepa en qué contexto están operando los expertos.

> **Cambios V3:** se añade el **modelo de Markov oculto (HMM)** como segunda capa de
> contexto (régimen alcista/neutro/bajista), además del clustering de escenarios
> existente. El estado del HMM y el percentil de volatilidad forman el contexto que
> alimenta el orquestador de nivel 1.

---

## 1. Dos capas de contexto

| Capa | Qué captura | Frecuencia de actualización |
|---|---|---|
| **Clustering de escenarios** (V2) | Estructura local del precio: triángulos, canales, compresiones, tendencias | Por episodio (en la temporalidad de decisión) |
| **Régimen HMM** (nuevo V3) | Estado global del mercado: alcista / neutro / bajista | Diario o por barra de decisión |

Estas dos capas son **complementarias**, no alternativas:
- El escenario dice "hay un triángulo simétrico en ETH".
- El régimen dice "el mercado general está en estado alcista moderado".
- El orquestador usa ambos para decidir cuál experto activar.

---

## 2. Clustering de escenarios (sin cambios sustanciales de V2)

Ver módulo 04 V2 para el detalle. Resumen:

- **Embedding:** representación matemática de la ventana del precio (30/60/120 velas
  según el estilo).
- **Clustering:** k-means o HDBSCAN sobre los embeddings del conjunto de entrenamiento.
- **Escenario actual:** clúster al que pertenece la ventana actual.
- **OOD (fuera de distribución):** si la distancia al centroide supera el percentil 95
  del entrenamiento, se marca como escenario desconocido.

**Ajuste V3:** las ventanas se calculan en la temporalidad de decisión de cada estilo
(1H para táctico, 4H para swing corto, 1D para swing/posicional).

---

## 3. Modelo de Markov oculto — Régimen de mercado (nuevo V3)

### 3.1 Motivación

Los expertos pueden comportarse de forma muy diferente según si el mercado está en
tendencia alcista, lateralización o caída. La evidencia sugiere que:

- **3 estados** es el número más común en cripto (alcista, neutro, bajista).
- **4 estados** dio el mejor pronóstico fuera de muestra en BTC/ETH/XRP en un estudio,
  pero las propiedades de los estados no son consistentes entre monedas.
- Más de 4 estados mejora el ajuste dentro de muestra y empeora las predicciones.

**Decisión tentativa:** empezar con 3 estados y probar 4 (H5 incluye esta comparación).

### 3.2 Variables de entrada al HMM

| Feature | Descripción |
|---|---|
| `log_return_1d` | Retorno logarítmico diario de BTC (referencia de mercado) |
| `volatility_20d` | Volatilidad realizada a 20 días (desviación estándar de retornos) |
| `atr_ratio` | ATR(14) / precio, normalizado |
| `volume_ratio` | Volumen / media 20d |
| `btc_dominance_proxy` | Correlación móvil 20d entre el activo y BTC |

El HMM se ajusta **solo sobre BTC** como referencia de mercado general. El estado del
régimen es el mismo para todos los activos en un momento dado (es un estado de mercado,
no de un activo específico).

### 3.3 Propiedades esperadas de los estados (a verificar con datos)

| Estado | Retorno medio | Volatilidad | Duración típica |
|---|---|---|---|
| **Alcista** | Positivo | Media | Semanas a meses |
| **Neutro/lateral** | ~0 | Baja a media | Días a semanas |
| **Bajista** | Negativo | Alta | Días a semanas |

> Estos valores son hipotéticos y deben verificarse con los datos reales del dataset.
> Las propiedades de los estados **no son estables entre monedas** (resultado de la
> literatura); el HMM se entrena sobre BTC como proxy del mercado.

### 3.4 Entrenamiento y actualización

- Se ajusta **una sola vez por pliegue** con datos de entrenamiento (sin ver el test).
- Se usa la implementación `hmmlearn` (ya instalada en el entorno).
- Número de iteraciones hasta convergencia (EM); semilla fija para reproducibilidad.
- El estado en tiempo real se decodifica con el algoritmo de Viterbi (secuencia más
  probable) o con la probabilidad posterior suave (más informativa para el orquestador).

### 3.5 Problema de etiquetado de estados entre pliegues

Los estados del HMM pueden permutarse entre pliegues (lo que era estado 0 en F3 puede
ser estado 1 en F4). Solución: **alinear los estados por sus propiedades** (mayor
retorno medio = alcista, etc.) después del ajuste de cada pliegue.

---

## 4. Percentil de volatilidad como segundo eje de contexto

Además del estado del HMM, el orquestador usa el **percentil de volatilidad del
activo** en su ventana histórica:

$$v_{\text{percentil}} = P\left(\text{ATR}(14)_t, \text{distribución en train}\right)$$

Combinado con el estado del HMM:

| Régimen | Volatilidad baja (<30p) | Volatilidad media (30–70p) | Volatilidad alta (>70p) |
|---|---|---|---|
| Alcista | Accumulation | Trending up | Momentum extremo |
| Neutro | Squeeze | Chop | Rango amplio |
| Bajista | Distribution | Trending down | Capitulación |

Esta tabla es **orientativa** (no etiquetas reales); el orquestador aprende las
asociaciones con los expertos a partir de los datos.

---

## 5. Contrato de salida del Scenario Engine (V3)

```python
ScenarioContext = {
    "scenario_id": int,          # ID del clúster de estructura local
    "scenario_embedding": array, # z_t (vector de embedding)
    "p_scenario": dict,          # P(cada escenario) = distancias suavizadas
    "is_ood": bool,              # True si está fuera de la distribución de entrenamiento
    "regime_state": int,         # 0=alcista, 1=neutro, 2=bajista (HMM)
    "regime_proba": array,       # P(estado_0, estado_1, estado_2) suave
    "volatility_percentile": float,  # percentil de ATR(14) en [0, 1]
    "style": str,                # "tactical" | "swing_short" | "swing_positional"
    "timeframe": str,            # "1H" | "4H" | "1D"
}
```

---

## 6. Uso por el orquestador (módulo 06)

- **Nivel 0 (simple):** no usa el contexto del Scenario Engine.
- **Nivel 1 (jerárquico):** usa `regime_state` y `volatility_percentile` para la
  contracción jerárquica de la matriz de competencia.
- **Nivel 2 (gating):** usa el contrato completo como entrada al gating neuronal.

---

## 7. Biblioteca de escenarios y ajuste solo con train

La biblioteca de clústeres se ajusta **exclusivamente con datos de entrenamiento de
cada pliegue**. En el test, los nuevos episodios se asignan al clúster más cercano de
la biblioteca del pliegue.

Esto vale también para el HMM: se ajusta en train, se aplica a validación y test.

---

## 8. Riesgos

1. **Inestabilidad de estados HMM:** entre reajustes, el estado puede cambiar aunque el
   mercado no lo haya hecho. Mitigar con suavizado de probabilidades posteriores.
2. **BTC como proxy:** en períodos donde el altcoin se desacopla de BTC, el régimen de
   BTC puede no ser representativo.
3. **Número de escenarios:** demasiados clústeres → pocas muestras por celda en la
   matriz de competencia. Mantener el número mínimo que capture variedad real.
4. **Deriva de escenarios entre versiones:** al reentrenar el clustering, los IDs
   cambian. Mantener mapeo entre versiones (igual que en V2).
