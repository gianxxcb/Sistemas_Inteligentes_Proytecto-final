# Módulo 17 — Hipótesis y Experimentos (NUEVO en V3)

**Función:** definir las hipótesis científicas del sistema como experimentos
independientes con protocolo explícito, para distinguir lo que se sabe de lo que se
asume y evitar el problema de comparaciones múltiples.

---

## Principio general

Cada hipótesis define:
- **Qué se afirma** (la hipótesis nula H₀ y la alternativa H₁).
- **Experimento:** cómo se prueba (datos, modelo, métrica).
- **Criterio de éxito:** qué resultado mínimo se considera confirmación.
- **Riesgo de sesgo:** qué puede salir mal en el diseño.

Los experimentos usan **datos de validación** (pliegues no test). El test final 2025 y
las monedas reservadas se usan **una sola vez**, al final.

---

## H1 — Magnitud y volatilidad se predicen mejor que la dirección

**H₀:** el modelo de magnitud no supera al incondicional fuera de muestra.
**H₁:** predecir el rango esperado (|retorno|) o la volatilidad realizada fuera de
muestra tiene un error menor que el modelo de referencia (media histórica o ATR(20)).

**Experimento:**
- Variable objetivo: rango high-low de la barra siguiente (normalizado por ATR).
- Modelos: XGBoost de regresión, red neuronal temporal, baseline ATR rolling.
- Métrica: MAE y pinball loss en cuantiles P25/P75.
- Datos: episodios de los tres estilos, folds F1–F4, purga por horizonte.

**Criterio:** mejora >5 % en MAE sobre ATR(20) en al menos 3 de 4 folds y en al menos
2 de los 3 estilos.

**Riesgo:** usar features de volatilidad como predictores de volatilidad puede
introducir autoregresión espuria; verificar con experimento nulo (series barajadas).

---

## H2 — El filtro de costo mejora el resultado neto

**H₀:** operar todas las señales con $p > p^*_{\text{sin costo}}$ da el mismo resultado
neto que aplicar el filtro de ejecución completo (módulo 15 §5).
**H₁:** el filtro de ejecución (que incluye stop mínimo, margen de seguridad $\delta$
y tope de funding) mejora la expectativa neta fuera de muestra.

**Experimento:**
- Dos políticas: sin filtro de costo vs con filtro de costo.
- Mismo universo de señales candidatas, mismos episodios.
- Métrica: expectativa neta media en R (bootstrap por bloques, IC 90 %).
- Datos: folds F3–F4 (ya consultados en V2 para diagnóstico; no test final).

**Criterio:** límite inferior del IC del exceso de expectativa neta > 0 en al menos
uno de los tres estilos.

**Riesgo:** el filtro puede descartar demasiadas señales y reducir la potencia
estadística (pocos episodios restantes).

---

## H3 — XGBoost iguala o supera a las redes neuronales con nuestros datos

**H₀:** una red neuronal temporal (TCN o GRU) supera significativamente a XGBoost en
al menos una métrica clave fuera de muestra.
**H₁:** XGBoost no es superado por la red neuronal de forma consistente.

**Experimento:**
- Mismo conjunto de features para ambos modelos.
- Métricas: log loss (clasificación de dirección), pinball P50 (cuantil central), AUC.
- Datos: folds F1–F4, mismas particiones, purga idéntica.
- Se reporta sensibilidad al hiperparámetro más importante de cada modelo.

**Criterio de éxito de H3:** XGBoost no es superado en más de 2 de 4 folds en ninguna
métrica con IC que excluya el 0.

**Consecuencia:** si H3 se confirma, se usa XGBoost como modelo base de los expertos
(V2 del módulo 05). La red neuronal (V3 del módulo 05) solo se activa si supera a
XGBoost fuera de muestra.

**Riesgo:** los hiperparámetros de XGBoost pueden estar mejor ajustados que los de la
red; reportar el esfuerzo de búsqueda de hiperparámetros de cada modelo.

---

## H4 — La hora UTC y la cercanía al funding mejoran la ejecución

**H₀:** los features de hora UTC y cercanía al funding no mejoran la predicción de
volatilidad ni el resultado de ejecución.
**H₁:** `hour_sin`, `hour_cos` y `cerca_funding` tienen importancia positiva en los
modelos de los expertos tácticos y se asocian con menor slippage o mejor timing.

**Experimento (dos partes):**
- Parte A — Importancia en modelos: ablación de features, comparar log loss con y sin
  features de hora en el estilo táctico.
- Parte B — Timing de ejecución: comparar el resultado de operaciones que entran en
  pico de actividad (16–17h UTC) vs fuera del pico, con misma señal.

**Criterio:** Part A: importancia media > 0 con permutation importance. Parte B:
diferencia de expectativa neta con IC 90 % que excluya 0 en al menos 2 de 4 folds.

**Riesgo:** confusión entre hora UTC y régimen de mercado; controlar por régimen.

---

## H5 — Un filtro de régimen (HMM) mejora la selección de expertos

**H₀:** los regímenes HMM (3 estados: alcista/neutro/bajista) no mejoran la selección
de expertos frente a usar solo los clusters del Scenario Engine (módulo 04).
**H₁:** condicionar la selección de expertos al estado del HMM mejora la expectativa
neta del orquestador fuera de muestra.

**Experimento:**
- Dos versiones del orquestador nivel 1: con y sin estado HMM como contexto.
- Métricas: log loss en la predicción de resultado del experto seleccionado; expectativa
  neta media en la validación.
- Datos: F3–F4, monedas no reservadas.

**Criterio:** mejora >3 % en log loss y límite inferior IC > 0 en expectativa neta.

**Riesgo:** el HMM se reajusta en cada pliegue; verificar que los estados sean
comparables entre pliegues (problema de etiquetado de estados).

---

## H6 — El promedio simple o Hedge iguala al gating neuronal

**H₀:** el gating neuronal (orquestador nivel 2) supera al promedio simple de expertos
(nivel 0) de forma estadísticamente significativa.
**H₁:** el promedio simple o el algoritmo Hedge no es superado por el gating neuronal.

**Experimento:**
- Tres políticas: nivel 0 (promedio simple), nivel 0b (Hedge/Fixed Share), nivel 2
  (gating neuronal).
- Métricas: log loss combinado, expectativa neta, Sharpe deflactado por comparaciones.
- Validación: CPCV con al menos 6 trayectorias, purga completa.

**Criterio de éxito de H6:** el gating neuronal no supera al promedio simple con
p-valor < 0.05 ajustado por FDR en las métricas principales.

**Consecuencia:** si H6 se confirma, el sistema permanece en nivel 0/1. El gating
neuronal solo se activa si lo refuta claramente.

**Riesgo:** el gating puede sobreajustarse con pocos episodios por celda; reportar
siempre el tamaño efectivo de muestra.

---

## H7 — El muestreo CUSUM mejora la calidad de las etiquetas

**H₀:** usar todas las barras (muestreo uniforme) da etiquetas de igual calidad que el
muestreo CUSUM.
**H₁:** los episodios detectados por CUSUM tienen mayor unicidad media, mayor separación
de señal en la variable objetivo y mejor desempeño del modelo fuera de muestra.

**Experimento:**
- Dos conjuntos de episodios: todos los cierres de barra vs solo eventos CUSUM.
- Métricas: unicidad media, AUC en validación, expectativa neta.
- Se reporta el número de episodios en cada configuración.

**Criterio:** unicidad media >0.6 (vs ~0.44 en V2) y AUC mayor en al menos 3 de 4
folds.

**Riesgo:** CUSUM puede sesgar hacia períodos de alta volatilidad; verificar cobertura
temporal.

---

## H8 — La confirmación multitemporal mejora la expectativa

**H₀:** añadir confirmación en temporalidad superior (p. ej. diario alcista para un
setup de 4H) no mejora la expectativa neta del swing corto.
**H₁:** la confluencia entre temporalidad de decisión y superior mejora la hit rate o
la expectativa neta fuera de muestra.

**Experimento:**
- Dos políticas: swing corto con y sin confirmación de temporalidad superior.
- Métricas: hit rate, expectativa neta, profit factor, número de operaciones.
- Nota: esta es una afirmación de practicantes con **evidencia empírica débil** en la
  literatura; tratar el resultado como exploratorio.

**Criterio:** mejora >5 % en expectativa neta con IC 80 % positivo (criterio más
laxo dado el soporte teórico débil).

**Riesgo:** la confirmación multitemporal reduce drásticamente el número de señales;
verificar que haya suficientes episodios para sacar conclusiones.

---

## H9 — Los retornos rezagados de BTC y ETH predicen otras monedas

**H₀:** los retornos rezagados de BTC y ETH no mejoran la predicción de retorno de
otras criptomonedas fuera de muestra.
**H₁:** incluir retornos de BTC y ETH con rezagos de 1–4 períodos mejora el AUC o el
pinball del modelo de dirección/magnitud de otras monedas.

**Experimento:**
- Ablación de features: modelo con y sin retornos de BTC/ETH rezagados.
- Monedas objetivo: las no-BTC no-ETH del universo.
- Métricas: AUC de dirección, pinball P50 de magnitud.
- Datos: folds F1–F4, purga idéntica.

**Criterio:** mejora >2 % en AUC en al menos 3 de 4 folds y en al menos 60 % de las
monedas del universo.

**Riesgo:** BTC y ETH son líderes del mercado; el efecto puede ser mayor en altcoins
pequeñas y nulo en grandes; reportar por grupos de capitalización.

---

## Gestión de comparaciones múltiples

Con 9 hipótesis y múltiples métricas por hipótesis, el riesgo de falsos positivos es
alto. Se aplica:

1. **Corrección FDR (Benjamini-Hochberg)** al reportar conjuntamente los p-valores de
   las 9 hipótesis.
2. **Sharpe deflactado** (Bailey & López de Prado) para las métricas de trading.
3. **Experimento nulo** para cada hipótesis: aplicar el mismo experimento con etiquetas
   barajadas o series sintéticas; el modelo no debe "confirmar" la hipótesis nula.
4. **Pre-registro:** las hipótesis y el protocolo se fijan antes de consultar los datos
   de validación del pliegue correspondiente.

---

## Prioridad de ejecución

| Prioridad | Hipótesis | Razón |
|---|---|---|
| Alta | H2, H3, H7 | Afectan directamente el diseño del sistema base |
| Media | H1, H5, H6 | Afectan la arquitectura del orquestador |
| Exploratoria | H4, H8, H9 | Mejoras incrementales, evidencia más débil |
