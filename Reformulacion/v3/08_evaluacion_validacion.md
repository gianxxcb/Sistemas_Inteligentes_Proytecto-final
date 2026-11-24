# Módulo 08 — Evaluación y Validación (V3)

**Función:** comprobar si el sistema aprende algo real, usando el kit completo de
López de Prado y el protocolo de hipótesis explícitas (H1–H9).

> **Cambios V3:** CPCV en lugar de walk-forward simple, Sharpe deflactado por
> comparaciones múltiples, muestra efectiva informada siempre, purga adaptada a
> horizontes en horas, protocolo de las hipótesis H1–H9 como marco de evaluación.

---

## 1. Separación de datos (actualizada V3)

La división mantiene la estructura de V2 (fechas globales, walk-forward, monedas
reservadas, purga y embargo) con estas adaptaciones:

### 1.1 Purga adaptada a horizontes en horas

La purga elimina del entrenamiento las observaciones cuya ventana de etiqueta se solapa
con el test. Para episodios tácticos (horizonte de hasta 24h):

$$\text{purga} = \text{horizonte\_max\_estilo} + \text{embargo}$$

| Estilo | Horizonte máx. | Purga | Embargo |
|---|---|---|---|
| Táctico | 24 h | 24 h | 2 h |
| Swing corto | 7 días | 7 días | 1 día |
| Swing/posicional | 30 días | 30 días | 3 días |

### 1.2 CPCV — Combinatorial Purged Cross-Validation (nuevo V3)

En lugar de un único backtest walk-forward, se usan todas las combinaciones posibles de
pliegues de entrenamiento y test (con purga), produciendo **múltiples trayectorias de
backtest**:

- **Resultado:** distribución del rendimiento (no un único número).
- **Ventaja:** permite calcular el Sharpe con su distribución y detectar si un único
  pliegue "bueno" domina el resultado.
- **Implementación:** `mlfinlab` o implementación propia basada en la descripción de
  López de Prado (2018, cap. 12).

**Número de trayectorias:** con 4 pliegues (F1–F4), CPCV produce $\binom{4}{2} = 6$
trayectorias de longitud 2. Para más pliegues: $\binom{N}{k}$ combinaciones.

### 1.3 Muestra efectiva informada siempre

En cada reporte de métrica se informa obligatoriamente:

- **n episodios totales** en el período.
- **n episodios independientes** (descontando solapamiento, equivalente a unicidad media × n).
- **n bloques temporales** (para el bootstrap).

Si n episodios independientes < 30 o n bloques < 6, la métrica se reporta como
**no concluyente** (no se usa para tomar decisiones de diseño).

---

## 2. Sharpe deflactado por comparaciones múltiples (nuevo V3)

Cuando se evalúan muchas configuraciones y se elige la mejor, el Sharpe observado está
inflado. El **Sharpe deflactado** (Bailey & López de Prado, 2014) corrige esto:

$$\widehat{SR}^* = \widehat{SR} \cdot \left(1 - \hat{\gamma}_3 \cdot \frac{\widehat{SR}}{\sqrt{T}} + \hat{\gamma}_4 \cdot \frac{\widehat{SR}^2}{4T} \right)^{1/2}$$

Ajustado además por el número de pruebas realizadas $K$:

$$p\text{-valor deflactado} = \Phi\left(\sqrt{T} \cdot \frac{\widehat{SR}^* - \widehat{SR}_0}{\hat{\sigma}_{SR}}\right)$$

donde $\widehat{SR}_0$ es el Sharpe mínimo esperado por azar dado $K$ pruebas.

**Uso:** siempre que se compare el orquestador contra líneas base o se evalúen hipótesis
de trading (H1, H2, H5, H6).

---

## 3. Métricas — cuatro dimensiones (actualizada V3)

| Dimensión | Métricas |
|---|---|
| **Predicción** | Log Loss, Brier Score, AUC, pinball loss (cuantiles P10–P90) |
| **Calibración** | Reliability diagrams, ECE, cobertura de intervalos por horizonte |
| **Escenarios** | Estabilidad de clusters y del HMM entre pliegues, recurrencia, separación |
| **Trading** | CAGR, Sharpe deflactado, Sortino, max drawdown (realizado), profit factor — **tras costos**; R realizado medio, expectativa neta por episodio, MAE/MFE, funding acumulado |

---

## 4. Protocolo de evaluación por hipótesis (nuevo V3)

Cada hipótesis H1–H9 (módulo 17) se evalúa con su propio protocolo:

1. **Pre-registro:** antes de consultar los datos, se fija el diseño del experimento.
2. **Datos:** solo pliegues de validación (F1–F4). El test final 2025 permanece intacto.
3. **Corrección FDR:** los p-valores de las 9 hipótesis se corrigen conjuntamente.
4. **Experimento nulo:** aplicar el mismo experimento con etiquetas barajadas.
5. **Reporte:** intervalo de confianza (bootstrap por bloques), tamaño efectivo de
   muestra, si es concluyente o no.

---

## 5. Generalización entre criptomonedas (igual que V2)

Matriz de pruebas: **monedas vistas / reservadas × periodo entrenamiento / futuro**.

La diferencia entre celdas calibra los umbrales de admisión de monedas (módulo 14).

---

## 6. Evaluación de expertos y del orquestador (actualizada V3)

### 6.1 Expertos

- Hit-rate y expectativa neta (en R) de sus eventos, con y sin filtro de ejecución.
- Profit factor, drawdown, calibración de `p_exito`.
- Sensibilidad al parámetro más importante del playbook.
- Comparación V1 → V2 → V3 en cada estilo.

### 6.2 Orquestador (tres niveles)

- Nivel 0 vs incondicional: si no supera, el sistema no tiene valor.
- Nivel 1 vs nivel 0: si no supera, mantener nivel 0.
- Nivel 2 vs nivel 1: si no supera, mantener nivel 1 (H6).
- Calidad de la abstención (las predicciones evitadas ¿eran peores?).
- Distribución de pesos del nivel 2 (sin colapso hacia un experto).
- Sharpe deflactado en cada comparación.

### 6.3 Ablaciones del sistema

| Ablación | Qué mide |
|---|---|
| Sin filtro de ejecución | Valor del filtro de costos (H2) |
| Sin CUSUM | Valor del muestreo por eventos (H7) |
| Sin régimen HMM | Valor del contexto de régimen (H5) |
| Sin pesos de unicidad | Valor de la corrección de dependencia |
| Sin features horarias | Valor de la hora UTC y funding (H4) |
| Sin retornos de BTC/ETH | Valor de la correlación entre activos (H9) |

---

## 7. Controles contra el autoengaño (igual que V2 + extensión)

- **Experimento nulo:** etiquetas barajadas; el sistema no debe encontrar "ventaja".
- **Corrección por comparaciones múltiples** (FDR + Sharpe deflactado).
- **Intervalos de confianza** (bootstrap por bloques), no solo medias.
- **Análisis por régimen** (alcista, bajista, lateral) y por activo.
- **Reporte de n efectivo** en cada métrica.
- **Test final independiente:** periodo y activos reservados definidos antes de la
  ejecución; se consulta una sola vez al final del desarrollo.

---

## 8. Promoción a revisión confirmatoria (igual que V2)

Una política se promueve a revisión confirmatoria si, **por separado en F3/F4 y en
activos vistos/reservados**:

- Al menos 30 operaciones y 6 bloques temporales.
- Límite inferior del IC del Sharpe deflactado > 0 frente a buy & hold y reglas
  disparadas.
- El experimento nulo no "confirma" la hipótesis.

Cumplir esto **no autoriza uso operativo** ni sustituye el test final.
Estos son criterios de diseño; no implican que la corrida CPCV táctica disponible los
haya satisfecho.

---

## 9. Riesgos

1. **CPCV costoso computacionalmente:** con muchos activos y estilos, el número de
   episodios por trayectoria puede ser pequeño; verificar n efectivo.
2. **Sharpe deflactado sensible al número de pruebas K:** documentar exhaustivamente
   cuántas configuraciones se probaron.
3. **Bootstrap por bloques con episodios horarios:** el tamaño de bloque debe adaptarse
   a la autocorrelación de los retornos tácticos (tentativo: bloques de 24–48h).

## 10. Evidencia de validación V3 disponible (2026-11-20)

Se ejecutó una corrida CPCV para el estilo táctico (1H) con 6 bloques y 2 bloques de test,
completando 15/15 splits sobre 36.000 filas y 24 símbolos. La fracción de splits rentables
fue 0; el R neto medio por split fue −0,091, el Sharpe anualizado medio −5,60 y la cobertura
P10–P90 media 36,6%.

Este resultado no es confirmatorio: los splits no alcanzan los mínimos de muestra efectiva
y bloques indicados en §1, por lo que su dictamen es **no concluyente y sin ventaja observada**.
No se ejecutó aquí una evaluación final confirmatoria en los cinco activos reservados. El
reporte completo es [`validation_v3_cpcv_1h_tactico.json`](../../reports/validation/validation_v3_cpcv_1h_tactico.json).
La cobertura registrada tampoco debe confundirse con una validación de calibración suficiente.
