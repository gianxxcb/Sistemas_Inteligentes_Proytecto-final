# Módulo 05b — Catálogo de técnicas de trading (base de conocimiento de los expertos)

Este documento reúne **cómo funciona cada técnica** que usan los expertos del módulo 05: qué mide, su fórmula, parámetros estándar, lectura clásica, rol dentro del experto, qué debe aprender el ML y cómo suele fallar. Es el "manual" de cada técnica; los playbooks (módulo 05) dicen **cómo se combinan**, y el Feature Engine (módulo 02) las **calcula**.

> **Cómo leer los valores:** los parámetros y umbrales (14, 20, 70/30, 25…) son **convenciones de la literatura**, no ventajas probadas. En QuantLab ML son **valores iniciales y rangos de búsqueda**; el aprendizaje y la validación fuera de muestra deciden cuáles sirven. Varias fuentes advierten además contra sobreajustar los periodos hasta que el backtest "se vea bien".

---

## 0. Convenciones comunes

| Concepto | Definición |
|---|---|
| Datos | Velas diarias cripto (24/7, UTC). Un periodo N = **N días calendario** (una SMA de 20 = 20 días corridos, no 20 sesiones bursátiles) |
| Precio de referencia | Cierre, salvo que se indique |
| Barras completadas | Toda señal se evalúa **solo con velas cerradas** (sin look-ahead) |
| Pivote confirmado | Máximo/mínimo local que ya tiene *k* velas posteriores que lo confirman. Se detecta con retraso de *k* velas y ese retraso se respeta |
| Normalización | Cada salida se expresa de forma relativa (porcentaje, z-score, percentil o múltiplo de ATR) para comparar activos |

### 0.1 Bloques de construcción

**Media móvil simple (SMA) y exponencial (EMA)**

$$ SMA_N(t)=\frac{1}{N}\sum_{i=0}^{N-1}P_{t-i} \qquad EMA_N(t)=\alpha P_t+(1-\alpha)EMA_N(t-1),\;\; \alpha=\frac{2}{N+1} $$

La EMA se inicializa con una SMA de los primeros *N* datos. La EMA pesa más lo reciente y reacciona antes; la SMA pesa todo igual y es más estable.

**Suavizado de Wilder (RMA)** — usado en RSI, ATR y ADX:

$$ RMA_N(t)=\frac{RMA_N(t-1)\,(N-1)+x_t}{N} $$

Equivale a una EMA con $\alpha=1/N$. **Sustituirlo por una EMA o SMA cambia los valores**, por lo que hay que fijar una sola convención y documentarla.

**Rango verdadero (TR) y ATR** (Wilder, 1978)

$$ TR_t=\max\big(H_t-L_t,\;|H_t-C_{t-1}|,\;|L_t-C_{t-1}|\big) \qquad ATR_t=\frac{ATR_{t-1}(N-1)+TR_t}{N} $$

El primer ATR es la media simple de los primeros *N* TR. Estándar: N = 14. **ATR% = ATR / precio** es la versión comparable entre monedas.

**Volumen relativo:** $RVOL_t = V_t / \overline{V}_{20}$ (media de 20 velas; algunos usan 50). Un "volumen alto" solo tiene sentido frente al volumen habitual del activo.

---

## 1. Trend Expert

### 1.1 Medias móviles y cruces (SMA/EMA)
- **Qué mide:** dirección y persistencia de la tendencia suavizando el ruido.
- **Parámetros usuales:** 9/21 (rápidas), 20, 50 (tendencia media), 100, 200 (tendencia larga). Cruce clásico 50/200 ("golden/death cross").
- **Lectura clásica:**
  - Media creciente → tendencia alcista; decreciente → bajista; plana → rango.
  - Cruce de la rápida sobre la lenta → posible inicio de tendencia alcista (y al revés).
  - En una tendencia, una media (p. ej. EMA 20/50) actúa como **soporte/resistencia dinámico** y los retrocesos hasta ella son zonas de entrada.
  - Regla de filtro frecuente: operar largos solo con precio sobre la media larga (200) y cortos solo bajo ella.
- **Rol en el experto:** define el **sesgo direccional** y el contexto de continuación.
- **Qué debe aprender el ML:** cuándo un cruce es fiable (con ADX alto, volatilidad en expansión) y cuándo es ruido; qué periodos funcionan por activo y régimen.
- **Modos de fallo:** **whipsaws** (señales falsas repetidas) en mercados laterales; retraso (lagging) que entra tarde y devuelve parte del movimiento. Mitigación habitual: filtro de volatilidad (ATR) o de fuerza de tendencia (ADX).
- **Features:** `dist_SMA_N`, `dist_EMA_N`, pendiente normalizada, `cruce_rapida_lenta` (con edad del cruce), alineación de medias, % de días sobre la media.

### 1.2 MACD (Appel; histograma de Aspray, 1986)
- **Qué mide:** momentum de tendencia como diferencia entre dos EMA.
- **Fórmula:**

$$ MACD_t=EMA_{12}-EMA_{26} \quad Signal_t=EMA_9(MACD) \quad Hist_t=MACD_t-Signal_t $$

- **Parámetros:** (12, 26, 9) por defecto en gráficos diarios. Variantes más rápidas (p. ej. 6/13/5 u 8/17/9) para marcos cortos; se cambian solo si se prueban con las mismas reglas de entrada, invalidación y salida **incluyendo costos**.
- **Lectura clásica:**
  - MACD > 0 ⇔ EMA12 > EMA26 (sesgo alcista); < 0 sesgo bajista.
  - Cruce MACD/Signal ⇔ histograma cambia de signo.
  - El histograma refleja la **velocidad** del movimiento (si crece o se contrae).
  - Divergencias MACD–precio (ver 3.3).
- **Rol:** en el Trend Expert, confirma dirección y fuerza; en el Momentum Expert, mide aceleración (ver 5.3).
- **Qué debe aprender el ML:** que el MACD sigue extendiéndose en tendencias fuertes (a diferencia del RSI, que se satura) y que sus cruces son poco fiables en rangos.
- **Modos de fallo:** es un indicador **rezagado** (derivado de EMAs); señales falsas en laterales.
- **Features:** MACD/precio (normalizado), Signal/precio, histograma/precio, pendiente del histograma, tiempo desde el último cruce, MACD relativo a su propia desviación (z-score).

### 1.3 ADX / DMI — fuerza de tendencia (Wilder, 1978)
- **Qué mide:** **cuán fuerte es la tendencia, no su dirección**.
- **Cálculo (resumen):**
  1. Movimiento ascendente $U=H_t-H_{t-1}$ y descendente $D=L_{t-1}-L_t$; $+DM=U$ si $U>D$ y $U>0$ (si no 0); $-DM=D$ si $D>U$ y $D>0$ (si no 0).
  2. Se suavizan $+DM$, $-DM$ y $TR$ con RMA(14).
  3. $+DI=100\cdot\frac{+DM_{14}}{TR_{14}}$, $\;-DI=100\cdot\frac{-DM_{14}}{TR_{14}}$.
  4. $DX=100\cdot\frac{|+DI-(-DI)|}{+DI+(-DI)}$.
  5. $ADX=RMA_{14}(DX)$ (segundo suavizado: de ahí su memoria larga).
- **Lectura clásica (convención):**

| ADX | Lectura |
|---|---|
| < 20 | Sin tendencia / rango (las señales de seguimiento de tendencia suelen fallar) |
| 20–25 | Zona de transición |
| 25–40 | Tendencia presente, moderada a fuerte |
| > 40 | Tendencia muy fuerte (cuidado con agotamiento) |

  - **+DI sobre −DI** → predominio alcista; **−DI sobre +DI** → bajista. El cruce de DI es la señal de dirección clásica de Wilder, más fiable si el ADX ya está elevado (20–25) o subiendo.
  - Un ADX alto puede darse en tendencia alcista o bajista: **la dirección sale de los DI**.
- **Rol:** es el **filtro de régimen** del Trend Expert (¿vale la pena seguir tendencia hoy?) y sirve de contexto para otros expertos (el Reversal Expert lo usa para saber si una reversión es peligrosa).
- **Qué debe aprender el ML:** umbrales de ADX adecuados por activo; interacción ADX × cruces de medias; ADX subiendo vs bajando.
- **Modos de fallo:** muy rezagado por el doble suavizado; los umbrales (20/25) son convenciones; un periodo corto multiplica falsos cruces de DI y uno largo retrasa los cambios.
- **Features:** ADX, +DI, −DI, (+DI − −DI), pendiente del ADX, ADX percentil.

### 1.4 Estructura de máximos y mínimos (swing structure)
- **Definición estándar:** tendencia alcista = máximos y mínimos crecientes; bajista = decrecientes; ruptura de estructura = el precio pierde el último mínimo (alcista) o máximo (bajista) relevante.
- **Rol:** define la **invalidación** natural del Trend Expert.
- **Features:** número de HH/HL consecutivos, distancia al último mínimo/máximo estructural confirmado (en ATR).

---

## 2. Breakout Expert

### 2.1 Soportes y resistencias como **zonas**
- **Qué son:** niveles donde el precio se ha frenado o rebotado repetidamente. Se tratan como **zonas con grosor**, no líneas exactas.
- **Construcción:** pivotes confirmados agrupados por cercanía (tolerancia en ATR); **fuerza** = número de toques + recencia; opcionalmente volumen asociado.
- **Rol:** definen dónde está el "borde" que la ruptura debe superar.
- **Features:** distancia relativa a la zona más cercana (en ATR y %), fuerza, número de toques, antigüedad.

### 2.2 Canales de Donchian y reglas Turtle
- **Fórmula:** para N velas previas,

$$ Upper_t=\max(H_{t-N},\dots,H_{t-1}) \quad Lower_t=\min(L_{t-N},\dots,L_{t-1}) \quad Mid_t=\tfrac{Upper_t+Lower_t}{2} $$

- **Reglas históricas:**
  - Donchian original: canal de 4 semanas (≈20 días).
  - Turtle **Sistema 1:** entrada en ruptura de 20 días, salida en ruptura opuesta de 10 días.
  - Turtle **Sistema 2:** entrada en ruptura de 55 días, salida en ruptura opuesta de 20 días.
  - Stop inicial de ~2 ATR y dimensionamiento por volatilidad. En el Sistema 1 se saltaba la siguiente señal si la ruptura anterior había sido ganadora (para evitar rupturas falsas en mercados laterales).
  - **Turtle Soup** es la lógica inversa: vender una ruptura de 20 días que fracasa y regresa al rango (apuesta a falsa ruptura).
- **Variantes:** exigir **cierre** fuera del canal (no solo mecha); filtro de tendencia (p. ej. media de 50); filtro ADX (esto es una adición posterior, no una regla Turtle); entrada en retroceso a la línea media.
- **Rol:** disparador objetivo de ruptura y trailing exit.
- **Qué debe aprender el ML:** qué rupturas de N días continúan y cuáles revierten; el N y el periodo de salida según volatilidad; el efecto del filtro de tendencia/ADX.
- **Modos de fallo:** la salida de 10 días devuelve parte de la ganancia abierta (retraso); en rangos genera señales falsas repetidas. Recordar que en cripto 20 periodos = 20 días calendario.
- **Features:** distancia al Upper/Lower (en ATR), ancho del canal / precio, edad desde el último máximo de N días, `ruptura_N` (booleano de cierre fuera), canal de salida.

### 2.3 Confirmación de ruptura (cierre, volumen, retesteo)
- **Cierre:** una ruptura confiable requiere **cierre más allá del nivel**, no solo una mecha; una mecha puede ser una exploración de liquidez.
- **Volumen:** ruptura con volumen superior al habitual es más creíble; una convención frecuente es $RVOL \ge 1.2$ (120 % de la media de 20 velas). Una ruptura con volumen bajo tiene mayor probabilidad de ser falsa. Ninguna confirmación elimina el fallo.
- **Retesteo (break & retest):** tras romper, el precio vuelve al nivel y este actúa desde el otro lado (antigua resistencia como soporte). La entrada en el retesteo ofrece mejor riesgo (stop justo más allá del extremo del retesteo) pero puede perder los movimientos rápidos. Una referencia práctica es que el retesteo ocurra en pocas sesiones (del orden de 6–10) tras la ruptura.
- **Invalidación:** un **cierre de vuelta dentro de la zona** (o más allá del extremo del retesteo) = falsa ruptura.
- **Objetivos:** altura de la estructura proyectada desde la ruptura (measured move), múltiplos del riesgo (1–2R) o trailing por ATR.
- **Confirmación en marco superior:** un cierre diario puede confirmar un nivel semanal.
- **Stop:** más allá del nivel/retesteo, o 1–1.5 × ATR más allá.
- **Qué debe aprender el ML:** probabilidad de continuación vs falsa ruptura a partir de volumen relativo, compresión previa, fuerza del nivel, tendencia mayor y comportamiento del retesteo.
- **Features:** `cierre_fuera_zona`, margen de ruptura (ATR), RVOL en la ruptura, días de compresión previa, `retesteo_ok`, tiempo hasta retesteo, regreso a la zona (etiqueta de falsa ruptura).

---

## 3. Reversal Expert

### 3.1 RSI (Wilder, 1978)
- **Qué mide:** velocidad y magnitud de los cambios recientes en una escala 0–100.
- **Fórmula:**

$$ RS=\frac{\text{ganancia media}_{14}}{\text{pérdida media}_{14}} \qquad RSI=100-\frac{100}{1+RS} $$

con ganancias y pérdidas suavizadas por RMA(14).
- **Parámetros:** N = 14 por defecto; 7–9 más rápido y ruidoso; 21 más lento. Niveles 70/30; 80/20 para mercados con tendencia; 60/40 para rangos.
- **Lectura clásica:**
  - > 70 "sobrecompra" y < 30 "sobreventa", pero **son condiciones, no disparadores**: en tendencias fuertes (frecuente en cripto) el RSI puede quedarse en un extremo durante días o semanas.
  - 50 es la línea media (sesgo alcista por encima, bajista por debajo).
  - **Cambio de rango (range shift, Cardwell):** en tendencia alcista el RSI tiende a oscilar en torno a 40–80 y en bajista en torno a 20–60; en una tendencia alcista la "sobreventa" se desplaza hacia 40 y en una bajista la "sobrecompra" hacia 60.
  - Entradas prudentes: esperar a que el RSI **regrese cruzando** 30 o 70 en lugar de actuar al primer toque.
  - Divergencias (3.3) y "failure swings".
- **Rol:** en el Reversal Expert, detecta extensión y agotamiento; en el Momentum Expert, mide fuerza (ver 5.2).
- **Qué debe aprender el ML:** distinguir reversión de continuación usando ADX/tendencia, cercanía a soporte/resistencia, volatilidad y volumen; reconocer el *range shift* como detector de régimen.
- **Modos de fallo:** "atrapar cuchillos que caen" en tendencias; el suavizado retrasa el cruce de vuelta; sobreajustar el periodo.
- **Features:** RSI, RSI − 50, pendiente del RSI, días consecutivos > 70 / < 30, percentil de RSI propio, RSI sobre la razón alt/BTC.

### 3.2 Bandas de Bollinger como herramienta de reversión
- **Fórmula (estándar 20, 2):**

$$ M_t=SMA_{20},\;\; U_t=M_t+2\sigma_{20},\;\; L_t=M_t-2\sigma_{20} \quad \%B=\frac{C-L}{U-L} $$

- **Lectura:** precio fuera de la banda = movimiento extendido respecto a la volatilidad reciente. La regla de "2σ ≈ 95 %" supone distribución normal, que **los retornos de mercado no cumplen** (colas pesadas).
- **Uso para reversión:** %B < 0 o > 1 con confirmación (RSI, vuelta dentro de la banda, volumen).
- **Uso para tendencia ("caminar la banda"):** en tendencia fuerte los cierres pueden seguir pegados a la banda (%B > 0.8 sostenido) sin que sea señal de reversión.
- **Qué debe aprender el ML:** si un cierre fuera de banda es el inicio de un "paseo" o una oportunidad de reversión (clave: ADX, ancho de banda, volumen).
- **Features:** %B, distancia a bandas (en σ y %), `cierres_fuera_banda`, BandWidth (ver 4.2).

### 3.3 Divergencias (precio vs oscilador)
- **Tipos:**

| Tipo | Precio | Oscilador | Interpretación |
|---|---|---|---|
| Regular alcista | Mínimo más bajo | Mínimo más alto | Posible reversión al alza |
| Regular bajista | Máximo más alto | Máximo más bajo | Posible reversión a la baja |
| Oculta alcista | Mínimo más alto | Mínimo más bajo | Posible continuación alcista |
| Oculta bajista | Máximo más bajo | Máximo más alto | Posible continuación bajista |

- **Cómo se detectan (reglas del Feature Engine):**
  1. Marcar los **dos últimos pivotes confirmados** en precio (no pivotes aún en formación: la divergencia dibujada sobre un pivote no confirmado puede desaparecer en la vela siguiente).
  2. Comparar con los pivotes del oscilador (RSI, MACD, OBV) en las mismas velas.
  3. Exigir una **distancia mínima** entre pivotes y una **diferencia mínima** del oscilador para filtrar ruido.
- **Confirmación:** la divergencia es un aviso y no una entrada. Se espera confirmación del precio (p. ej. cierre sobre el máximo que hay entre los dos mínimos, o ruptura de una línea de tendencia).
- **Fiabilidad:** mejora con contexto (marcos superiores, niveles establecidos, extremos de RSI) y se degrada si se opera como señal aislada. En tendencias fuertes se pueden formar varias divergencias antes del giro, o ninguno.
- **Qué debe aprender el ML:** qué divergencias preceden a reversiones reales (profundidad, ubicación en RSI, cercanía a soporte, volumen).
- **Features:** `div_regular_alcista/bajista`, `div_oculta_*`, amplitud de la divergencia, distancia entre pivotes, edad de la divergencia, `confirmada` (booleano).
- **Aviso de calidad de fuente:** algunas páginas publican "tasas de acierto" por tipo de divergencia sin metodología; **no se usan**. Las tasas se medirán con nuestros datos.

### 3.4 Reversión a la media (z-score)
- **Fórmula:** $z_t=\dfrac{P_t-\mu_n}{\sigma_n}$ (media y desviación móviles de *n* = 20–60).
- **Lectura:** |z| > 2 = desviación extrema con más probabilidad de regreso; |z| < 1 = zona neutra. El z-score es comparable entre activos y marcos temporales.
- **Salidas típicas:** regreso del z hacia 0 (±0.5), el RSI de vuelta a 50, o el precio en la banda media; objetivos como múltiplos de ATR.
- **Invalidación:** una desviación muy extrema (p. ej. |z| ≥ 3–3.5) puede indicar **cambio estructural y no una oportunidad**.
- **Condiciones:** funciona mejor en **rangos** (volatilidad baja, ADX bajo, ATR bajo su media) y sufre en tendencias fuertes; **en cripto, con su tendencia marcada, es especialmente peligrosa a horizontes medios**. Se recomienda identificar el régimen sin sesgo de hindsight, y validar estadísticamente (tests de estacionariedad como ADF, exponente de Hurst) antes de asumir que una serie revierte.
- **Redundancia:** %B de Bollinger y z-score comparten información (misma media y dispersión): no son dos confirmaciones independientes.
- **Features:** z-score a varios *n*, distancia % a la media, ADX/ATR como filtro de régimen, tiempo desde la última cruce de la media.

---

## 4. Volatility Expert

### 4.1 ATR (ver 0.1)
- **Qué mide:** cuánto se mueve el activo (**magnitud**, no dirección). Incluye los gaps mediante el cierre previo.
- **Usos principales:**
  - **Stops:** 1.5–2 × ATR para swing; ~3 × ATR para posiciones de tendencia.
  - **Trailing stop "Chandelier" (LeBeau):** largo = máximo desde la entrada − 3·ATR; corto = mínimo desde la entrada + 3·ATR; solo sube (largo), nunca baja.
  - **Tamaño de posición:** $\text{tamaño}=\dfrac{\text{riesgo en dinero}}{\text{distancia al stop}}$, de modo que el riesgo por operación sea constante aunque cambie la volatilidad.
  - Detección de régimen: una expansión sostenida del ATR señala cambio de régimen.
- **Detalle práctico:** para stops conviene un ATR más corto (5–14) para que se adapte; para dimensionar, uno más largo (14–36) para que el tamaño no oscile con cada pico.
- **Aviso:** el ATR describe lo que ya ocurrió; los regímenes de volatilidad pueden cambiar bruscamente.
- **Features:** ATR%, percentil de ATR% (propio, a 120–250 días), razón ATR corto / ATR largo, pendiente del ATR.

### 4.2 Ancho de banda de Bollinger y "squeeze" de Bollinger
- **Fórmulas:**

$$ BandWidth_t=\frac{U_t-L_t}{M_t} \qquad \text{(con 20, 2: equivale a 4 veces el coeficiente de variación)} $$

- **Squeeze:** el ancho de banda está en un **mínimo respecto a su propia historia** (p. ej. el más bajo de las últimas 120 velas; 120 es una elección, no una ley). **Un squeeze marca baja volatilidad y no una dirección.** Un valor absoluto fijo (como "ancho < 0.10") no sirve entre activos distintos: se usa percentil propio.
- **Rol:** el Volatility Expert detecta la compresión; la **dirección** la deben aportar otros módulos (Breakout, Momentum, Trend).
- **Features:** BandWidth, percentil de BandWidth (120/250), días en squeeze, rango de las últimas N velas / ATR.

### 4.3 Canales de Keltner
- **Fórmula moderna (la más común en plataformas):**

$$ Mid_t=EMA_{20}(C),\quad Upper_t=Mid_t+m\cdot ATR,\quad Lower_t=Mid_t-m\cdot ATR $$

con m = 2 (1.5 también frecuente) y ATR de 10 o 20. El Keltner original (1960) usaba SMA de 10 del precio típico y el rango medio de 10 días, por lo que "Keltner (20, 2)" casi siempre se refiere a la versión moderna.
- **Diferencia con Bollinger:** Bollinger mide dispersión (desviación estándar); Keltner usa ATR, que incorpora gaps y rango completo. Bollinger reacciona antes a picos súbitos.

### 4.4 TTM Squeeze (Bollinger dentro de Keltner)
- **Definición:** **squeeze ON** cuando las bandas de Bollinger quedan completamente dentro de los canales de Keltner; **OFF** cuando salen.
- **Configuración común:** Bollinger (20, 2σ) y Keltner (20, 1.5 × ATR). Matemáticamente, estar dentro equivale a que la desviación estándar sea menor que el ATR por un factor: la volatilidad "direccional" cae por debajo de la "no direccional".
- **Dirección:** se infiere de un histograma de momentum (regresión) tras la liberación; el squeeze en sí no indica dirección.
- **Qué debe aprender el ML:** duración de squeezes que preceden expansiones relevantes; qué squeezes se resuelven con ruptura y cuáles se prolongan; tamaño esperado de la expansión (magnitud).
- **Features:** `squeeze_on`, días en squeeze, `squeeze_release` (evento), BB/KC (razón de anchos), momentum tras la liberación.

---

## 5. Momentum Expert

### 5.1 ROC — tasa de cambio
- **Fórmula:** $ROC_n=\left(\dfrac{P_t}{P_{t-n}}-1\right)\cdot100$.
- **Lectura:** sobre/bajo cero indica momentum positivo/negativo; su crecimiento indica aceleración y su caída, desaceleración. Se usa también para **rankear** activos por fuerza (rotación) y para detectar divergencias.
- **Parámetros iniciales:** n = 10–14 (probar también 5, 21, 63).
- **Features:** ROC a varios n, aceleración (ΔROC), ROC percentil propio, ROC suavizado.

### 5.2 RSI como medidor de fuerza
- Mismo cálculo (3.1), pero leído como **fuerza y persistencia**: RSI > 50 sostenido y rangos desplazados indican tendencia; RSI > 70 puede significar fuerza (no sólo "caro"). El RSI se **satura** en tendencias fuertes mientras el MACD sigue extendiéndose: justamente donde discrepan resulta informativo.

### 5.3 MACD como medidor de aceleración
- Histograma creciente = aceleración; contraído = pérdida de fuerza (ver 1.2).

### 5.4 Fuerza relativa frente a BTC / mercado
- **Fórmula:** $Ratio_t=\dfrac{P^{activo}_t}{P^{BTC}_t}$, a veces re-basado a 100 en una fecha inicial. Dividir elimina la dirección que comparten ambos y deja el **rendimiento relativo**.
- **Usos en cripto:** el par alt/BTC (p. ej. ETH/BTC) es el clásico de valor relativo; la **dominancia de BTC** (cuota de capitalización) sirve como filtro de régimen (una dominancia creciente suele acompañar debilidad de las altcoins respecto a BTC). Se puede calcular ROC, z-score o RSI **sobre la razón**.
- **Rol:** el Momentum Expert la usa para saber si la moneda lidera o rezaga; también entra como contexto del orquestador.
- **Features:** ROC(razón), z-score de la razón, ranking dentro del universo, RSI de la razón, pendiente de dominancia BTC.

### 5.5 Volumen: OBV y volumen relativo (confirmación)
- **OBV (Granville, 1963):** línea acumulada: suma el volumen si el cierre sube y lo resta si baja (sin cambio si es igual).

$$ OBV_t=OBV_{t-1}+\begin{cases}+V_t & C_t>C_{t-1}\\ -V_t & C_t<C_{t-1}\\ 0 & C_t=C_{t-1}\end{cases} $$

- **Lectura:** el nivel absoluto no importa, importa su **dirección** y si **confirma** al precio. OBV con nuevos máximos junto al precio apoya el avance; precio subiendo con OBV plano o decreciente = avance "ligero" en volumen (aviso).
- **Divergencia OBV–precio:** un aviso temprano, tratado como cualquier divergencia: **necesita confirmación del precio**.
- **Limitaciones:** el signo "todo o nada" ignora cuán fuerte fue la vela; una sola barra con volumen enorme distorsiona la línea; la calidad del dato de volumen condiciona todo. En cripto, además, conviene recordar que el volumen reportado por algunos exchanges puede no ser genuino, otra razón para usar volumen **relativo** y un exchange fiable.
- **Uso:** confirmación/filtro para Breakout (RVOL), Reversal (rebote con volumen vs débil) y Momentum.
- **Features:** RVOL, pendiente del OBV, OBV vs su media, divergencia OBV–precio, volumen en la ruptura.

---

## 6. Tabla resumen: técnica → experto y papel

| Técnica | Trend | Reversal | Breakout | Volatility | Momentum |
|---|---|---|---|---|---|
| SMA/EMA y cruces | **Núcleo** | — | Filtro de tendencia | — | — |
| MACD | **Núcleo** | Divergencia | — | — | **Núcleo** (aceleración) |
| ADX/DMI | **Filtro de régimen** | Filtro (¿tendencia fuerte?) | Filtro opcional | — | — |
| Estructura HH/HL | **Núcleo** (invalidación) | — | Contexto | — | — |
| S/R como zonas | Contexto | **Contexto** | **Núcleo** | — | — |
| Donchian / Turtle | — | — | **Núcleo** | — | — |
| Confirmación (cierre/volumen/retesteo) | — | Volumen | **Núcleo** | — | — |
| RSI | — | **Núcleo** | — | — | **Núcleo** (fuerza) |
| Bollinger (%B) | — | **Núcleo** | — | Contexto | — |
| Divergencias | — | **Núcleo** | — | — | Aviso |
| Z-score / reversión a la media | — | **Núcleo** | — | — | — |
| ATR | Stops | Stops | Objetivos/stops | **Núcleo** | — |
| BandWidth / Squeeze | — | — | Setup (compresión) | **Núcleo** | — |
| Keltner / TTM Squeeze | — | — | Setup | **Núcleo** | Dirección (histograma) |
| ROC | — | — | — | — | **Núcleo** |
| Fuerza relativa vs BTC | — | — | — | — | **Núcleo** |
| OBV / RVOL | Confirmación | Confirmación | **Confirmación** | Contexto | Confirmación |

(Un mismo indicador puede entrar a varios expertos; cada uno lo interpreta desde su playbook. Se mide la correlación entre salidas de expertos para no contar dos veces la misma información, ver módulo 05 §8.)

---

## 7. Compatibilidad y coherencia entre técnicas de un mismo experto

Reglas que el playbook respeta para que las técnicas no se contradigan:

| Experto | Reglas de coherencia |
|---|---|
| **Trend** | Cruces de medias y MACD solo se consideran fiables con ADX por encima del umbral o creciente; el sesgo direccional lo da el precio vs la media larga; la invalidación es estructural (HH/HL) |
| **Reversal** | Se activa con **extensión** (RSI/%B/z) y se valida solo con **agotamiento** (divergencia confirmada, cierre de vuelta); el ADX alto es señal de peligro; nunca se mezcla con lógica de seguimiento de tendencia |
| **Breakout** | Disparador = cierre fuera de zona/canal; confirmación = volumen y/o retesteo; invalidación = regreso a la zona; la compresión previa (setup) refuerza el evento |
| **Volatility** | Predice **magnitud**, no dirección; squeeze (setup) + liberación (disparador); si la volatilidad no se expande en el plazo esperado, la lectura se invalida |
| **Momentum** | Combina ROC, RSI (fuerza), MACD (aceleración) y fuerza relativa; una desaceleración sostenida o cruce inverso invalida |

---

## 8. Estructura interna propuesta del experto: módulos de técnica + integrador

Para que un experto "aprenda a usar cada técnica" y no solo una mezcla opaca de indicadores:

1. **Un módulo por técnica** (p. ej. en el Trend Expert: *Medias*, *MACD*, *ADX*, *Estructura*). Cada módulo recibe **solo las features de su técnica** y produce una lectura estándar: estado (`inactivo/setup/disparado`), dirección, score de aplicabilidad y calidad de señal. Se entrena y se valida **por separado** (frente a su regla fija V1).
2. **Integrador intra-experto:** un modelo pequeño que combina los módulos del experto aprendiendo **cuándo cada técnica es fiable según el contexto** (p. ej. pesar más el MACD cuando el ADX es alto; ignorar los cruces cuando es bajo). Es un *mini-gating* dentro del experto, con las reglas de coherencia del §7.
3. **Salida del experto:** el contrato estándar del módulo 05 §5, con `traza_reglas` que indica qué módulos de técnica aportaron.

**Variables de contexto compartidas por los módulos** (para que aprendan cuándo fiarse): ADX y su pendiente, percentil de ATR%, percentil de BandWidth, distancia a zona de S/R, RVOL, posición respecto a la media larga.

**Registro de técnicas (`technique_registry`)**: cada técnica se describe en un archivo declarativo (fórmula, parámetros por defecto, rangos de búsqueda, rol, condiciones de aplicabilidad, invalidación) que el código lee, de modo que añadir una técnica no requiere reescribir el experto. Ejemplo:

```yaml
technique: adx_dmi
expert: trend
role: regime_filter
formula_ref: "05b §1.3"
smoothing: wilder_rma
params:
  period: {default: 14, search: [7, 10, 14, 21]}
  strong_threshold: {default: 25, search: [20, 25, 30]}
outputs: [adx, plus_di, minus_di, adx_slope]
applicability: "adx > strong_threshold"
failure_modes: [rezago, rango, umbral_convencional]
```

**Control del sobreajuste de parámetros:** se parte de los valores estándar, se explora solo un rango acotado alrededor de ellos y se reporta la **sensibilidad** del resultado al parámetro. Si el rendimiento depende de un valor "mágico", se descarta.

---

## 9. Consideraciones específicas para cripto diario

- **24/7:** no hay gaps de apertura como en bolsa, pero sí cambios de régimen de fin de semana; los periodos son días corridos.
- **Tendencias marcadas y prolongadas:** osciladores (RSI, %B, z-score) permanecen en extremos durante largos periodos → mayor riesgo para el Reversal Expert, mayor peso del filtro ADX.
- **Colas pesadas:** los supuestos de normalidad (2σ ≈ 95 %) son optimistas; usar percentiles propios del activo.
- **Volumen heterogéneo entre exchanges y monedas:** trabajar con volumen relativo por activo y un solo exchange fuente.
- **Correlación con BTC:** la fuerza relativa y el régimen de BTC deben entrar como contexto.

---

## 10. Fuentes consultadas

Las fórmulas son estándar y coinciden entre fuentes; los umbrales y las reglas de uso son convenciones de práctica. Muchas fuentes son material educativo de brókers y plataformas, por lo que se tratan como **descripción de cómo se usan las técnicas**, no como evidencia de rentabilidad.

| Tema | Fuentes (consultadas en la investigación) |
|---|---|
| MACD | NexusFi; LuxAlgo Library (MACD); IQ Option blog; Coinpaprika (MACD en cripto); Upstox; Investing.com |
| ADX/DMI | LuxAlgo Library (ADX/DMI); TA-Lib (`ta-lib.org/functions/adx`); CMC Markets; MarketInOut; Option Alpha docs |
| RSI | Wikipedia/Wikiwand (RSI: divergencia y *range shift*); CryptoSlate; The Right Trader; The Kingfisher glossary; Spark Money |
| Bollinger | StockCharts ChartSchool; The Right Trader (BandWidth, %B, squeeze); LuxAlgo (BandWidth, TTM Squeeze); Wikipedia |
| Keltner / TTM Squeeze | LuxAlgo Library; The Right Trader; NexusFi (TTM Squeeze); Barchart; CrossTrade |
| Donchian / Turtle | LuxAlgo (Donchian Trend Rules); The Right Trader; NexusFi; Altrady; TradingView (Turtle Trade Channels) |
| Breakouts, retesteo, falsas rupturas | Forextester; Capital.com; Pepperstone; TradeZero; The Right Trader (breakout retest); TradingView (ideas educativas) |
| ATR y stops | Babypips; LuxAlgo (ATR); StockAlarm; NexusFi; IQ Option blog |
| Divergencias | LuxAlgo (Regular divergence); Capital.com; TastyTrade; scripts open-source de TradingView (reglas de detección) |
| Medias móviles / EMA | Capital.com (EMA); Britannica Money; TitanFX; Jainam |
| ROC y fuerza relativa | OptionAlpha docs (ROC); Changelly; Bitfinex blog; LuxAlgo (Ratio Charts, ROC) |
| Reversión a la media | LuxAlgo blog (mean reversion); Backtrex; QuantEdge (Substack); TradingView (ideas educativas) |
| OBV / volumen | Wikipedia; IG; Forextester; LuxAlgo (OBV); CFI; Angel One |

**Pendientes de verificación con nuestros propios datos:** todas las tasas de acierto, la eficacia de cada umbral (20/25, 70/30, 120 % de volumen, 6–10 sesiones de retesteo, 120 velas de squeeze) y la idea de que la reversión a la media falla en cripto. Esas afirmaciones son hipótesis a contrastar en la validación del módulo 08.
