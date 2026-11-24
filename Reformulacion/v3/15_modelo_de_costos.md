# Módulo 15 — Modelo de Costos (NUEVO en V3)

**Función:** cuantificar el costo total de cada operación en Binance USDT-M Futures
con apalancamiento, integrarlo en las etiquetas de aprendizaje y filtrar operaciones
cuyo valor esperado no supere ese costo.

> **Nota:** las tarifas concretas deben verificarse en la página oficial de Binance
> antes de cada uso; las cifras aquí son las vigentes en la investigación y pueden
> cambiar. Las fuentes usadas son en parte páginas de terceros.

---

## 1. Componentes del costo

### 1.1 Comisión de trading (Binance USDT-M Futures)

| Tipo | Tarifa base | Con descuento BNB (10 %) |
|---|---|---|
| Maker (orden limit) | 0.020 % | 0.018 % |
| Taker (orden market) | 0.040–0.050 % | 0.036–0.045 % |

- **Estrategia:** usar órdenes limit siempre que sea posible para pagar maker en lugar
  de taker.
- El descuento BNB en futuros es solo del **10 %** (no 25 % como en spot).
- VIP 1 exige >1 M USD en 30 días: no aplica al operador individual.

**Costo de ida y vuelta mínimo (ambos lados, limit):**

$$c_{\text{comisión}} = 2 \times 0.020\% = 0.040\%$$

**Con mix real (entrada limit, salida market):**

$$c_{\text{comisión}} = 0.020\% + 0.040\% = 0.060\%$$

### 1.2 Spread y slippage

- **Spread:** diferencia bid-ask en el libro de órdenes. Para majors (BTC, ETH) en
  Binance Futures es típicamente <0.01 %, despreciable.
- **Slippage:** impacto de mercado al ejecutar. Para posiciones pequeñas en BTC/ETH
  puede ser <0.05 %, pero para altcoins o en momentos de volatilidad puede ser
  significativo.
- **Supuesto de trabajo:** 0.15 % por lado (0.30 % ida y vuelta) como estimación
  conservadora que incluye slippage en altcoins. **Este valor debe medirse con datos
  propios del libro de órdenes.**

### 1.3 Funding en perpetuos

Los contratos perpetuos cobran/pagan una tasa de *funding* cada **8 horas**
(normalmente 00:00, 08:00 y 16:00 UTC). Solo paga quien tiene posición abierta en
ese momento exacto.

| Parámetro | Valor |
|---|---|
| Tasa base | 0.01 % cada 8 h |
| Equivalente anual (base) | ~11 % |
| En manías alcistas | hasta 0.10 % cada 8 h (>100 % anual) |

**Costo de funding por duración:**

$$c_{\text{funding}}(T) = \sum_{k} r_k \approx 0.01\% \times \left\lfloor\frac{T_{\text{horas}}}{8}\right\rfloor$$

| Duración posición | Pagos de funding aprox. | Costo funding aprox. |
|---|---|---|
| 6 h (táctico) | 0–1 | 0–0.01 % |
| 1 día | 3 | 0.03 % |
| 3 días | 9 | 0.09 % |
| 7 días | 21 | 0.21 % |
| 15 días | 45 | 0.45 % |
| 30 días | 90 | 0.90 % |

> El funding se suma al costo de la operación y se descuenta de la etiqueta. Para
> posiciones largas es un costo adicional; para cortas puede ser un ingreso (si la
> tasa es positiva, los largos pagan a los cortos).

### 1.4 Efecto del apalancamiento

El apalancamiento **multiplica el costo en efectivo** relativo al margen:

$$c_{\text{efectivo}} = c_{\text{nominal}} \times \text{apalancamiento}$$

Con 5× de apalancamiento y 0.06 % de comisión nominal:
comisión efectiva sobre margen = 0.30 %.

> Por eso la gestión del apalancamiento es parte del modelo de riesgo (módulo 09),
> no solo un multiplicador de ganancia.

---

## 2. Costo total por operación

$$c_{\text{total}} = c_{\text{comisión}} + c_{\text{slippage}} + c_{\text{funding}}(T)$$

**Costo total típico por estilo** (sin apalancamiento, ida y vuelta):

| Estilo | Horizonte típico | Comisión | Slippage | Funding | **Total** |
|---|---|---|---|---|---|
| Táctico | 6–24 h | 0.06 % | 0.30 % | 0–0.03 % | **~0.36–0.39 %** |
| Swing corto | 2–7 días | 0.06 % | 0.30 % | 0.09–0.21 % | **~0.45–0.57 %** |
| Swing/posicional | 10–30 días | 0.06 % | 0.30 % | 0.30–0.90 % | **~0.66–1.26 %** |

---

## 3. Punto de equilibrio neto

La fórmula del módulo 12 se extiende con el costo total:

$$p^* = \frac{1 + c_R}{R + 1}$$

donde $c_R$ es el costo total expresado en múltiplos de R (distancia al stop):

$$c_R = \frac{c_{\text{total}}}{\text{distancia al stop (\%)}}$$

**Tabla de $p^*$ por distancia al stop y costo total** (R = 2):

| Distancia stop | $c_R$ (costo 0.39 %) | $p^*$ | $c_R$ (costo 0.57 %) | $p^*$ |
|---|---|---|---|---|
| 0.5 % | 0.78 R | 59 % | 1.14 R | 71 % — **inviable** |
| 1 % | 0.39 R | 46 % | 0.57 R | 52 % |
| 2 % | 0.195 R | 40 % | 0.285 R | 43 % |
| 4 % | 0.098 R | 37 % | 0.143 R | 38 % |

> **Conclusión:** con stops menores de ~1 % los costos consumen la ventaja incluso con
> apalancamiento moderado. El stop mínimo es un **parámetro del sistema**, no una
> decisión de cada operación.

---

## 4. Stop mínimo como parámetro del sistema

Se define `stop_min_pct` como la distancia mínima al stop (en % del precio) por debajo
de la cual el sistema no opera.

- **Valor tentativo:** 1.0–2.0 % dependiendo del activo y la volatilidad.
- Se calibra en validación: el valor que maximiza la expectativa neta fuera de muestra,
  no el que maximiza operaciones.
- Se expresa también en ATR: `stop_min_atr = stop_min_pct / ATR_actual`.

---

## 5. Filtro de ejecución

El sistema solo genera una operación si se cumplen **todos** los criterios:

1. $p_{\text{éxito}} > p^*$ (esperanza neta positiva).
2. Distancia al stop ≥ `stop_min_pct`.
3. Valor esperado neto $E[R_{\text{neto}}] = p \cdot R - (1-p) - c_R > \delta$, donde
   $\delta > 0$ es un margen de seguridad (p. ej. 0.1 R).
4. Funding acumulado estimado no supera el 30 % del objetivo esperado.

Cuando no se cumple algún criterio, el sistema registra el candidato como "filtrado por
costo" (contrafactual, módulo 12 §5) para aprender de ello.

---

## 6. Integración en las etiquetas

Las etiquetas de entrenamiento se calculan **netas de costos**:

```python
# Pseudocódigo de etiquetado neto
def calcular_etiqueta_neta(episodio):
    r_bruto = calcular_r_bruto(episodio)          # resultado en R sin costos
    n_funding = episodio.duracion_horas // 8
    c_funding = n_funding * tasa_funding_media
    c_comision = comision_maker + comision_market  # según tipo de orden
    c_slippage = slippage_estimado                 # medido o supuesto
    c_total = c_comision + c_slippage + c_funding
    c_r = c_total / episodio.distancia_stop_pct
    return r_bruto - c_r  # R neto
```

---

## 7. Presupuesto de operaciones

Con múltiples activos y estilos activos simultáneamente puede haber decenas de señales
por semana. Para gestionar el riesgo global y la correlación:

- **Presupuesto semanal:** top-N señales por ventaja neta esperada
  ($E[R_{\text{neto}}]$), con N como parámetro (tentativo: 3–5).
- **Límite de exposición:** no más del X % del capital en posiciones correlacionadas
  con BTC (módulo 09).
- **Exposición bruta máxima:** 100 % del capital (sin pirámide inicial).

---

## 8. Hipótesis vinculadas

- **H2:** el filtro de ejecución por costo mejora el resultado neto fuera de muestra
  frente a operar todas las señales con $p > p^*_{\text{sin costo}}$.
- **H4:** la hora UTC (picos 16–17 h) y la cercanía al funding mejoran la ejecución
  (menor slippage, mejor timing de funding).

---

## 9. Riesgos

1. **Slippage no medido:** el 0.30 % es un supuesto conservador. Con datos reales del
   libro de órdenes puede ser menor en BTC/ETH y mayor en altcoins pequeñas.
2. **Funding en manías:** 0.01 % base pero hasta 0.1 % en extremos. El sistema debe
   usar tasas históricas reales, no solo la base.
3. **Cambios de tarifa de Binance:** verificar trimestralmente en la página oficial.
4. **Liquidación:** con apalancamiento, una caída del precio igual al margen inicial
   liquida la posición. El stop debe estar siempre por encima del precio de liquidación.
