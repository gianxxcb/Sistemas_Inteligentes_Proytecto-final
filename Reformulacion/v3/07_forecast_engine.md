# Módulo 07 — Probabilistic Forecast Engine (V3)

**Función:** producir distribuciones de resultados por estilo y horizonte, con costos
integrados y filtro de ejecución incorporado.

> **Cambios V3:** horizontes en horas y días (tres estilos, no 7/14/30/90d), magnitud
> y volatilidad como objetivos primarios, filtro de ejecución integrado, horizonte de
> 90d eliminado como no confirmado.

---

## 1. Horizontes por estilo (nuevo V3)

| Estilo | Horizonte objetivo | Horizonte máximo | Episodios/moneda/año (aprox.) |
|---|---|---|---|
| **Táctico** | 6–12 h | 24 h | ~4 400 (a 12 h) |
| **Swing corto** | 2–4 días | 7 días | ~365–730 |
| **Swing/posicional** | 10–20 días | 30 días | ~73–183 |

Los tres estilos coexisten: pueden generar predicciones simultáneas sobre el mismo
activo con horizontes diferentes y sin contradicción (rebote táctico dentro de tendencia
bajista swing).

**Por qué se eliminan 90 días:** solo ~24 episodios independientes por moneda por año.
La evidencia de V2 no confirmó ventaja en 30 ni 90 días. Los estilos posicionales
tienen ahora un techo de 30 días.

**Por qué se añade el estilo táctico (horas):** el ritmo real del operador (1 op/día)
encaja mejor con horizontes de horas a días. Los episodios independientes son
sustancialmente más numerosos, lo que mejora la potencia estadística.

---

## 2. Objetivos primarios: magnitud y volatilidad (H1)

La literatura indica que la magnitud/volatilidad es más predecible que la dirección a
frecuencia horaria y diaria (H1). Por eso el Forecast Engine entrega:

| Salida | Prioridad | Descripción |
|---|---|---|
| **Magnitud esperada** | **Alta** | Rango esperado / ATR(14) normalizado |
| **Volatilidad esperada** | **Alta** | Desviación estándar del retorno en el horizonte |
| P(retorno positivo) | Media | Probabilidad de dirección alcista |
| P(retorno negativo) | Media | Probabilidad de dirección bajista |
| P(lateral) | Media | Umbral dependiente de volatilidad y costos |
| Cuantiles P10–P90 | Alta | Distribución completa del retorno |
| Incertidumbre | Media | Aleatoria vs epistémica |

---

## 3. Filtro de ejecución integrado (nuevo V3)

Antes de emitir una señal operativa, el Forecast Engine aplica el filtro del módulo 15 §5:

```python
def emitir_senal(prediccion, plan, costos):
    p_exito = prediccion.p_exito
    p_star = costos.p_equilibrio_neto(plan.R, plan.distancia_stop)
    e_neto = p_exito * plan.R - (1 - p_exito) - costos.c_R

    if p_exito <= p_star:
        return Abstencion("esperanza neta negativa")
    if plan.distancia_stop < costos.stop_min_pct:
        return Abstencion("stop demasiado cerca")
    if e_neto <= DELTA_MINIMO:
        return Abstencion("margen de seguridad insuficiente")
    if costos.funding_estimado > 0.3 * plan.objetivo_pct:
        return Abstencion("costo de funding excesivo")

    return Senal(prediccion, plan, e_neto=e_neto)
```

Las abstenciones se registran como contrafactuales (módulo 12 §5) para aprender de
ellas.

---

## 4. Escalado por volatilidad (igual que V2)

Los umbrales de "lateral" y los rangos se expresan como múltiplos de $\sigma\sqrt{h}$:

- **Lateral (táctico):** |retorno| < $k \times \sigma_{\text{hora}} \times \sqrt{12}$
  (para horizonte de 12 h).
- **Lateral (swing corto):** |retorno| < $k \times \sigma_{\text{día}} \times \sqrt{3}$
  (para horizonte de 3 días).
- **Lateral (posicional):** |retorno| < $k \times \sigma_{\text{día}} \times \sqrt{15}$
  (para horizonte de 15 días).

$k$ es un parámetro del sistema (tentativo: 0.5).

---

## 5. Salidas por estilo y horizonte

```
BTC/USDT · estilo: táctico · escenario: compresión · régimen: alcista · OOD: no
Horizonte 12h:  P(+)=0.55 · P(−)=0.35 · P(lateral)=0.10
                P10=-1.8%  P25=-0.8%  P50=+0.5%  P75=+1.5%  P90=+2.8%
                Magnitud esperada: 2.3% (1.4× ATR(14))
                Costo total estimado: 0.38%  |  E[R neto]: +0.34R
                Filtro ejecución: OK  |  Funding estimado: 0.01%
Expertos: Breakout 0.45 · Volatility 0.30 · Momentum 0.25
```

---

## 6. Regla de oro (igual que V2)

Los percentiles son **cuantiles de una distribución predictiva**, no objetivos
garantizados. Si el sistema no está calibrado, las probabilidades no se muestran como
válidas. Umbral mínimo: 30 observaciones y 6 bloques temporales para declarar una
salida como calibrada.

---

## 7. Hipótesis vinculadas

- **H1:** magnitud/volatilidad se predice mejor que dirección.
- **H2:** el filtro de ejecución mejora la expectativa neta.
- **H4:** la hora UTC y el funding mejoran las predicciones tácticas.
