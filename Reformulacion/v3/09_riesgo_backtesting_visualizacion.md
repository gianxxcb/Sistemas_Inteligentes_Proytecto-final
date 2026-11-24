# Módulo 09 — Riesgo, Backtesting y Visualización (V3)

**Función:** gestionar el riesgo de las posiciones con apalancamiento en futuros
perpetuos, realizar el backtesting realista y visualizar los planes de operación.

> **Cambios V3:** gestión de margen y liquidación para futuros perpetuos, funding
> acumulado como componente de riesgo, stop mínimo en R, presupuesto de operaciones,
> riesgo de correlación entre activos simultáneos.

---

## 1. Gestión de riesgo con futuros perpetuos y apalancamiento

### 1.1 Margen e isolamiento

- **Margen aislado por posición** (recomendado): cada operación usa solo el margen
  asignado, sin riesgo de que una pérdida afecte otras posiciones.
- **Precio de liquidación:** con apalancamiento $L$, el precio de liquidación está
  aproximadamente a $1/L$ del precio de entrada (menos el margen de mantenimiento).

$$P_{\text{liquidación}} \approx P_{\text{entrada}} \times \left(1 - \frac{1}{L} + \text{margen\_mantenimiento}\right)$$

- **Regla crítica:** el stop del playbook debe estar **por encima del precio de liquidación**
  con un margen de seguridad (tentativo: 10 % del margen de mantenimiento).

### 1.2 Apalancamiento recomendado por estilo

| Estilo | Apalancamiento tentativo | Razón |
|---|---|---|
| Táctico | 2–5× | Movimientos pequeños en horas; stops cercanos |
| Swing corto | 2–3× | Balance entre amplificación y riesgo de liquidación |
| Swing/posicional | 1–2× | Horizonte largo; funding acumulado significativo |

> El apalancamiento es un parámetro del sistema, no una decisión por operación.
> Se calibra para que el riesgo por operación sea siempre el 1 % del capital
> (gestión fija de riesgo en R).

### 1.3 Riesgo por operación

El tamaño de posición se calcula para que una pérdida de 1R equivalga al **1 % del
capital** (parámetro configurable):

$$\text{tamaño\_posición} = \frac{\text{capital} \times 0.01}{\text{distancia\_stop} \times L}$$

---

## 2. Funding como componente de riesgo

El funding en perpetuos no es solo un costo: es también un **riesgo de timing**:

- Si el activo está en manía alcista, el funding puede superar el 0.1 % cada 8 horas
  (>100 % anual). Una posición larga abierta en ese momento paga tasas extremas.
- **Control:** el filtro de ejecución (módulo 15 §5) bloquea operaciones cuando el
  funding estimado supera el 30 % del objetivo esperado.
- **Monitoreo continuo:** durante el replay de un episodio, si la tasa de funding supera
  el 0.05 %, el experto recibe una señal de alerta para considerar el cierre anticipado.

---

## 3. Stop mínimo en R

El stop mínimo (módulo 15 §4) se expresa también en términos de riesgo:

$$\text{stop\_min\_R} = \frac{\text{stop\_min\_pct}}{\text{precio\_entrada}} \times \frac{1}{c_R}$$

Una operación con stop mínimo de 1 % y costo total de 0.39 % tiene $c_R = 0.39$ R;
el stop mínimo en R es el que hace que $p^* < 0.50$ (esperanza teóricamente alcanzable).

---

## 4. Gestión de riesgo de cartera

### 4.1 Límites de concentración

- **Máximo por activo:** 10 % del capital en nocional.
- **Máximo correlacionado con BTC:** no más del 40 % del capital en activos con
  correlación >0.7 con BTC en los últimos 20 días.
- **Exposición bruta máxima:** 100 % del capital (sin pirámide al inicio).

### 4.2 Riesgo de correlación entre episodios simultáneos

Si dos episodios están abiertos simultáneamente en activos correlacionados, su riesgo
combinado no es la suma de sus riesgos individuales. El orquestador (módulo 06 §7)
descuenta señales redundantes usando la correlación móvil de retornos.

---

## 5. Motor de backtesting realista (actualizado V3)

El motor de replay del módulo 12 incluye ahora:

| Componente | V2 | V3 |
|---|---|---|
| Temporalidad | Diaria | Por eventos (15m/1H/4H según estilo) |
| Costos | Fijos (14 bps) | Variables: comisión (maker/taker) + slippage + funding |
| Ambigüedad intravela | Caso peor (stop primero) | Resolución con barra inferior o caso peor |
| Margen y liquidación | No modelado | Precio de liquidación calculado; alerta si stop > liquidación |
| Funding | No modelado | Acumulado por episodio según duración real |
| Presupuesto | Sin límite | Top-N por ventaja neta esperada |

---

## 6. Métricas de riesgo (actualizada V3)

| Métrica | Descripción |
|---|---|
| **R realizado medio** | Expectativa neta en R, tras costos y funding |
| **MAE / MFE** | Excursión máxima adversa / favorable por episodio |
| **Drawdown realizado** | Sobre equity cerrada (no mark-to-market) |
| **Profit factor neto** | Suma de ganancias netas / suma de pérdidas netas |
| **Duración media** | Días/horas hasta resolución por estilo |
| **Funding acumulado** | Total de funding pagado/cobrado en el período |
| **Sharpe deflactado** | Ajustado por número de pruebas y asimetría de retornos |
| **Tasa de liquidaciones** | Número de episodios liquidados por margen / total (debe ser 0 en backtesting correcto) |

---

## 7. Visualización del plan de operación

Para cada episodio se genera un gráfico con:

- **Precio** en la temporalidad de gestión.
- **Niveles del plan:** entrada, stop (invalidación), objetivo.
- **Precio de liquidación** (línea de peligro, en rojo).
- **Eventos de despertar** (E1–E4) marcados sobre el gráfico.
- **Funding pagado/cobrado** acumulado (eje secundario).
- **R neto realizado** vs R esperado.
- **MAE/MFE** con flechas de excursión.

---

## 8. Ensayo en vivo y modo uso real (igual que V2)

- **Modo prueba:** todos los episodios del período (incluyendo cerrados en el pasado).
- **Modo uso real:** solo el último episodio abierto de cada activo/estilo.
- El ensayo en vivo no consume el test final 2025; usa solo datos posteriores a la
  fecha de corte.

---

## 9. Riesgos

1. **Riesgo de liquidación en mercados volátiles:** un gap nocturno puede liquidar una
   posición antes de que el stop se active. Mitigar con apalancamiento bajo y stops
   más amplios.
2. **Funding extremo en manías:** puede convertir una operación ganadora en perdedora.
   El monitoreo continuo durante el replay es esencial.
3. **Correlación dinámica:** en caídas de mercado, activos normalmente descorrelacionados
   se correlacionan; los límites de concentración se tornan insuficientes.
4. **Drawdown mark-to-market vs realizado:** el sistema reporta drawdown realizado; el
   drawdown intratrade (en episodios abiertos) puede ser mayor y no se muestra en la
   curva de equity.
