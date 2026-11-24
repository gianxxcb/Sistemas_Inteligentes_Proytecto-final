# Módulo 16 — Paso del Tiempo por Eventos (NUEVO en V3)

**Función:** reemplazar el reloj fijo (una vela diaria = un paso) por un sistema
orientado a eventos que activa a los expertos solo cuando ocurre algo relevante,
adaptando la frecuencia de observación a la temporalidad de cada estilo.

---

## 1. El problema del reloj fijo

Un sistema que procesa una barra por período muestrea de forma **uniforme**:
- **Exceso** en períodos de baja actividad (una barra de 4H con movimiento < 0.1 ATR
  no aporta información nueva).
- **Defecto** en períodos de alta actividad (una ruptura de volatilidad puede
  resolverse en 15 minutos, pero el sistema espera hasta la barra siguiente).

Además, con horizontes en horas (estilo Táctico), procesar todas las velas de 15m de
un episodio de 24h genera 96 pasos con ruido, la mayoría irrelevantes.

---

## 2. Los cuatro eventos de despertar

Un experto "despierta" (recibe información y puede actuar) solo ante uno de estos
cuatro eventos:

| Evento | Descripción | Temporalidad de gestión |
|---|---|---|
| **E1 — Cierre de vela de gestión** | Cierra una vela en la temporalidad de gestión del estilo | 15m (táctico), 1H (swing corto), 4H (swing/posicional) |
| **E2 — Toque de nivel** | El precio toca o cruza stop, objetivo o un nivel clave del plan | Inmediato (tick o barra mínima disponible) |
| **E3 — Pico de volatilidad** | Rango de la vela de gestión supera *k* × ATR(20) del estilo | Mismo que E1 |
| **E4 — Tiempo máximo** | Se cumple el horizonte máximo del episodio | Al cierre de la barra correspondiente |

Solo E2 y E4 cierran el episodio. E1 y E3 lo revisan.

---

## 3. Temporalidades por estilo

| Estilo | Temporalidad de decisión | Temporalidad de gestión | Horizonte máx. |
|---|---|---|---|
| Táctico | 1H | 15m | 6–24 h |
| Swing corto | 4H | 1H | 2–7 días |
| Swing/posicional | 1D + disparador 4H | 4H (1H cerca de niveles clave) | 10–30 días |

**Flujo de datos requerido (módulo 01):**
- Táctico: velas de 1H y 15m.
- Swing corto: velas de 4H y 1H.
- Swing/posicional: velas de 1D, 4H y opcionalmente 1H.

---

## 4. Resolución de ambigüedad intravela

Cuando stop y objetivo caen dentro de la misma vela de gestión (tanto high como low
tocan los niveles), no se sabe cuál ocurrió primero.

**Protocolo de resolución (en orden de preferencia):**

1. **Descargar velas de temporalidad inferior:** si el episodio es de gestión en 1H,
   recorrer las velas de 15m dentro de esa barra en orden temporal para determinar cuál
   nivel se tocó primero.
2. **Si no hay temporalidad inferior disponible:** aplicar el **criterio conservador**
   (caso peor): asumir que el stop se tocó primero.
3. **Registrar** siempre el motivo de resolución en el diario de casos (módulo 14).

---

## 5. Bucle del simulador orientado a eventos

```python
# Pseudocódigo V3 — simulador por eventos
def simular_episodio(plan, datos_gestion, datos_decision):
    estado = abrir_posicion(plan, apertura_siguiente(datos_decision))
    cola = construir_cola_eventos(datos_gestion, plan)  # E1, E3 de cada barra

    for evento in cola:
        # E2: verificar toque de nivel en cada barra de gestión
        if toca_nivel(evento.vela, plan.stop, plan.objetivo):
            estado = resolver_intravela(evento.vela, plan, datos_temporalidad_inferior)
            break

        # E1 / E3: cierre de vela de gestión o pico de volatilidad
        if evento.tipo in (E1, E3):
            accion = experto.revisar(datos_hasta(evento.fecha), estado, plan)
            estado = aplicar_accion(accion, estado)

        # E4: tiempo máximo
        if evento.fecha >= plan.fecha_decision + plan.horizonte_max:
            estado = cerrar_por_tiempo(estado)
            break

    resultado = calcular_r_neto(estado, plan.costos)
    diario.registrar(episodio, plan, estado, resultado)
    return resultado
```

---

## 6. Muestreo CUSUM para detección de eventos candidatos

En lugar de evaluar el sistema en cada barra (todos los cierres de 1H o 4H), los
candidatos a operación se detectan mediante el **filtro CUSUM simétrico** de López de
Prado:

$$S_t^+ = \max(0, S_{t-1}^+ + r_t - E[r]) \quad S_t^- = \min(0, S_{t-1}^- + r_t - E[r])$$

Se genera un evento cuando $S_t^+ \geq h$ o $S_t^- \leq -h$, donde $h$ es el umbral
(parámetro del sistema, en unidades de ATR o de retorno).

**Ventajas del CUSUM:**
- Muestrea más frecuentemente en períodos de alta actividad (donde hay señal).
- Reduce drásticamente el número de observaciones redundantes.
- Mejora la independencia entre episodios (menos solapamiento).

**Parámetro $h$:** se configura como múltiplo del ATR relativo y un límite superior
por estilo: 2,0× (táctico, cap 8 %), 2,4× (swing corto, cap 8 %) y 3,0× (posicional,
cap 25 %). Son valores candidatos que deben comprobarse en train/calibración; no se
ajustan usando test.

---

## 7. Pesos por unicidad

Cuando dos episodios se solapan en el tiempo (sus ventanas de formación o sus períodos
de gestión se superponen), no son estadísticamente independientes. Se corrige con
pesos por **unicidad** (López de Prado):

$$w_i = \frac{1}{c_i}$$

donde $c_i$ es el número de episodios concurrentes en el instante $i$. El promedio de
estos pesos sobre la duración del episodio da su peso en el entrenamiento.
Para el entrenamiento de especialistas, la concurrencia se calcula dentro del mismo
activo y experto: al ajustarse modelos separados, un episodio de otro experto no debe
reducir la unicidad de esta muestra.

**Unicidad media esperada:** con episodios de gestión en 1H y muestreo CUSUM, la
unicidad media debería superar el 0.6 (en V2 con episodios diarios era 0.435).

---

## 8. Picos horarios de actividad (hora UTC)

La investigación muestra que la actividad, volatilidad e iliquidez tienen picos entre
las **16:00 y 17:00 UTC** en Binance y otros exchanges.

Implicaciones para el sistema:

1. **Ejecución táctica:** preferir entrada/salida fuera del pico (antes de las 16:00 o
   después de las 17:00) para reducir slippage, **salvo que la señal sea específica del
   pico**.
2. **Features de hora:** `hour_sin`, `hour_cos` (codificación cíclica de la hora UTC)
   como feature para los modelos de los expertos tácticos.
3. **Cercanía al funding:** feature binaria `cerca_funding` (ventana de ±30 min antes
   de 00:00, 08:00 y 16:00 UTC) como factor de riesgo de ejecución.
4. **No operar en funding spike:** si la tasa de funding histórica del activo supera el
   0.05 % en la última liquidación, aumentar el margen de seguridad del filtro de
   ejecución.

---

## 9. Integración con módulos existentes

| Módulo | Relación V3 |
|---|---|
| 01 Data Engine | Provee velas 15m, 1H, 4H y datos de funding histórico por horario |
| 05 Specialist Models | Usan el bucle por eventos para formación, plan y replay |
| 12 Episodios | El simulador V3 reemplaza el bucle diario por el bucle por eventos |
| 15 Modelo de Costos | El evento E2 (toque de nivel) desencadena el cálculo de costo de funding acumulado |
| 08 Validación | La purga se calcula en horas, no en días, para episodios tácticos |

---

## 10. Riesgos

1. **Latencia de datos:** para el estilo táctico, los datos de 15m deben estar
   disponibles en tiempo real. En backtesting histórico esto no es un problema; en uso
   real sí.
2. **Picos de volatilidad nocturnos:** entre las 00:00 y 06:00 UTC la liquidez es
   menor; los triggers de E3 pueden generar falsas señales.
3. **Desfase de funding:** si el sistema entra justo antes del funding y el activo está
   alcista, el costo de funding puede ser significativamente mayor que la tasa base.
4. **Ambigüedad intravela sin datos inferiores:** para activos con poca historia en 15m
   el criterio conservador subestima la rentabilidad real.
