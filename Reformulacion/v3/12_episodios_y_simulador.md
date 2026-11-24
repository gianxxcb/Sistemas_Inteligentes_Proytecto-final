# Módulo 12 — Episodios y Simulador de Paso del Tiempo (V3)

**Función:** ejecutar el bucle de aprendizaje de los expertos por eventos, con
temporalidades múltiples, resolución intravela con barra inferior, y muestreo CUSUM.

> **Cambios V3:** reemplaza el bucle diario por el bucle por eventos del módulo 16,
> añade gestión en temporalidad fina (15m/1H/4H), resolución intravela con barra
> inferior, episodios tácticos (horarios), y pesos por unicidad.

---

## 1. El episodio V3: cuatro fases con temporalidades múltiples

| Fase | Qué ocurre | Temporalidad |
|---|---|---|
| **1. Formación** | El experto recibe la ventana en la temporalidad de decisión y dibuja la estructura | 1H / 4H / 1D según estilo |
| **2. Plan** | Emite el plan condicional completo con niveles en la temporalidad de gestión | Niveles expresados en precio |
| **3. Replay por eventos** | El simulador revela velas de gestión; el experto actúa solo ante E1–E4 | 15m / 1H / 4H según estilo |
| **4. Resultado** | R neto tras costos (comisión + slippage + funding acumulado), MAE, MFE, duración en horas | — |

---

## 2. Reglas del replay por eventos (actualizado V3)

1. **Un evento a la vez** (módulo 16 §2–§5). El simulador no revela velas hasta que el
   experto haya procesado el evento anterior.
2. **E2 tiene prioridad:** si una vela de gestión toca stop o objetivo, se resuelve antes
   de cualquier otro evento.
3. **Resolución intravela** con barra inferior (módulo 16 §4): si high y low tocan ambos
   niveles, recorrer las barras inferiores en orden temporal.
4. **Funding acumulado:** en cada vela de gestión que cruza una hora de liquidación
   (00:00, 08:00, 16:00 UTC), se suma el funding de esa liquidación al costo acumulado.
5. **Alerta de funding extremo:** si la tasa supera 0.05 % en la última liquidación, el
   experto recibe una señal en su estado para considerar cierre.
6. **Sin solapamiento de episodios:** misma regla que V2 (ponderación por unicidad si
   hay solapamiento en el entrenamiento).
7. **Censura al final de los datos:** igual que V2.

---

## 3. El plan condicional V3 (actualizado para perpetuos)

```yaml
episodio_id: ep_000312
activo: SOL/USDT (perpetuo)
contrato: USDT-M Futures
estilo: swing_corto
experto: breakout
tecnica: triangulo_simetrico
fecha_decision: 2024-03-15T16:00:00Z  # cierre de vela 4H
horizonte_max: 7_dias
apalancamiento: 3x
estructura:
  linea_superior: {pendiente: -0.6%/4H, nivel_hoy: 142.0}
  linea_inferior: {pendiente: +0.2%/4H, nivel_hoy: 128.0}
entrada: {tipo: apertura_siguiente_vela_1H, direccion: largo}
stop: 126.0  # nivel estructural; distancia: 1.56%
objetivo: 148.0  # extensión del triángulo; R=3.8
tiempo_maximo: 2024-03-22T16:00:00Z
precio_liquidacion: 128.0 × (1 - 1/3) ≈ 85.3  # con 3x, muy por debajo del stop

costos:
  comision_ida: 0.020%  # maker (limit)
  comision_salida: 0.040%  # taker (market stop)
  slippage_estimado: 0.30%
  funding_estimado: 9_pagos × 0.01% = 0.09%  # para 3 días
  c_total: 0.410%
  c_R: 0.263R  # = 0.41% / 1.56%

p_exito_calibrada: 0.38
p_equilibrio_neto: (1 + 0.263) / (3.8 + 1) = 0.264  # p* neto
filtro_ok: true  # 0.38 > 0.264 y distancia stop 1.56% > stop_min 1.0%
e_neto_esperado: 0.38 × 3.8 - 0.62 - 0.263 = 0.581R

acciones_condicionales:
  si_rompe_superior: trailing_stop 1.5×ATR(14) | tomar_50%
  si_rompe_inferior: salir_inmediato (stop)
  E3_volatilidad: revisar_plan_con_datos_confirmados
  E4_tiempo: cerrar_por_tiempo
```

---

## 4. Contrafactuales y candidatos filtrados (igual que V2 + extensión)

Se registran todos los candidatos, incluyendo los filtrados por costo:

```python
candidato = {
    "tipo": "filtrado_por_costo",  # o "filtrado_por_stop_min" o "operado"
    "motivo_filtro": "distancia_stop < stop_min_pct",
    "resultado_hipotetico": calcular_r_bruto(episodio),  # sin el filtro
    "costo_hipotetico": calcular_costo(episodio),
    "resultado_neto_hipotetico": resultado_hipotetico - costo_hipotetico,
}
```

Esto permite evaluar H2 (¿el filtro mejora la expectativa neta?) con datos reales de
lo que habría pasado sin el filtro.

---

## 5. Bucle del simulador V3 (pseudocódigo actualizado)

```python
def simular_episodio_v3(plan, datos_gestion, datos_decision, costos):
    # Verificar filtro de ejecución
    if not filtro_ejecucion_ok(plan, costos):
        registrar_candidato_filtrado(plan, datos_decision, costos)
        return

    estado = abrir_posicion(plan, apertura_siguiente(datos_decision))
    cola_eventos = construir_cola_cusum(datos_gestion, plan)  # E1, E3

    funding_acumulado = 0.0

    for evento in cola_eventos:
        # Acumular funding en cada liquidación
        funding_acumulado += calcular_funding_en_periodo(evento, estado)
        if funding_acumulado > 0.05 * plan.objetivo_pct:
            estado.alerta_funding = True

        # E2: verificar toque de nivel
        if toca_nivel(evento.vela, plan.stop, plan.objetivo):
            estado = resolver_intravela(
                evento.vela, plan,
                datos_barra_inferior=obtener_barra_inferior(evento)
            )
            break

        # E1 / E3: revisar plan
        if evento.tipo in (E1, E3):
            contexto = {
                "datos_hasta": datos_hasta(evento.fecha),
                "estado": estado,
                "plan": plan,
                "funding_acumulado": funding_acumulado,
            }
            accion = experto.revisar(contexto)
            estado = aplicar_accion(accion, estado)

        # E4: tiempo máximo
        if evento.fecha >= plan.tiempo_maximo:
            estado = cerrar_por_tiempo(estado)
            break

    resultado = calcular_r_neto(estado, costos, funding_acumulado)
    diario.registrar(episodio, plan, estado, resultado)
    return resultado
```

---

## 6. Métricas del episodio V3 (actualizada)

- **R neto realizado** (tras comisión, slippage y funding).
- **R bruto** (sin costos, para medir el valor de las señales independientemente del costo).
- **Días/horas hasta resolución** y motivo de cierre (objetivo, stop, tiempo, revisión).
- **MAE y MFE** en precio y en R.
- **Funding acumulado** pagado/cobrado.
- **Distancia al precio de liquidación** al momento del cierre (medida de seguridad).
- **Expectativa por estilo/experto/escenario:** $\overline{R}_{\text{neto}}$, hit-rate,
  profit factor neto, con IC bootstrap por bloques.

---

## 7. Estado de implementación

- El bucle diario de V2 (`src/simulator/episode_engine.py`) se extiende con el
  despacho por eventos (E1–E4).
- El cálculo de funding acumulado se añade como componente del `EpisodeState`.
- La resolución intravela con barra inferior se activa cuando los datos de 15m/1H
  están disponibles; si no, aplica el caso peor.
- Los candidatos filtrados por costo se registran en el diario con sus resultados
  hipotéticos (contrafactuales de H2).
- Los episodios tácticos (1H) usan el mismo motor pero con diferentes parámetros de
  temporalidad.
