# Módulo 14 — Diario de Casos y Admisión de Monedas (V3)

**Función:** almacenar todos los episodios históricos resueltos como una base de casos
reales indexada para retrieval y aprendizaje curricular, y protocolizar la admisión
estricta de nuevos pares en futuros perpetuos solicitados por el usuario.

> **Cambios V3:** adaptación del esquema de episodios para incluir métricas de futuros
> perpetuos (apalancamiento, precio de liquidación, tasa de funding acumulada, comisiones
> maker/taker), indexación por estilo de operación (táctico 1H, swing 4H, posicional 1D)
> y criterios de admisión que verifican liquidez en el libro de futuros y riesgo de funding.

---

## A. Diario de casos reales indexados

Cada episodio resuelto o censurado por el simulador de eventos (módulo 12) se archiva
como un registro inmutable estructurado:

### A.1 Esquema del caso en V3

```yaml
caso_id: ep_v3_001482
activo: ETH/USDT
mercado: Binance USDT-M Futures
estilo: tactical_1h
fecha_decision: "2024-04-18T16:00:00Z"
horizonte_max_horas: 24
apalancamiento: 4x

experto: breakout
tecnica: donchian_squeeze
escenario_id: 12
regimen_hmm: 0  # alcista
volatilidad_percentil: 0.42
embedding_z: [...]

plan:
  precio_entrada: 3080.50
  stop_invalidacion: 3042.00   # 1.25% de distancia (> stop_min)
  objetivo_salida: 3180.00     # R = 2.58
  precio_liquidacion: 2315.00  # muy seguro respecto al stop

costos_asociados:
  comision_entrada_maker: 0.020%
  comision_salida_taker: 0.040%
  slippage_observado: 0.08%
  funding_acumulado: +0.015%  # 2 pagos de 8h
  costo_total_R: 0.124R

desenlace:
  operado: true
  resultado_R_bruto: +2.58R
  resultado_R_neto: +2.456R
  duracion_horas: 17
  evento_cierre: E2_toque_objetivo
  mae_R: -0.35R
  mfe_R: +2.62R
  motivo_fallo: null

contrafactuales:
  sin_filtro_costo: true
  sin_trailing: false
```

### A.2 Usos del diario en V3

1. **Curriculum learning (módulo 05 §5):** entrenamiento de especialistas en dos fases:
   primero casos claros ($|R| > 1.5$) y luego casos ambiguos marginales.
2. **Memoria de retrieval para el operador:** búsqueda por similitud de $z_t$ para
   recuperar los 5 casos pasados más análogos antes de ejecutar una operación.
3. **Taxonomía de errores en futuros:**
   `falsa_ruptura` · `spike_volatilidad_funding` · `stop_demasiado_corto` ·
   `slippage_excesivo` · `cambio_regimen_hmm` · `cierre_por_tiempo_e4`.

---

## B. Admisión de nuevas criptomonedas (Futuros Perpetuos)

El usuario puede proponer cualquier par de Binance Futures. El sistema no opera
a ciegas; ejecuta una evaluación rigurosa de admisión antes de permitir señales:

| Criterio | Requisito mínimo | Razón técnica en V3 |
|---|---|---|
| **Historial en 1H y 4H** | ≥ 1 año continuo sin gaps críticos | Necesario para calibrar features y CUSUM |
| **Liquidez en Futuros** | Volumen diario > 15M USDT; spread medio < 3 bps | Evita que el slippage sobrepase el stop mínimo |
| **Estabilidad de Funding** | Historial sin tasas > 0.15 % frecuentes | Mitiga el riesgo de sangría por costo de acarreo |
| **Soporte en espacio $z_t$** | Score OOD en el percentil < 95 | La estructura del token debe parecerse a lo entrenado |
| **Profundidad de libro** | Profundidad a 1 % > 100k USDT | Garantiza órdenes limit sin deslizamiento severo |

### Dictamen de admisión:
- **`Admitida completa`:** apta para los 3 estilos de operación.
- **`Admitida solo Swing (4H/1D)`:** liquidez suficiente para swing, pero spread alto para táctico 1H.
- **`Rechazada`:** par ilíquido o atípico; el sistema se abstiene por protección de capital.
