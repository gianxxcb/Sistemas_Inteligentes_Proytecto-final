# Módulo 11 — Redes Neuronales en QuantLab ML (V3)

**Función:** compendio unificado de todas las arquitecturas de Deep Learning del sistema,
su justificación científica, orden de activación y criterios de descarte frente a líneas
base más simples.

> **Regla de oro V3 (H3 y H6):** ninguna red neuronal se asume superior por defecto.
> En datos tabulares y financieros ruidosos con series cortas, modelos simples (promedio,
> Hedge, XGBoost) con frecuencia baten a redes profundas. Las redes se prueban como
> alternativas que deben ganarse su inclusión con significancia estadística.

---

## 1. Mapa de modelos neuronales

```
Ventana multitemporal X_t (24 barras 1H / 30 barras 4H / 60 barras 1D)
       │
       ├─► [N1] TCN / Convolución Causal ──┐
       │                                   ├─► [N2] Autoencoder Temporal ─► z_t (embedding)
       │                                                                       │
       │                                                                       ▼
       │                                                               Scenario Engine
       │                                                               (Clusters + HMM)
       │                                                                       │
       ├─► [N4] Especialistas V3 (Opcional si H3 refuta XGBoost)                │
       │              (TCN / GRU con triple barrera de playbook)               │
       │                                   │                                   │
       │                                   ▼                                   │
       │                         [N5] Orquestador Nivel 2 ◄────────────────────┘
       │                               (Gating MLP - solo si H6 refuta Nivel 0/1)
       │                                   │
       │                                   ▼
       └────────────────────► [N6] Forecast Engine (Regresión Cuantílica)
```

---

## 2. Catálogo de modelos y prioridades V3

| ID | Red | Módulo | Función | Condición de activación |
|---|---|---|---|---|
| **N1** | TCN Causal 1D | 03 | Extracción de features locales temporales causales | Base para autoencoder |
| **N2** | Autoencoder temporal | 03 | Compresión no supervisada en $z_t$ | Reconstrucción sin fuga de activo |
| **N4** | Especialistas V3 (TCN/GRU) | 05 | Versión neuronal temporal de los 5 expertos | Comparación purgada con HistGradientBoosting y referencia nula en activos no vistos; diagnóstico hasta validación completa (**H3**) |
| **N5** | Gating neuronal (Nivel 2) | 06 | Ponderación dinámica de expertos | Solo si supera a Nivel 0 (Hedge) y Nivel 1 en CPCV (**H6**) |
| **N6** | Forecast cuantílico | 07 | Red de cuantiles P10–P90 y probabilidades | Calibrada en validación sin cuantiles cruzados |

---

## 3. Especificación detallada

### N1 & N2 — TCN y Autoencoder de Representación
- Redes convolucionales causales con dilataciones crecientes (1, 2, 4, 8) para respetar
  la flecha del tiempo.
- Pérdida de reconstrucción MSE pura.
- Evaluación mediante prueba de adversario: un clasificador lineal no debe ser capaz de
  predecir qué criptomoneda generó el vector $z_t$.

### N4 — Especialistas V3 (H3)
- Entrada: ventanas causales de features del experto, terminadas en la barra de decisión.
- Comparación obligatoria en folds agrupados por activo y separados cronológicamente:
  HistGradientBoosting tabular, TCN, GRU y una referencia nula basada en la prevalencia y
  los cuantiles del train. Los activos reservados se mantienen fuera de todos los folds.
- Multi-head:
  - Cabeza 1: probabilidad calibrada de tocar objetivo antes del stop ($p_{\text{éxito}}$).
  - Cabeza 2: cuantiles de retorno condicional.
  - Cabeza 3: aplicabilidad técnica (0 a 1).
- Pérdida implementada: $\mathcal{L} = \text{BCE ponderada} + \text{Pinball ponderada}$;
  las cabezas P10/P50/P90 no se cruzan. La aplicabilidad sigue siendo un criterio de
  validación independiente y no se simula como una etiqueta aprendida.
- **TCN/GRU no se seleccionan por defecto**: deben superar al baseline y a la referencia
  nula con suficientes episodios, calibración y generalización entre activos.

### N5 — Gating Neuronal de Orquestación (H6)
- MLP compacto (2 capas, 64 unidades, Dropout 0.3).
- Entrada: $z_t$, estado HMM, percentil de volatilidad, contratos de salida de los 5 expertos.
- Salida: pesos softmax por experto + probabilidad de abstención.
- Regularización con penalización de entropía para evitar el colapso (que un solo experto
  acapare siempre el 100 % del peso).
- **Si el promedio simple o Hedge (Nivel 0) empata estadísticamente en CPCV, se descarta N5.**

---

## 4. Prácticas de entrenamiento en PyTorch

- Optimizador: `AdamW` con weight decay 1e-4 y scheduler `ReduceLROnPlateau`.
- Lotes estratificados y balanceados temporalmente.
- Early stopping estricto monitoreando la pérdida de validación en el bloque purgado.
- Semillas fijadas para reproducibilidad completa en los experimentos.
