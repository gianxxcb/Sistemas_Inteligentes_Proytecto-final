# Módulo 06 — Orquestador / Meta-Model (V3)

**Función:** combinar las señales de los expertos de forma robusta, con una jerarquía
de tres niveles que empieza simple y añade complejidad solo si esta supera
estadísticamente al nivel anterior.

> **Cambios V3:** arquitectura de tres niveles escalonados (simple → jerárquico →
> neuronal), incorporación de Hedge/Fixed Share como alternativa al promedio simple,
> contracción jerárquica con régimen HMM en el nivel 1, gating neuronal en el nivel 2
> solo si supera a los niveles anteriores con CPCV y Sharpe deflactado (H6).

---

## 1. Motivación del cambio arquitectónico

La evidencia empírica es clara:

- Las **combinaciones simples de pronósticos** superan a las ponderaciones sofisticadas
  cuando la muestra de entrenamiento es pequeña.
- La causa es el **error de estimación de los pesos**: con pocas muestras, el promedio
  simple es difícil de batir.
- Con 5 expertos, ~10 escenarios y 3 horizontes se obtienen ~150 celdas en la matriz
  de competencia, cada una con muy pocos episodios independientes.
- El gating neuronal de V2 no tenía datos suficientes para justificar su complejidad.

**Principio V3:** empezar con el modelo más simple que funcione y subir solo si los
datos lo justifican.

---

## 2. Arquitectura en tres niveles escalonados

```
Nivel 0 — Combinación simple
    │  Si supera la línea base incondicional con CPCV:
    ▼
Nivel 1 — Contracción jerárquica con régimen
    │  Si supera al nivel 0 con CPCV y Sharpe deflactado:
    ▼
Nivel 2 — Gating neuronal
    │  Solo si supera al nivel 1 con CPCV y Sharpe deflactado (H6)
    ▼
Forecast Engine (módulo 07)
```

**Regla de activación:** cada nivel se activa solo si supera al anterior en validación.
No se activa nunca retroactivamente (no se reoptimiza el nivel inferior si el superior
falla).

---

## 3. Nivel 0 — Combinación simple de expertos

### 3.0a Promedio simple

Peso igual a todos los expertos factibles (aquellos cuyo estado ≠ `inactivo`):

$$\hat{p}(t) = \frac{1}{|E_t|} \sum_{e \in E_t} p_e(t)$$

Es la línea base mínima. Si no supera al incondicional, el sistema no tiene valor.

### 3.0b Algoritmos de expertos con aprendizaje online

Alternativas al promedio simple que se adaptan sin entrenamiento offline:

**Hedge (multiplicative weights):**

$$w_e(t+1) = w_e(t) \cdot \eta^{-\ell_e(t)}$$

donde $\ell_e(t)$ es la pérdida del experto $e$ en el paso $t$ y $\eta \in (0,1)$ es
la tasa de aprendizaje. Garantiza que el peso del mejor experto histórico domina
eventualmente.

**Fixed Share:**

Extiende Hedge cediendo una fracción $\alpha$ del peso de cada experto a todos los
demás en cada paso. Esto permite seguir al mejor experto cuando este **cambia** con el
tiempo (deriva de régimen).

$$w_e(t+1) \propto (1-\alpha) \cdot w_e(t) \cdot \eta^{-\ell_e(t)} + \frac{\alpha}{|E|} \sum_{e'} w_{e'}(t) \cdot \eta^{-\ell_{e'}(t)}$$

**Parámetros:** $\eta$ y $\alpha$ tienen muy pocas opciones razonables; no hay
sobreajuste con una búsqueda en rejilla pequeña.

**Decisión de nivel 0:** probar promedio simple, Hedge y Fixed Share; reportar los tres
y usar el mejor en validación para alimentar el nivel 1.

---

## 4. Nivel 1 — Contracción jerárquica con régimen

### 4.1 Qué cambia respecto a V2

En V2, la matriz de competencia (capa A) tenía ~150 celdas con pocas muestras y
generó 0 celdas `factible` en F3 y F4. El problema era doble: pocos episodios por
celda y sin regularización suficiente.

En V3 se añade:

1. **Contracción jerárquica (empirical Bayes / shrinkage):** la estimación de cada
   celda (escenario *s*, experto *e*, horizonte *h*) se contrae hacia:
   - La media del experto en todos los escenarios.
   - La media del escenario con todos los expertos.
   - La media global.
   El grado de contracción depende del tamaño de muestra: con pocos episodios, la
   estimación se acerca a la media; con muchos, se aleja.

2. **Régimen HMM como contexto adicional:** en lugar de una única matriz de competencia,
   se tienen **tres matrices** (una por estado del HMM: alcista, neutro, bajista). La
   contracción jerárquica une las tres: cada celda dentro de un régimen se contrae
   hacia la celda equivalente en los otros regímenes.

3. **Percentil de volatilidad como segundo eje:** se añade como factor adicional de
   contracción, no como una dimensión completa (para no multiplicar el número de celdas).

### 4.2 Estados de celda (igual que V2)

| Estado | Criterio |
|---|---|
| `factible` | Ventaja significativa y estable fuera de muestra, con n suficiente |
| `no factible` | Sin ventaja o ventaja negativa |
| `desconocido` | Pocos episodios (mínimo n no alcanzado) |

La contracción jerárquica no cambia los estados; sí cambia el umbral efectivo de n
necesario para declarar `factible` (porque la estimación es más precisa).

### 4.3 Corrección por comparaciones múltiples

Con muchas celdas, se aplica FDR (Benjamini-Hochberg) como en V2.

---

## 5. Nivel 2 — Gating neuronal (solo si H6 se rechaza)

Igual que la Capa B de V2, con una condición de activación explícita:

**Condición de activación:** el nivel 2 solo se entrena y evalúa si el nivel 1 supera
al nivel 0 con p-valor < 0.05 (ajustado por FDR) en CPCV.

**Si se activa:**

- **Entradas:** contrato de salida de cada experto, escenario actual (z_t), régimen
  HMM, percentil de volatilidad, contexto del activo.
- **Salida:** pesos por experto (softmax) + abstención.
- **Arquitectura:** MLP pequeño (2–3 capas, 64–128 unidades) con regularización.
- **Pérdida:** log loss / pinball sobre salidas OOF de los expertos, más entropía de
  balance de carga.

---

## 6. Capa C — Arbitraje de conflictos y abstención (sin cambios)

Ver módulo 06 V2 §3 (Capa C). Se mantiene igual en los tres niveles:

- **Concordancia:** señal reforzada.
- **Conflicto:** distribución más ancha, escenarios condicionales.
- **Abstención:** cuando escenario es OOD, ningún experto activo es `factible`, o
  incertidumbre supera umbral.

---

## 7. Presupuesto de operaciones (nuevo V3)

Cuando hay múltiples señales simultáneas, el orquestador selecciona las top-N por
**ventaja neta esperada** ($E[R_{\text{neto}}]$ del módulo 15):

$$\text{top-N} = \text{argsort}(E[R_{\text{neto}}])[-N:]$$

Criterios adicionales para el presupuesto:
- No más de M activos correlacionados con BTC simultáneamente (límite de concentración).
- Exposición bruta máxima = 100 % del capital.
- N tentativo: 3–5 operaciones por semana (parámetro del sistema).

---

## 8. Comparaciones obligatorias (igual que V2 + extensión)

El orquestador solo se justifica si supera, fuera de muestra, a:

1. Promedio simple de expertos (nivel 0a).
2. Hedge / Fixed Share (nivel 0b).
3. Mejor experto único.
4. Mejor experto por régimen (sin gating).
5. Nivel 1 (contracción jerárquica) vs nivel 0.
6. Nivel 2 (gating) vs nivel 1.

---

## 9. Entrenamiento (actualizado V3)

**Calendario anidado (igual que V2):**

| Etapa | Periodo (tentativo) | Qué se hace |
|---|---|---|
| 1 | 2018–2019 | Entrenamiento inicial de expertos |
| 2 | 2020–2022 | Expertos reentrenados solo con años previos; salidas OOF forman el material del orquestador |
| 3 | 2023 | Validación: umbrales, presupuesto, ajustes del nivel 1 |
| 4 | 2024 | Prueba intermedia del sistema completo |
| 5 | 2025 | Test final (una sola vez) + monedas reservadas |

**El HMM** se ajusta al principio de cada etapa con los datos de entrenamiento de esa
etapa. Los estados del HMM en las etapas de validación se predicen con el modelo de la
etapa anterior.

---

## 10. Riesgos

1. **Nivel 0 ya es difícil de superar:** si Hedge iguala al nivel 1, la inversión en
   complejidad no se justifica.
2. **Inestabilidad del HMM entre pliegues:** los estados pueden permutarse; usar el
   protocolo de alineación del módulo 04 §3.5.
3. **Presupuesto de operaciones reduce la potencia estadística:** con top-3, habrá
   pocas operaciones por pliegue para intervalos de confianza fiables.
4. **Colapso del gating (nivel 2):** un experto domina; vigilar la distribución de
   pesos y aplicar regularización de balance de carga.
