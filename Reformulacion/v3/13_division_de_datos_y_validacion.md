# Módulo 13 — División de Datos y Validación (V3)

**Función:** definir las particiones de datos que garantizan que ningún módulo vea
el futuro durante el entrenamiento.

> **Cambios V3:** purga adaptada a horizontes en horas para episodios tácticos, CPCV
> como método de validación principal, ajuste del calendario para cubrir los tres
> estilos.

---

## 1. Estructura global (actualizada V3)

| Período | Uso |
|---|---|
| 2018–2019 | Entrenamiento inicial de expertos (etapa 1 del calendario anidado) |
| 2020–2022 | Generación de salidas OOF para el orquestador (etapa 2) |
| 2023 | Validación: umbrales del orquestador, presupuesto, ajustes (etapa 3) |
| 2024 | Prueba intermedia del sistema completo (etapa 4) |
| **2025** | **Test final — una sola vez, al final de todo el desarrollo** |

Las monedas reservadas (≈5 de 25) no participan en ninguna etapa de entrenamiento ni
validación. Solo se usan en el test final 2025.

---

## 2. Pliegues walk-forward / CPCV (actualizado V3)

**V2:** walk-forward con F1–F4 (un único backtest por pliegue).

**V3:** CPCV — se generan todas las combinaciones de pliegues de entrenamiento y test
con purga. Con 4 pliegues (F1–F4) se producen $\binom{4}{2} = 6$ trayectorias de
backtest, cada una con un período de test diferente y el resto como entrenamiento.

Esto produce una **distribución del Sharpe y del retorno**, no un único número, lo que
permite aplicar el Sharpe deflactado y detectar si un solo pliegue "bueno" domina.

---

## 3. Purga y embargo por estilo (actualizado V3)

La purga elimina del entrenamiento las observaciones cuya ventana de etiqueta se solapa
con el test. El embargo añade un margen adicional tras la purga.

| Estilo | Horizonte máx. | Purga (antes del test) | Embargo (después del test) |
|---|---|---|---|
| Táctico | 24 h | 24 h | 2 h |
| Swing corto | 7 días | 7 días | 1 día |
| Swing/posicional | 30 días | 30 días | 3 días |

**Purga por replay:** para episodios cuya ventana de gestión cruza el límite del
pliegue, se purga el episodio completo del entrenamiento.

---

## 4. Calendarios separados por estilo

Los tres estilos tienen calendarios diferentes:

| Estilo | Frecuencia de episodios | Bloque mínimo para bootstrap |
|---|---|---|
| Táctico | Horas (CUSUM ~12h promedio) | Bloques de 24–48h |
| Swing corto | Días (CUSUM ~3d promedio) | Bloques de 7 días |
| Swing/posicional | Semanas (CUSUM ~15d promedio) | Bloques de 30 días |

---

## 5. Monedas reservadas (igual que V2)

- ≈5 de 25 monedas, estratificadas (1 principal, 2 alta cap., 2 alternativas).
- Fijadas y versionadas en `config/reserved_assets_v3.json` **antes** de entrenar.
- Solo se usan en el test final 2025 y para evaluar la generalización del sistema.

---

## 6. Calendario anidado para el orquestador (igual que V2)

Los expertos se entrenan solo con datos anteriores a cada pliegue. Sus salidas
fuera de muestra (OOF) forman el material de entrenamiento del orquestador.

El HMM (módulo 04) se ajusta al inicio de cada etapa con los datos de entrenamiento
de esa etapa. El estado del HMM en validación/test se predice con el modelo ajustado
en la etapa anterior.

---

## 7. Decisiones tentativas a validar con datos en V3

- Número y composición exacta de monedas reservadas.
- Umbral CUSUM $h$ (afecta el número de episodios por pliegue).
- Tamaño de bloque para bootstrap por estilo.
- Si los pliegues del estilo táctico tienen suficientes episodios independientes para
  el CPCV (mínimo 30 por trayectoria).

---

## 8. Protección del test final 2025

El test final 2025 permanece **completamente intacto** hasta que:

1. Todas las decisiones de diseño estén fijadas (sin cambios posibles).
2. Las hipótesis H1–H9 hayan sido evaluadas y los resultados documentados.
3. El sistema esté en modo "uso real" (no investigación).

**Cualquier consulta del test final lo consume**. No se puede usar parcialmente.
