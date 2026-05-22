# Registro de Cambios en la Documentación del Pipeline

**Fecha:** Mayo 2026
**Alcance:** Actualización de 6 archivos de documentación para reflejar la migración del pipeline de regresión a clasificación ordinal del riesgo epidémico de dengue.

---

## 1. Cambios Transversales (aplican a todos los archivos)

### 1.1 Terminología: "riesgo de transmisión" → "riesgo epidémico"
- **Antes:** La documentación se refería al modelo como predictor del "riesgo de transmisión de dengue."
- **Ahora:** Se utiliza "riesgo epidémico" en toda la documentación.
- **Justificación:** El modelo estima la probabilidad de que los casos superen el patrón histórico del municipio (riesgo epidémico), no la aptitud del territorio para sostener la transmisión del virus (riesgo de transmisión). Un municipio con transmisión activa permanente puede estar clasificado como "éxito" si sus casos están por debajo de lo esperado.

### 1.2 Target: de conteo de casos a clasificación ordinal
- **Antes:** Variable objetivo = `casosconfirmados` (regresión, conteo). Métricas: R², RMSE, MAE.
- **Ahora:** Variable objetivo = `nivel_riesgo` (clasificación ordinal, 3 clases). Métricas: Macro F1-Score, precision, sensibilidad, tasa de falsos positivos, MAE ordinal, Kappa de Cohen.
- **Justificación:** El conteo de casos varía enormemente entre municipios (0-2000+). El nivel de riesgo normaliza por la escala del municipio, haciendo comparables territorios heterogéneos.

### 1.3 Clases: de 4 a 3 niveles
- **Antes:** éxito (0), seguridad (1), alerta (2), epidemia (3) basados en percentiles P25/P50/P75.
- **Ahora:** éxito (0), en rango (1), epidemia (2) basados en canal endémico de Bortman (LI IC95%, LS IC95%).
- **Justificación:** Las clases seguridad y alerta (franja entre la media geométrica y los límites del IC95%) eran indistinguibles para los clasificadores, con F1 por clase de apenas 0.33 para alerta. Colapsar ambas en "en rango" alineó la clasificación con la interpretación operativa del INS en los BES (por debajo, dentro, por encima del canal).

### 1.4 Canal endémico: de percentiles a Bortman
- **Antes:** Percentiles P25/P50/P75 de la distribución histórica de casos.
- **Ahora:** Media geométrica con intervalo de confianza del 95%, metodología de Bortman (OPS, 1999).
- **Justificación:** El método Bortman es el estándar oficial del INS en los BES de Colombia. La media geométrica suaviza distribuciones sesgadas (log-normal, típicas de conteos de casos) mejor que los percentiles simples.

---

## 2. Cambios por Archivo

### 2.1 `documentacion_metodologica.md`
| Sección | Cambio | Justificación |
|:---|:---|:---|
| §1 Descripción de datos | Target descrito como "nivel de riesgo epidémico" derivado del canal de Bortman, no como conteo de casos | Refleja la variable objetivo real del modelo |
| §4 (NUEVA) | Sección completa sobre canal endémico de Bortman: fórmulas, clasificación en 3 niveles, prevención de leakage, nivel_endemico como feature | Documenta la decisión de diseño con mayor impacto (+20 pts en F1) |
| §5 Transformaciones | Agregada compresión PCA (5 grupos, 35→22 columnas) | Refleja la reducción de dimensionalidad implementada |
| §6 Resultados | Tabla de resultados reescrita con métricas de clasificación (F1, precision, sensibilidad). Agregada tabla de métricas por clase y perfil de alerta epidémica | Reemplaza métricas de regresión (R², RMSE) que ya no aplican |
| §7 Limitaciones | Agregada recomendación de validación prospectiva 3-6 meses | Requisito para despliegue en salud pública |

### 2.2 `S1_data_preparation.md`
| Sección | Cambio | Justificación |
|:---|:---|:---|
| Descripción general | "modelos de Machine Learning" → "construcción del canal endémico y modelado de clasificación" | Refleja el flujo actual |
| §4 ENSO | Mención de que las 16 dummies se comprimen en 2 PCA en S3.5 | Trazabilidad de la transformación |
| §6 Lags | "dinámica de transmisión" → "dinámica epidémica" | Consistencia terminológica |

### 2.3 `S2_pre_processing.md`
| Sección | Cambio | Justificación |
|:---|:---|:---|
| Descripción general | Agregado: construcción de variable objetivo vía canal endémico de Bortman | S2 ahora no solo imputa sino que construye el target |
| §2 Canal Endémico (NUEVO) | Sección completa: cálculo Bortman, prevención de leakage, colapso 4→3 clases, nivel_endemico | Documenta la funcionalidad central de S2 |
| §3 Feature Engineering | Agregados: lags de nivel_riesgo, columna nivel_riesgo_label | Nuevas features generadas |
| §4 Checkpoints (NUEVO) | Documentado el sistema de checkpoints y --skip_channel | Funcionalidad crítica para re-ejecución eficiente |
| Outputs | Actualizado: dengue_imputed con nivel_riesgo (0-2), umbrales Bortman, checkpoints | Refleja los archivos actuales |

### 2.4 `S3_feature_engineering.md`
| Sección | Cambio | Justificación |
|:---|:---|:---|
| §1 Descripción | Nota sobre target ordinal y uso de Kruskal-Wallis | El diagnóstico opera sobre 3 clases, no sobre conteo |
| §2A Pearson | Referencia a PCA ENSO como solución a correlaciones 0.80-0.84 | Documenta la acción tomada ante las alertas de redundancia |
| §2D Kruskal-Wallis (NUEVO) | Sección completa sobre test de separabilidad entre clases | Reemplaza la correlación lineal con el target (que era para regresión) |
| §2E Random Forest | "Random Forest Regressor" → "Random Forest Classifier". Hallazgo de nivel_endemico al 30% | Refleja el clasificador actual y su resultado |
| §3 Guía de lectura | Actualizada distribución esperada del target (3 clases). Secciones de diagnóstico actualizadas | Alineación con los reportes actuales de S3 |

### 2.5 `S3_5_testing_feature_engineering.md`
| Sección | Cambio | Justificación |
|:---|:---|:---|
| §1A Pre-procesamiento | Detalle de los 5 grupos PCA con dimensiones. Documentado que nivel_endemico se preserva | Refleja setup_pre_s4.JSON actual |
| §1B Diagnóstico | Adaptado a target ordinal (3 clases). Kruskal-Wallis documentado | El diagnóstico ahora evalúa clasificación, no regresión |
| §4 Estructura (NUEVA) | Tabla completa de las 22 columnas del dataset final | Referencia clara para S4 |
| Target en outputs | `nivel_riesgo` ahora tiene valores 0, 1, 2 (no 0-3) | Consistencia con 3 clases |

### 2.6 `S4_processing.md`
| Sección | Cambio | Justificación |
|:---|:---|:---|
| Descripción general | "Regresión" → "Clasificación ordinal de riesgo epidémico (3 clases)" | Cambio fundamental del enfoque |
| §4 Modelos | Ridge Regression eliminado. Logistic Regression agregado como baseline. XGB/LGBM con num_class=3. CatBoost con auto_class_weights | Modelos de clasificación, no de regresión |
| §5 Validación | SpatioTemporalSplit → TimeSeriesSplit (5 folds) + hold-out 2024 | Esquema de validación simplificado |
| §6 Métricas (NUEVA) | Tabla de métricas: Macro F1, exactitud, MAE ordinal, Kappa Cohen, precision/sensibilidad/TFP por clase | Reemplaza R², RMSE, MAE de regresión |
| §7 Flujo | Umbral de guardado: RMSE < 15 → F1_val ≥ 0.50. Exportación de predicciones con etiquetas textuales | Criterios de clasificación |
| §8 Outputs | Nomenclatura estándar en español: sensibilidad, exactitud, tasa_fp, _val, metodo_imputacion | Consistencia terminológica |

---

## 3. Archivos NO Modificados
- `README.md`: No se encontraba en el directorio de outputs. Se recomienda actualizar si existe una versión controlada.

---

## 4. Resumen del Impacto

| Aspecto | Antes | Ahora |
|:---|:---|:---|
| Problema | Regresión (predecir conteo de casos) | Clasificación ordinal (predecir nivel de riesgo epidémico) |
| Target | casosconfirmados (conteo) | nivel_riesgo (0, 1, 2) |
| Canal endémico | Percentiles P25/P50/P75 | Bortman: MG + IC95% |
| Clases | 4 (éxito, seguridad, alerta, epidemia) | 3 (éxito, en rango, epidemia) |
| Feature clave | No existía | nivel_endemico (MG histórica, 30% importancia) |
| Mejor métrica | R² ≈ 0.80 (regresión RF) | Macro F1 = 0.729 (clasificación LGBM) |
| Precision epidemia | N/A | 91.7% |
| Sensibilidad epidemia | N/A | 76.6% |
| Terminología | "riesgo de transmisión" | "riesgo epidémico" |
