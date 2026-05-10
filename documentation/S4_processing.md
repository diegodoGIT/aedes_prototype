# Documentación Técnica: S4_processing_script.py (Entrenamiento y Modelado)

## Descripción General
El script `S4_processing_script.py` es el componente final del pipeline de modelado. Su única responsabilidad es el entrenamiento, optimización y validación de arquitecturas de Machine Learning (Ridge, RF, XGB, LGBM, CatBoost) utilizando datasets previamente enriquecidos y validados por el script **S3.5**.

---

## 1. Entrada de Datos
A diferencia de versiones anteriores, S4 **no realiza ingeniería de variables ni pre-procesamiento manual**. El script busca exclusivamente archivos que contengan el sufijo `_modified_` en la carpeta `data/processed/`.

*   **Patrón de búsqueda**: `dengue_imputed_M*_modified_*.csv`
*   **Origen**: Estos archivos deben ser generados previamente por `S3_5_testing_feature_engineering.py`.

---

## 2. Recuperación de Metadatos Geográficos
Dado que los datasets procesados por S3.5 suelen carecer de etiquetas de texto (`departamento`, `municipio`) debido a la codificación dummy o eliminación explícita para el modelado, el script S4 integra una fase de restauración:
*   **Fuente**: Utiliza el archivo auxiliar `data/processed/municipios_coordenadas.csv` generado en la etapa S2.
*   **Mecanismo**: Realiza un cruce (*merge*) basado en las columnas `latitud` y `longitud`.
*   **Propósito**: Esta restauración es esencial para generar el reporte de métricas regionales (`regional_performance.csv`), permitiendo desglosar el error del modelo por entidad territorial real.

---

## 3. Modos de Ejecución
El script mantiene su flexibilidad para pruebas y producción:

### A. Modo Iterativo (Por defecto)
Entrena todos los modelos soportados realizando una búsqueda de hiperparámetros mediante muestreo aleatorio (`ParameterSampler`).
*   **Uso**: `python S4_processing_script.py --n_iter 10`

### B. Modo Fast (Rápido)
Entrena un único modelo específico con configuración base.
*   **Uso**: `python S4_processing_script.py --fast xgb`

### Parámetros de Comando:
| Parámetro | Tipo | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--fast` | `str` | `None` | Activa el modo rápido para un modelo: `ridge`, `rf`, `xgb`, `lgbm` o `catboost`. |
| `--n_iter` | `int` | `5` | Número de combinaciones de hiperparámetros a probar por modelo. |

---

## 3. Metodología de Validación
Se mantiene el uso de `SpatioTemporalSplit` para garantizar:
-   **Consistencia de Años**: Sin fugas estacionales.
-   **Gap de Purga**: Un año de separación entre train y val.
-   **Causalidad**: Solo se usa el pasado para predecir el futuro.

---

## 4. Flujo de Trabajo ML
1.  **Carga**: Lectura de datasets modificados por S3.5.
2.  **Filtrado**: Exclusión de datos de 2024 para validación final externa.
3.  **Escalado**: Normalización mediante `StandardScaler` ajustado solo en el set de entrenamiento.
4.  **Optimización**: Búsqueda aleatoria de hiperparámetros.
5.  **Persistencia**: Los mejores modelos (con RMSE < 15.0) se guardan como archivos `.pkl` en `processing/`.

---

## 5. Resultados y Métricas
-   **`experiment_technical_log.csv`**: Registro técnico global.
-   **`regional_performance.csv`**: Desempeño detallado por departamento para evaluar equidad espacial.

---

## 6. Arquitecturas de Modelos y Espacios de Búsqueda
(Se mantienen las especificaciones de Ridge, RF, XGB, LGBM y CatBoost detalladas anteriormente, con espacios de búsqueda de aproximadamente 5,000 combinaciones cada uno).
