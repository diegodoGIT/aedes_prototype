# Documentación Técnica: S4_processing_script.py (Clasificación de Riesgo Epidémico)

## Descripción General
El script `S4_processing_script.py` es el componente final del pipeline. Su responsabilidad es el entrenamiento, optimización de hiperparámetros y evaluación de modelos de clasificación ordinal para predecir el nivel de riesgo epidémico de dengue en tres categorías: éxito (0), en rango (1) y epidemia (2).

---

## 1. Entrada de Datos
S4 **no realiza ingeniería de variables ni pre-procesamiento**. Busca exclusivamente archivos con sufijo `_modified_` generados por S3.5:

*   **Patrón de búsqueda**: `dengue_imputed_M*_modified_*.csv`
*   **Columnas esperadas**: 22 variables numéricas incluyendo `nivel_riesgo` (target, valores 0-2) y `nivel_endemico` (feature).

---

## 2. Recuperación de Metadatos Geográficos
Los datasets de S3.5 carecen de etiquetas textuales (`departamento`, `municipio`). S4 restaura esta información:
*   **Fuente**: `data/processed/municipios_coordenadas.csv` (generado por S2).
*   **Mecanismo**: Cruce por `latitud` y `longitud`.
*   **Propósito**: Generar el reporte de métricas regionales por departamento.

---

## 3. Modos de Ejecución

### A. Modo Iterativo (Por defecto)
Entrena 5 clasificadores con búsqueda aleatoria de hiperparámetros (RandomizedSearchCV):
```bash
python S4_processing_script.py --n_iter 10
```

### B. Modo Fast (Rápido)
Entrena un único modelo con configuración base:
```bash
python S4_processing_script.py --fast lgbm
```

| Parámetro | Tipo | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--fast` | `str` | `None` | Modelo específico: `logistic`, `rf`, `xgb`, `lgbm` o `catboost`. |
| `--n_iter` | `int` | `5` | Combinaciones de hiperparámetros a evaluar por modelo. |

---

## 4. Modelos y Espacios de Hiperparámetros

### Logistic Regression (Baseline)
Clasificador lineal multinomial. Sirve como referencia para cuantificar el aporte de la no linealidad.
- `C`: 10⁻³ a 10³ (log-uniforme)
- `solver`: lbfgs, saga
- `max_iter`: 5000

### Random Forest Classifier
Ensemble de árboles con bagging. Robusto y capaz de capturar interacciones no lineales.
- `n_estimators`: 100-1000
- `max_depth`: None, 5-30
- `min_samples_split`: 2-20
- `max_features`: sqrt, log2, 0.3-0.7
- `class_weight`: None, balanced, balanced_subsample

### XGBoost Classifier
Gradient Boosting con regularización L1/L2. `objective='multi:softmax'`, `num_class=3`.
- `n_estimators`: 100-500
- `max_depth`: 2-6
- `learning_rate`: 0.01-0.2
- `subsample/colsample_bytree`: 0.7-0.9
- `reg_lambda`: 1-100

### LightGBM Classifier
Gradient Boosting leaf-wise. `objective='multiclass'`, `num_class=3`.
- `n_estimators`: 100-500
- `max_depth`: -1, 5-15
- `num_leaves`: 31-255
- `learning_rate`: 0.01-0.2
- `class_weight`: None, balanced

### CatBoost Classifier
Gradient Boosting con Ordered Boosting para datos temporales.
- `iterations`: 100-500
- `depth`: 4-12
- `learning_rate`: 0.01-0.1
- `l2_leaf_reg`: 1-15
- `auto_class_weights`: SqrtBalanced, Balanced

---

## 5. Metodología de Validación

### Validación Cruzada Temporal
Se utiliza `TimeSeriesSplit` con 5 folds. Solo se usa el pasado para predecir el futuro, preservando la causalidad temporal.

### Validación Externa Hold-Out
El año 2024 se excluye completamente del entrenamiento y la validación cruzada. Sirve como evaluación final que simula el uso operativo real del modelo.

---

## 6. Métricas de Evaluación

Todas las métricas utilizan nomenclatura estándar:

| Métrica | Descripción |
|:---|:---|
| **Macro F1-Score** | Métrica principal de selección. Promedio no ponderado del F1 de cada clase. |
| **Exactitud** | Fracción de predicciones correctas sobre el total. |
| **MAE Ordinal** | Error absoluto medio entre categorías ordinales. Rango: 0 (perfecto) a 2 (peor caso). Usa `.ravel()` para prevenir error de memoria con arrays 2D. |
| **Kappa de Cohen** | Concordancia ajustada por azar con ponderación cuadrática. |
| **Precision por clase** | Fracción de predicciones correctas para cada nivel de riesgo. |
| **Sensibilidad por clase** | Fracción de casos reales detectados para cada nivel de riesgo. |
| **Tasa de Falsos Positivos** | Proporción de no-epidemias clasificadas erróneamente como esa clase. |

---

## 7. Flujo de Trabajo
1.  **Carga**: Lectura de datasets `_modified_` de S3.5.
2.  **Split**: Separación de features numéricas (excluyendo `nivel_riesgo` y `nivel_riesgo_label`) y target.
3.  **Filtrado temporal**: Exclusión del año 2024 para validación externa.
4.  **Escalado**: `StandardScaler` ajustado solo en el set de entrenamiento.
5.  **Búsqueda de hiperparámetros**: `RandomizedSearchCV` con `TimeSeriesSplit`.
6.  **Evaluación externa**: Predicción sobre año 2024 con métricas completas.
7.  **Persistencia**: Modelos con F1_val ≥ 0.50 se guardan como `.pkl`.
8.  **Predicciones**: El mejor modelo exporta CSV con columnas `etiqueta_real` y `etiqueta_predicha` en texto.

---

## 8. Resultados (Outputs)
Almacenados en `processing/`:
*   **`experiment_technical_log.csv`**: Registro de cada iteración con métricas globales y por clase.
*   **`regional_performance.csv`**: Desempeño por departamento.
*   **`{método}_{modelo}_best.pkl`**: Modelo serializado.
*   **`{método}_{modelo}_predicciones.csv`**: Predicciones con etiquetas textuales.

### Nomenclatura de columnas en el CSV de resultados
| Columna | Descripción |
|:---|:---|
| `metodo_imputacion` | M1-M5 |
| `modelo` | logistic, rf, xgb, lgbm, catboost |
| `macro_f1_score_cv` | F1-Score macro en validación cruzada |
| `macro_f1_score_val` | F1-Score macro en validación externa (2024) |
| `sensibilidad_{clase}_val` | Sensibilidad por clase en validación externa |
| `precision_{clase}_val` | Precision por clase en validación externa |
| `tasa_fp_{clase}_val` | Tasa de falsos positivos por clase |
| `duracion_seg` | Tiempo de entrenamiento en segundos |
| `parametros` | Hiperparámetros de la iteración |
