# Documentación Técnica: S3_5_testing_feature_engineering.py

## Descripción General
Este script actúa como un puente crítico entre la imputación de datos (S2) y el modelado (S4). Su función principal es aplicar transformaciones de ingeniería de variables sobre el target (`casosconfirmados`) y ejecutar ajustes manuales definidos por el usuario, validando la calidad del dataset resultante mediante pruebas estadísticas.

---

## 1. Responsabilidades Principales

### A. Pre-procesamiento Manual (`setup_pre_s4.JSON`)
Aplica las reglas de limpieza y consolidación definidas en el archivo JSON:
*   **Promedios de columnas**: Combina variables redundantes.
*   **Renombramiento**: Normaliza nombres de atributos.
*   **Eliminación selectiva**: Remueve columnas ruidosas.
*   **Transformación PCA**: Aplica Análisis de Componentes Principales a grupos de variables para reducir dimensionalidad:
    *   **Escalado Automático**: El script aplica `StandardScaler` antes del PCA.
    *   **Configuración**: Se define en `pca_groups` especificando columnas, número de componentes y prefijo.

### B. Ingeniería del Target
Genera variables predictivas basadas en el comportamiento histórico del dengue:
*   **Rezagos (Lags)**: Crea columnas con los valores de `casosconfirmados` de meses anteriores (configurable mediante `--target_lags`).
*   **Media Móvil (Moving Average)**: Calcula el promedio de casos en una ventana temporal previa (configurable mediante `--moving_avg`), aplicando un `shift(1)` para evitar fuga de información (*data leakage*).

### C. Diagnóstico de Calidad
Ejecuta el mismo conjunto de pruebas que **S3_feature_engineering.py** para asegurar que las nuevas variables no introduzcan problemas:
*   **VIF (Variance Inflation Factor)**: Detecta multicolinealidad.
*   **Pruebas de Granger**: Valida la precedencia temporal de las nuevas variables.
*   **Importancia con Random Forest**: Evalúa la relevancia predictiva no lineal.

---

## 2. Parámetros de Ejecución
```bash
python S3_5_testing_feature_engineering.py --target_lags 3 --moving_avg 3
```

| Parámetro | Tipo | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--target_lags` | `int` | `3` | Cantidad de meses de historia a incluir como features individuales. |
| `--moving_avg` | `int` | `3` | Tamaño de la ventana para el promedio móvil suavizado. |

---

## 3. Configuración de PCA en `setup_pre_s4.JSON`
Para aplicar PCA, se debe editar el campo `pca_groups` en el archivo de configuración:

```json
"pca_groups": [
    {
        "columns": ["tempmax", "tempmin", "temp_avg"],
        "n_components": 1,
        "prefix": "pca_temperatura",
        "drop_originals": true
    }
]
```

*   **`columns`**: Lista de variables a colapsar.
*   **`n_components`**: Cantidad de dimensiones a mantener.
*   **`prefix`**: Nombre base para las nuevas columnas (ej. `pca_temperatura_1`).
*   **`drop_originals`**: Si es `true`, elimina las variables de entrada tras crear los componentes.

---

## 4. Salida y Trazabilidad
El script genera nuevos archivos en `data/processed/` siguiendo el patrón:
`dengue_imputed_M{X}_modified_{YYYYMMDD_HHMMSS}.csv`

Estos archivos son los **únicos** que el script **S4_processing_script.py** procesará para el entrenamiento final.
