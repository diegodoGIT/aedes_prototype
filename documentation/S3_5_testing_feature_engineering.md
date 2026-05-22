# Documentación Técnica: S3_5_testing_feature_engineering.py

## Descripción General
Este script actúa como un puente crítico entre la imputación de datos (S2) y el modelado (S4). Su función principal es aplicar transformaciones de reducción de dimensionalidad definidas en `setup_pre_s4.JSON`, ejecutar diagnóstico estadístico adaptado al target ordinal (`nivel_riesgo`, 3 clases) y generar los datasets finales listos para el entrenamiento.

---

## 1. Responsabilidades Principales

### A. Pre-procesamiento Configurable (`setup_pre_s4.JSON`)
Aplica las reglas de limpieza y consolidación definidas en el archivo JSON:
*   **Eliminación selectiva:** Remueve columnas que no deben entrar al modelo (`casosconfirmados`, `umbral_li`, `umbral_ls`, `nivel_riesgo_label`, metadatos geográficos textuales).
*   **Promedios de columnas:** Combina `tempmax` y `tempmin` en `temp_avg` (y sus lags).
*   **Transformación PCA:** Aplica Análisis de Componentes Principales a 5 grupos de variables:
    *   `pca_dew_point` (4 vars → 1 comp): Punto de rocío + lags
    *   `pca_casos_lags` (3 vars → 1 comp): Lags de casos confirmados
    *   `pca_riesgo_lags` (3 vars → 1 comp): Lags del nivel de riesgo
    *   `pca_temp_avg` (4 vars → 1 comp): Temperatura promedio + lags
    *   `pca_enso` (16 vars → 2 comp): Dummies ENSO (4 categorías × 4 lags)
*   **Preservación de `nivel_endemico`:** La media geométrica histórica se mantiene como feature (no se elimina ni se incluye en PCA).

### B. Diagnóstico de Calidad (adaptado a target ordinal)
Ejecuta un conjunto de pruebas estadísticas sobre el dataset transformado:
*   **Correlación de Pearson:** Detecta redundancia bivariada (umbral 0.80). Tras PCA, las alertas de ENSO desaparecen; persisten en NDVI para M3/M5 (r=0.80 entre lags).
*   **Kruskal-Wallis:** Test no paramétrico que evalúa si cada variable discrimina entre los 3 niveles de riesgo. Variables con p>0.05 se reportan como "sin poder discriminativo".
*   **Causalidad de Granger:** Valida la precedencia temporal de cada variable respecto al target ordinal.
*   **Importancia con Random Forest Classifier:** Evalúa la relevancia predictiva no lineal de cada variable para la clasificación ordinal.

---

## 2. Parámetros de Ejecución
```bash
python S3_5_testing_feature_engineering.py --moving_avg 0
```

| Parámetro | Tipo | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--moving_avg` | `int` | `0` | Tamaño de la ventana para el promedio móvil suavizado (0 = desactivado). |

---

## 3. Configuración de PCA en `setup_pre_s4.JSON`

```json
{
    "drop_columns": ["departamento", "municipio", "date", "cod_muni", "tipo",
                     "casosconfirmados", "umbral_li", "umbral_ls", "nivel_riesgo_label"],
    "pca_groups": [
        {
            "columns": ["fenomeno_nina_mod", "fenomeno_nina_sev", "...16 dummies..."],
            "n_components": 2,
            "prefix": "pca_enso",
            "drop_originals": true
        }
    ]
}
```

*   **`columns`**: Lista de variables a comprimir.
*   **`n_components`**: Dimensiones a mantener.
*   **`prefix`**: Nombre base para las nuevas columnas (ej. `pca_enso_1`, `pca_enso_2`).
*   **`drop_originals`**: Si es `true`, elimina las variables de entrada tras crear los componentes.

---

## 4. Estructura Final del Dataset

Tras las transformaciones, el dataset resultante contiene 22 columnas:

| Variable | Tipo | Descripción |
|:---|:---|:---|
| year, mes | int | Coordenadas temporales |
| longitud, latitud | float | Coordenadas geográficas del municipio |
| precipitacion + 3 lags | float | Precipitación actual y rezagada |
| ndvi + 3 lags | float | Índice de vegetación actual y rezagado |
| nivel_endemico | float | Media geométrica histórica (escala del municipio) |
| mes_sin, mes_cos | float | Codificación cíclica del mes |
| pca_dew_point_1 | float | PCA del punto de rocío + lags |
| pca_casos_lags_1 | float | PCA de casos rezagados (1-3 meses) |
| pca_riesgo_lags_1 | float | PCA del nivel de riesgo rezagado |
| pca_temp_avg_1 | float | PCA de temperatura promedio + lags |
| pca_enso_1, pca_enso_2 | float | PCA del estado ENSO |
| nivel_riesgo (TARGET) | int (0,1,2) | Éxito / En rango / Epidemia |

---

## 5. Salida y Trazabilidad
El script genera nuevos archivos en `data/processed/` siguiendo el patrón:
`dengue_imputed_M{X}_modified_{YYYYMMDD_HHMMSS}.csv`

Estos archivos son los **únicos** que el script **S4_processing_script.py** procesará para el entrenamiento final.
