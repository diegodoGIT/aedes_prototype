# Proyecto Aedes Prototype: Predicción de Dengue en Colombia

Este proyecto implementa un pipeline automatizado de nivel industrial para la recolección, procesamiento, imputación y modelado predictivo de casos de dengue en Colombia, integrando variables climáticas, satelitales y fenómenos macroclimáticos.

## Estructura del Pipeline

El flujo de trabajo se divide en cuatro etapas principales, cada una documentada en la carpeta `documentation/`:

1.  **S1: Preparación de Datos (`S1_data_preparation_script.py`):**
    *   Consolida casos de SISPRO con coordenadas DANE (DIVIPOLA).
    *   Extrae elevación y clima (Temperatura/Precipitación) desde archivos Raster de WorldClim.
    *   Integra el fenómeno ENSO (Niño/Niña) y variables ambientales (NDVI/Dew Point).
    *   Genera rezagos (lags) configurables para todas las variables ambientales.
2.  **S2: Preprocesamiento e Imputación (`S2_pre_processing_script.py`):**
    *   Aplica 5 métodos de imputación (M1-M5) para tratar vacíos en el target.
    *   Genera características cíclicas (`mes_sin`, `mes_cos`) para capturar estacionalidad.
    *   Produce un reporte de cobertura geográfica por departamento.
3.  **S3: Ingeniería de Características y Diagnóstico (`S3_feature_engineering.py`):**
    *   Realiza un diagnóstico multidimensional (VIF, Pearson, Causalidad de Granger).
    *   Detecta redundancia, multicolinealidad y relevancia no lineal (Random Forest).
    *   Genera un **Resumen Compilado** de recomendaciones para ajustar el dataset.
4.  **S4: Modelado Espacio-Temporal (`S4_processing_script.py`):**
    *   Aplica transformaciones manuales basadas en el diagnóstico de S3 mediante el archivo `setup_pre_s4.JSON`.
    *   Implementa una **Validación Cruzada Espacio-Temporal** (`SpatioTemporalSplit`) con purga (GAP) de 1 año.
    *   Entrena modelos de ML (XGBoost, CatBoost, LightGBM, Random Forest, Ridge).
    *   Genera métricas de desempeño globales y desglosadas por departamento.

---

## Diccionario de Datos

### Identificadores y Tiempo
*   **`departamento` / `municipio`**: Nombres normalizados de la ubicación geográfica.
*   **`date`**: Fecha del registro (formato YYYY-MM-01).
*   **`year` / `mes`**: Año y mes (1-12) calendarios.

### Variables Objetivo y Geográficas
*   **`casosconfirmados`**: Número de casos de dengue y dengue grave reportados.
*   **`longitud` / `latitud` / `elevacion`**: Datos de ubicación y altitud.
*   **`target_lagN`**: Rezagos de la variable objetivo generados dinámicamente en la etapa S4.

### Variables Ambientales y Climáticas (WorldClim/Satelital)
*   **`TempMin` / `TempMax` / `Precipitacion`**: Variables climáticas mensuales.
*   **`ndvi` / `dew_point`**: Índice de vegetación y punto de rocío (humedad).
*   **`fenomeno`**: Categoría ENSO (Niño/Niña/Neutral).
*   **`[VAR]_lagN`**: Valores históricos de cada variable según el número de rezagos configurados en S1.

### Ingeniería de Características (Etapas S2 y S3)
*   **`mes_sin` / `mes_cos`**: Transformación trigonométrica del mes.
*   **Ajustes de Redundancia**: Variables combinadas o eliminadas según el archivo `setup_pre_s4.JSON`.

---

## Instrucciones de Ejecución

### 1. Preparación
Se recomienda Python 3.12. Instale las dependencias necesarias:
```bash
pip install -r requirements.txt
```

### 2. Ejecución Secuencial del Pipeline
Para un flujo completo, ejecute los scripts en este orden:
```bash
python S1_data_preparation_script.py
python S2_pre_processing_script.py
python S3_feature_engineering.py   # Diagnóstico: revise logs/consola antes de seguir
```

### 3. Configuración del Entrenamiento (S4)
El script `S4_processing_script.py` es altamente configurable mediante flags. A continuación se detallan las opciones y combinaciones:

#### Parámetros Disponibles:
| Flag | Obligatorio | Valores / Tipo | Descripción |
| :--- | :---: | :--- | :--- |
| `--fast` | No | `ridge`, `rf`, `xgb`, `lgbm`, `catboost` | **Modo Rápido**: Entrena solo el modelo indicado con parámetros base. |
| `--n_iter` | No | `int` (Default: `5`) | **Modo Iterativo**: Número de combinaciones de hiperparámetros a probar por cada modelo. |
| `--target_lags` | No | `int` (Default: `3`) | Número de meses previos de la variable objetivo a incluir como predictores. |

#### Combinaciones Comunes de Ejecución:

**A. Entrenamiento Exhaustivo (Todos los modelos):**
Útil para benchmarking inicial. Prueba 10 combinaciones por cada arquitectura con 4 meses de historia.
```bash
python S4_processing_script.py --n_iter 10 --target_lags 4
```

**B. Validación Rápida de un Modelo Específico:**
Útil para probar cambios en el pre-procesamiento o en el archivo `setup_pre_s4.JSON` sin esperar a todo el pipeline.
```bash
python S4_processing_script.py --fast xgb --target_lags 3
```

**C. Optimización Profunda de un Modelo (Simulado):**
Aunque `--n_iter` aplica a todos los modelos en modo iterativo, puedes combinar la lógica de S3 para dejar solo variables clave y correr:
```bash
python S4_processing_script.py --n_iter 50 --target_lags 6
```

---

## Sistema de Trazabilidad y Resultados
*   **Logs Generales:** `project_logs.log` (historial cronológico).
*   **Logs Técnicos:** `processing/experiment_technical_log.csv` (métricas de cada iteración).
*   **Desempeño Regional:** `processing/regional_performance.csv` (error desglosado por departamento).
*   **Configuración Manual:** `setup_pre_s4.JSON` (reglas de pre-procesamiento para S4).
*   **Modelos:** Carpeta `processing/` (archivos `.pkl` de los mejores modelos).
