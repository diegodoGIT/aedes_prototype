# Documentación: Preprocesamiento e Imputación de Datos (S2)

## Descripción General
El script `S2_pre_processing_script.py` constituye la segunda fase del pipeline. Su función principal es resolver el problema de los datos faltantes en la variable objetivo (casos de dengue) mediante diversos métodos estadísticos, generar nuevas características para el modelado y evaluar la representatividad geográfica de los datos recuperados.

## Funcionamiento del Script
El script opera a través de un pipeline de cuatro componentes principales:

### 1. Métodos de Imputación (M1 - M5)
Se generan cinco datasets independientes para comparar diferentes estrategias de recuperación de datos:
*   **M1 (Media Histórica):** Imputa usando el promedio de casos para el mismo mes y municipio en años anteriores (requiere al menos 4 años de historia).
*   **M2 (Mediana Histórica):** Similar a M1, pero utiliza la mediana para mayor robustez ante brotes atípicos.
*   **M3 (Interpolación Lineal):** Rellena huecos temporales mediante una pendiente lineal entre puntos conocidos (límite de brecha: 3 meses).
*   **M4 (EWM - Exponential Weighted Mean):** Utiliza un promedio móvil exponencial con un factor de suavizado ($\alpha=0.3$) para capturar tendencias recientes.
*   **M5 (Baseline):** No aplica imputación; solo elimina los registros que permanecen nulos tras el procesamiento.

### 2. Ingeniería de Características (Feature Engineering)
Para cada dataset imputado, se aplican las siguientes transformaciones:
*   **Variables Cíclicas:** Transformación del mes en coordenadas `mes_sin` y `mes_cos` para que el modelo entienda la naturaleza circular del año (diciembre cerca de enero).
*   **Lags del Target:** Creación de rezagos configurables para la variable `casosconfirmados` (por defecto 3 meses).
*   **Recuperación Ambiental:** Imputación de variables climáticas faltantes (NDVI, Dew Point, etc.) usando la media local (municipio o departamento) para evitar la pérdida de registros recuperados por la imputación del target.

## Parámetros de Ejecución
```bash
python S2_pre_processing_script.py --target_lags 3
```

| Parámetro | Tipo | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--target_lags` | `int` | `3` | Cantidad de meses de historia de la variable objetivo a generar como predictores. |

## Requisitos de Entrada (Inputs)
El script genera un análisis crítico de representatividad:
*   Evalúa cada departamento para verificar si posee suficientes datos tras la imputación y el feature engineering.
*   Registra las razones de exclusión (ej. datos insuficientes) para asegurar la transparencia en la selección del dataset final.

## Requisitos de Entrada (Inputs)
El script requiere el output generado por la etapa S1:
*   **Archivo:** `data/dengue_data_v2_ocur.csv` (o `_res.csv`).
*   **Estructura:** Debe contener columnas de ubicación (`departamento`, `municipio`), tiempo (`year`, `mes`) y las variables climáticas base con sus lags pre-calculados.

## Resultados (Outputs)
Los resultados se almacenan en la carpeta `data/processed/`:
*   `dengue_imputed_M1.csv` hasta `M5.csv`: Datasets finales listos para entrenamiento, cada uno correspondiente a un método de imputación.
*   `cobertura_departamentos.csv`: Reporte detallado de la recuperación de datos por región geográfica.

## Sistema de Logs
El script está integrado al sistema centralizado de trazabilidad:
*   **Archivo:** `project_logs.log`.
*   **Información Registrada:**
    *   Dimensiones del dataset de entrada (filas y columnas).
    *   Identificación de variables base con rezagos.
    *   Inferencia automática del número de meses de historia (lags) por variable.
    *   Timestamps de inicio y fin, incluyendo la duración total del proceso de imputación.
