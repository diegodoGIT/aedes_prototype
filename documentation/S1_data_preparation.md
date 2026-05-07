# Documentación: Pipeline de Preparación de Datos (S1)

## Descripción General
El script `S1_data_preparation_script.py` es el primer paso del pipeline de procesamiento del proyecto. Su objetivo es transformar los registros crudos de vigilancia epidemiológica (SIVIGILA) en un dataset enriquecido con variables biofísicas, climáticas y temporales, listo para el entrenamiento de modelos de Machine Learning.

## Funcionamiento del Script
El procesamiento sigue un flujo secuencial:

1.  **Normalización Geográfica:** Estandariza los nombres de departamentos y municipios (minúsculas, sin acentos) para garantizar el cruce entre diversas fuentes de datos (DANE vs. SISPRO).
2.  **Expansión Temporal:** Transforma los datos de formato ancho a largo y asegura que cada municipio tenga una serie de tiempo continua (mensual), rellenando huecos para evitar sesgos en los análisis de rezagos (lags).
3.  **Enriquecimiento Geospatial (Raster):** Utiliza la librería `rasterio` para extraer la elevación exacta y variables climáticas (Temperatura Mínima, Máxima y Precipitación) de archivos históricos de WorldClim basados en las coordenadas de cada municipio.
4.  **Integración ENSO:** Clasifica el estado del fenómeno de El Niño/Niña según anomalías térmicas y genera variables categóricas con rezagos temporales.
5.  **Variables Ambientales Adicionales:** Integra índices de vegetación (NDVI) y punto de rocío (Dew Point) para capturar condiciones de humedad y cobertura vegetal.
6.  **Generación de Lags:** Crea variables de rezago temporales para todas las variables climáticas y ambientales, permitiendo capturar el efecto histórico del clima sobre la incidencia del dengue. El número de rezagos es un parámetro configurable.

## Requisitos de Entrada (Inputs)
El script espera encontrar los siguientes archivos en la carpeta `data/`:

*   `Base_deng.xlsx`: Datos de casos confirmados de dengue por municipio.
*   `Listados_DIVIPOLA.xlsx`: Metadatos geográficos y códigos oficiales del DANE.
*   `nino34.long.anom.csv`: Índices de anomalías climáticas para el fenómeno ENSO.
*   `climaticas_aedes_colombia.csv`: Datos satelitales de NDVI y Dew Point.
*   **Directorio `worldclim/`**: Debe contener los archivos `.tif` organizados por carpetas (`tmin`, `tmax`, `prec`, `elev`).

## Resultados (Outputs)
El script genera un archivo CSV optimizado para análisis:

*   **Ruta:** `data/dengue_data_v2_ocur.csv` (o `_res.csv` dependiendo de la configuración).
*   **Formato:** Delimitado por tuberías (`|`), codificación UTF-8.
*   **Contenido:** Columnas originales de ubicación, fecha, casos confirmados y las nuevas variables climáticas/ambientales con sus respectivos rezagos temporales (lags).

## Sistema de Logs
Para trazabilidad y depuración, el script reporta su actividad en:

*   **Archivo:** `project_logs.log` (ubicado en la raíz del proyecto).
*   **Detalle:** Cada ejecución registra timestamps, el inicio y fin de cada etapa, errores críticos y metadatos del dataset generado (como la última fecha de datos disponible).
