# Documentación: Preprocesamiento, Canal Endémico e Imputación de Datos (S2)

## Descripción General
El script `S2_pre_processing_script.py` constituye la segunda fase del pipeline. Su función principal es resolver el problema de los datos faltantes en la variable epidemiológica (casos de dengue) mediante cinco métodos estadísticos, construir la variable objetivo (nivel de riesgo epidémico) a través del canal endémico de Bortman, generar características temporales y evaluar la representatividad geográfica de los datos recuperados.

## Funcionamiento del Script
El script opera a través de un pipeline de cinco componentes principales:

### 1. Métodos de Imputación (M1 - M5)
Se generan cinco datasets independientes para comparar diferentes estrategias de recuperación de datos:
*   **M1 (Media Histórica):** Imputa usando el promedio de casos para el mismo mes y municipio en años anteriores (requiere al menos 4 años de historia).
*   **M2 (Mediana Histórica):** Similar a M1, pero utiliza la mediana para mayor robustez ante brotes atípicos.
*   **M3 (Interpolación Lineal):** Rellena huecos temporales mediante una pendiente lineal entre puntos conocidos (límite de brecha: 3 meses).
*   **M4 (EWM - Exponential Weighted Mean):** Utiliza un promedio móvil exponencial con un factor de suavizado ($\alpha=0.3$) para capturar tendencias recientes.
*   **M5 (Baseline):** No aplica imputación; solo elimina los registros que permanecen nulos tras el procesamiento.

### 2. Canal Endémico de Bortman
Para cada dataset imputado, se construye el canal endémico usando la metodología de medias geométricas con IC 95%, consistente con los Boletines Epidemiológicos del INS:
*   **Cálculo:** Transformación logarítmica → media y desviación estándar → media geométrica (MG) → límites IC95% (LI, LS).
*   **Prevención de leakage:** Solo datos de años estrictamente anteriores, ventana de 7 años, exclusión de outliers IQR, mínimo 3 años.
*   **Clasificación:** Se asigna nivel de riesgo en 4 niveles (éxito, seguridad, alerta, epidemia) y se colapsa a 3 niveles operativos: éxito (< LI), en rango (LI ≤ x < LS) y epidemia (≥ LS).
*   **Regla de consistencia:** Si casosconfirmados = 0, siempre se clasifica como éxito (0), independientemente de los umbrales Bortman. Esto previene el caso donde municipios con historial de 0 casos (LI=MG=LS=0) se clasificaban erróneamente como epidemia por la condición `0 >= 0`.
*   **nivel_endemico:** La media geométrica (MG) se preserva como feature del modelo, proporcionando la escala de referencia del municipio.

### 3. Ingeniería de Características (Feature Engineering)
Para cada dataset imputado y clasificado, se aplican las siguientes transformaciones:
*   **Variables Cíclicas:** Transformación del mes en coordenadas `mes_sin` y `mes_cos` para que el modelo entienda la naturaleza circular del año.
*   **Lags del Target:** Creación de rezagos configurables para `casosconfirmados` y `nivel_riesgo` (por defecto 3 meses).
*   **Etiqueta textual:** Se agrega columna `nivel_riesgo_label` con los nombres textuales de cada nivel (éxito, en_rango, epidemia).
*   **Recuperación Ambiental:** Imputación de variables climáticas faltantes usando la media local para evitar la pérdida de registros recuperados por la imputación del target.

### 4. Sistema de Checkpoints
La construcción del canal endémico es computacionalmente intensiva (~1 hora por método). El script implementa un sistema de checkpoints que guarda cada método inmediatamente después de completar su canal endémico en `data/processed/checkpoints/`. El parámetro `--skip_channel` permite reutilizar checkpoints existentes y re-ejecutar solo el feature engineering.

### 5. Reporte de Cobertura Geográfica
Se evalúa cada departamento para verificar si posee suficientes datos tras la imputación. El reporte permite monitorizar la confianza en la representatividad territorial de los datos.

## Parámetros de Ejecución
```bash
python S2_pre_processing_script.py --target_lags 3
python S2_pre_processing_script.py --skip_channel   # Reutiliza checkpoints
```

| Parámetro | Tipo | Por Defecto | Descripción |
| :--- | :--- | :--- | :--- |
| `--target_lags` | `int` | `3` | Cantidad de meses de historia de la variable objetivo a generar como predictores. |
| `--skip_channel` | `flag` | `False` | Si se activa, omite el cálculo del canal endémico y carga los checkpoints existentes. |

## Requisitos de Entrada (Inputs)
*   **Archivo:** `data/dengue_data_v2_ocur.csv` generado por S1.
*   **Estructura:** Debe contener columnas de ubicación (`departamento`, `municipio`), tiempo (`year`, `mes`) y las variables climáticas base con sus lags pre-calculados.

## Resultados (Outputs)
Los resultados se almacenan en `data/processed/`:
*   `dengue_imputed_M1.csv` hasta `M5.csv`: Datasets con nivel_riesgo (0-2), nivel_riesgo_label, nivel_endemico, lags de casos y riesgo.
*   `checkpoints/checkpoint_M1_canal.csv` hasta `M5`: Checkpoints del canal endémico.
*   `umbrales_canal_endemico.csv`: Detalle de umbrales Bortman (LI, MG, LS) por municipio-mes.
*   `canal_endemico_log.csv`: Log de cada clasificación del canal.
*   `municipios_coordenadas.csv`: Archivo auxiliar de coordenadas para restauración de metadatos en S4.
*   `cobertura_departamentos.csv`: Reporte de recuperación de datos por departamento.
