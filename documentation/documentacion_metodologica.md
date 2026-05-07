# Documentación Metodológica: Pipeline de Preparación de Datos para la Predicción de Dengue

## 1. Descripción de la fuente de datos

La base de datos fundamental para este proyecto de modelado predictivo se sustenta en tres pilares de información oficial del Estado colombiano y fuentes biofísicas globales. El núcleo epidemiológico proviene del Sistema de Información de Salud y Protección Social (SISPRO) del Ministerio de Salud, abarcando un registro histórico consolidado desde enero de 2007 hasta diciembre de 2024. La granularidad de la información es mensual, agregada por municipio de ocurrencia del evento, lo cual permite capturar la dinámica de transmisión en el punto geográfico donde se produce la infección, factor crítico para intervenciones de salud pública dirigidas.

El dataset integra variables de tres dominios distintos:
1. **Dominio Epidemiológico (Target):** La variable dependiente principal es el conteo de casos confirmados de dengue (incluyendo dengue grave). Representa la carga de morbilidad reportada en el sistema de vigilancia.
2. **Dominio Geográfico e Identificación:** Se utiliza la codificación DIVIPOLA del Departamento Administrativo Nacional de Estadística (DANE) para estandarizar la nomenclatura de 1,060 municipios y 33 departamentos. Se incluyen coordenadas geográficas (latitud, longitud) y elevación sobre el nivel del mar, variable esta última fundamental dado que el ciclo biológico del vector *Aedes aegypti* está limitado térmicamente por la altitud.
3. **Dominio Ambiental y Climático (Features):** Se incorporan series temporales de precipitación acumulada, temperaturas máximas y mínimas (fuente WorldClim), así como el Índice de Vegetación de Diferencia Normalizada (NDVI) y el punto de rocío (humedad). Estas variables actúan como determinantes ecológicos que condicionan la disponibilidad de criaderos y la tasa de replicación viral dentro del mosquito.

A pesar de la robustez de las fuentes, el dato crudo presenta limitaciones inherentes a los sistemas de vigilancia pasiva, tales como el subregistro en zonas de conflicto o difícil acceso, y discontinuidades temporales en el reporte de municipios con baja densidad poblacional, lo que justifica la necesidad de un pipeline de imputación y estabilización de series temporales.

## 2. Caracterización de la calidad del dato

La calidad de la información fue evaluada mediante un análisis de completitud exhaustivo. Antes de la intervención del pipeline, el dataset original de ocurrencia presentaba un 16.7% de valores faltantes en la serie temporal de casos confirmados. Estos faltantes no se distribuyen de forma aleatoria (Missing at Random, MAR), sino que se concentran en municipios de categorías rurales dispersas y áreas no municipalizadas de departamentos como Amazonas, Vaupés y Guainía.

El análisis de la cobertura por departamento revela que regiones con infraestructuras de salud más consolidadas, como Antioquia o Valle del Cauca, mantienen series casi ininterrumpidas (con nulos inferiores al 5%), mientras que departamentos periféricos muestran "gaps" de información que coinciden con periodos de reestructuración administrativa o cambios en los protocolos de reporte de SIVIGILA. Un hallazgo crítico fue la identificación de nulos artificiales generados durante la expansión de la serie temporal (resampling), donde columnas constantes como el país o identificadores geográficos no se propagaban correctamente, problema que fue corregido en la lógica de preprocesamiento para evitar la pérdida sistémica de información en el proceso de limpieza final.

## 3. Estrategias de imputación del target

Dada la naturaleza estacional y la fuerte autocorrelación temporal del dengue, se rechazó el uso de imputaciones genéricas (como la media global o KNN simple) que ignoran la dinámica epidemiológica. En su lugar, se implementaron cinco métodos (M1 a M5) diseñados para capturar diferentes dimensiones del fenómeno:

### M1 y M2: Imputación por Estacionalidad Histórica (Media y Mediana)
Estos métodos asumen que el dengue responde a ciclos anuales predecibles vinculados a temporadas de lluvia o sequía. Para cada valor nulo en un mes $m$ de un año $y$, se calcula el valor central (media en M1, mediana en M2) de ese mismo mes en los años precedentes ($y-n$). Se exige un umbral mínimo de cuatro años de historia disponible para garantizar representatividad. La mediana (M2) se considera más robusta frente a brotes epidémicos atípicos que podrían sesgar la media. El supuesto implícito es la estabilidad del ciclo biológico del vector en el tiempo.

### M3: Interpolación Lineal Intra-anual
Este método aborda la continuidad local de la serie. Se aplica una interpolación lineal dentro de un municipio si el vacío de información no supera los tres meses. Epidemiológicamente, esto asume que la transición de una carga viral entre dos periodos cercanos es gradual. Es particularmente efectivo para corregir fallos puntuales en el reporte mensual sin introducir ruido histórico de años lejanos.

### M4: Promedio Móvil Exponencial (EWM)
El método EWM prioriza la tendencia reciente sobre la historia lejana, aplicando un factor de suavizado $\alpha = 0.3$. Requiere al menos 12 meses de datos previos. Metodológicamente, es el enfoque más "epidemiológico" en términos de brote, ya que reconoce que el riesgo actual depende fuertemente de la masa crítica de infectados en los meses inmediatamente anteriores, permitiendo que el valor imputado se adapte a cambios estructurales en la transmisión.

### M5: Baseline Experimental
Consiste en la eliminación estricta de nulos (listwise deletion). Sirve como control para evaluar si la ganancia en volumen de datos obtenida con M1-M4 compensa la posible introducción de sesgo de imputación.

### Reglas de consistencia (Floor-to-Zero y Redondeo)
En todos los métodos, se aplicó una regla de "piso a cero" para valores calculados inferiores a 1 y un redondeo al entero más cercano. Esto garantiza la interpretabilidad epidemiológica del dato: los casos son conteos de individuos, por lo cual no existen fracciones de contagio ni casos negativos.

## 4. Transformaciones aplicadas en el preprocesamiento

El pipeline ejecuta una secuencia de transformaciones diseñada para optimizar la capacidad de aprendizaje de los modelos matemáticos:

1. **Codificación Cíclica del Tiempo:** El mes (1-12) se transforma mediante funciones trigonométricas ($\sin$ y $\cos$). Esta decisión metodológica resuelve el problema de la discontinuidad del encoding entero, donde diciembre (12) y enero (1) aparecen numéricamente distantes cuando cronológicamente son adyacentes y comparten condiciones climáticas de fin de año.
2. **Construcción de Rezagos (Lags) Autoregresivos:** Se generan variables de retardo para la variable objetivo (1 a 3 meses) y variables ambientales (hasta 8 meses). El rezago de 1-3 meses en los casos responde al tiempo de incubación intrínseca y extrínseca, así como a la generación de nuevos mosquitos infectados a partir de los actuales. Los rezagos climáticos más largos (8 meses) capturan el impacto de fenómenos macroclimáticos como El Niño (ENSO) sobre el ecosistema.
3. **Imputación Ambiental Selectiva:** Para evitar que el `dropna()` final descarte filas donde el target fue recuperado exitosamente, se imputan los nulos menores en variables climáticas usando la media del municipio o departamento. Esto maximiza el tamaño de la muestra efectiva sin alterar significativamente la señal climática regional.
4. **Orden de Operaciones y Prevención de Data Leakage:** El pipeline sigue un orden estricto: primero la imputación del target (basada solo en historia pasada o local), luego la creación de lags, y finalmente la limpieza. El escalado de variables se reserva para el momento previo al modelado para asegurar que las estadísticas de normalización no contaminen los conjuntos de validación.

## 5. Cobertura geográfica final

La ejecución del pipeline arroja resultados diferenciados en términos de representatividad territorial. El método M5 (sin imputación) limita el dataset a 151,140 registros, excluyendo sistemáticamente municipios con reportes intermitentes. En contraste, el método M4 (EWM) logra la mayor recuperación de datos, alcanzando 181,836 registros efectivos.

Esta diferencia no es solo cuantitativa sino geográfica. Al utilizar M4 o M1/M2, se logra la inclusión de departamentos periféricos que de otro modo quedarían excluidos de los modelos nacionales. Sin embargo, existe un riesgo inherente de sesgo: los departamentos con peor infraestructura de reporte son precisamente aquellos donde los métodos de imputación deben "imaginar" más datos. El reporte `cobertura_departamentos.csv` permite monitorizar esta confianza; municipios con una "razón de exclusión" persistente a pesar de la imputación son marcados como inviables para el análisis debido a una carencia estructural de información base.

## 6. Resultados de Experimentación y Modelado

La fase de experimentación, implementada mediante el motor `main_v2.py`, evaluó seis arquitecturas de aprendizaje supervisado sobre los datasets generados por los cinco métodos de imputación. El diseño experimental utilizó una partición temporal (Time-Series Split) del 85% para entrenamiento y 15% para prueba, asegurando que el modelo fuera evaluado sobre datos cronológicamente posteriores a los de entrenamiento.

### Desempeño Comparativo de Modelos
El desempeño de las arquitecturas evaluadas muestra una clara superioridad de los métodos no lineales. A continuación, se resumen las métricas globales para el modelo Random Forest bajo los diferentes escenarios de imputación:

| Método de Imputación | Train R² | Test R² | Test RMSE | MAE | Registros Finales |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **M1 (Media Hist.)** | 0.959 | 0.791 | 53.05 | 6.41 | 167,052 |
| **M2 (Mediana Hist.)** | 0.960 | 0.794 | 52.63 | 6.39 | 167,052 |
| **M3 (Interpolación)** | 0.959 | 0.805 | 53.02 | 6.68 | 155,019 |
| **M4 (EWM)** | 0.960 | 0.799 | 49.91 | 6.00 | 181,836 |
| **M5 (Baseline)** | 0.959 | 0.807 | 53.50 | 6.77 | 151,140 |

### Impacto Geográfico de la Imputación
La efectividad de la recuperación de datos varía según la infraestructura de reporte regional. La siguiente tabla detalla el impacto de los métodos M1 (estacionalidad rígida) y M4 (adaptativo) en una selección de departamentos críticos:

| Departamento | Total Meses | Nulos Originales | Recuperados (M1) | Recuperados (M4) | % Recuperación (M4) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Antioquia** | 23,400 | 4,824 | 2,988 | 4,824 | 100% |
| **Cundinamarca** | 19,512 | 5,256 | 2,796 | 5,256 | 100% |
| **Santander** | 17,964 | 3,444 | 2,412 | 3,444 | 100% |
| **Nariño** | 8,208 | 3,132 | 996 | 3,132 | 100% |
| **Amazonas** | 1,584 | 708 | 48 | 708 | 100% |
| **Vaupés** | 744 | 348 | 72 | 348 | 100% |
| **Bogotá, D.C.** | 84 | 36 | 0 | 36 | 100% |

Como se observa, el método M4 (EWM) logra una recuperación total de los vacíos temporales siempre que exista una base mínima de 12 meses de historia, mientras que M1 es mucho más restrictivo al exigir 4 años del mismo mes estacional. Esta diferencia es vital en departamentos como Amazonas o Vaupés, donde M1 solo recupera una fracción mínima del dato, limitando la capacidad del modelo para entender la dinámica en zonas selváticas.

### Heterogeneidad Espacial de la Predicción
El análisis granular por departamento revela una variabilidad espacial significativa en la capacidad predictiva del modelo. Departamentos como **Tolima** y **Sucre** alcanzan niveles de precisión excepcionales ($R^2 > 0.88$ con Random Forest), lo que sugiere que en estas regiones la relación entre variables ambientales y casos de dengue es altamente estructurada y predecible. En contraste, otras zonas presentan métricas mucho más modestas ($R^2 \approx 0.20$), lo cual puede ser indicativo de factores locales no capturados por el modelo, como la efectividad de los programas locales de control de vectores o variaciones en la calidad del reporte epidemiológico.

## 7. Consideraciones y limitaciones metodológicas

Este pipeline opera bajo el supuesto de que el sistema de vigilancia (SISPRO/SIVIGILA) ha mantenido una definición de caso relativamente estable. Cambios mayores en los criterios de diagnóstico (ej. paso de diagnóstico clínico a confirmación por laboratorio) podrían introducir rupturas en la serie que los métodos de media histórica (M1/M2) no pueden detectar.

Asimismo, la imputación ambiental asume que la media del municipio es un buen estimador para datos faltantes de temperatura o NDVI, lo cual es aceptable en climas tropicales estables pero podría ocultar eventos extremos puntuales. Se recomienda que los resultados derivados de este pipeline sean validados cualitativamente con expertos territoriales de salud pública, especialmente en aquellos departamentos donde la tasa de imputación superó el 20% del total de la serie histórica. 
