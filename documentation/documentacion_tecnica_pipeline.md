# Documentación Técnica del Pipeline de Preparación de Datos y Modelado Predictivo para la Clasificación del Riesgo Epidémico de Dengue

---

## 1. Visión General del Pipeline

El presente documento describe la arquitectura, las decisiones metodológicas y los fundamentos estadísticos del pipeline desarrollado para estimar el nivel de riesgo epidémico de dengue a escala municipal en Colombia. El riesgo epidémico se define como la probabilidad de que los casos superen el patrón histórico esperado del municipio, a diferencia del riesgo de transmisión (que se refiere a la aptitud del territorio para sostener la transmisión del virus). El sistema está compuesto por cinco etapas secuenciales (S1 a S4, incluyendo S3.5), cada una con responsabilidades claramente delimitadas y archivos de salida trazables. El pipeline transforma registros crudos de vigilancia epidemiológica en un modelo de clasificación ordinal que predice tres niveles de riesgo — éxito, en rango y epidemia — a partir de variables climáticas, ambientales y de la dinámica histórica epidemiológica.

El flujo completo opera de la siguiente manera: S1 recolecta e integra las fuentes de datos; S2 imputa valores faltantes en la variable epidemiológica y construye la variable objetivo mediante el canal endémico; S3 diagnostica la calidad estadística de las variables predictoras; S3.5 aplica transformaciones de reducción de dimensionalidad y previene la fuga de información; y S4 entrena, optimiza y evalúa los modelos de clasificación utilizando una estrategia de validación cruzada espacio-temporal.

---

## 2. S1: Preparación y Enriquecimiento de Datos

### 2.1 Objetivo

S1 constituye la primera etapa del pipeline y tiene como finalidad consolidar los registros crudos de vigilancia epidemiológica del Sistema de Información de Salud y Protección Social (SISPRO) con fuentes biofísicas, climáticas y macroclimáticas para producir un dataset unificado a escala municipal-mensual.

### 2.2 Fuentes de datos integradas

El script integra cinco fuentes de información complementarias. La primera es la base de casos confirmados de dengue (incluyendo dengue grave) del SISPRO, con cobertura desde enero de 2007 hasta diciembre de 2024, organizada originalmente en formato ancho (meses como columnas) que se transforma a formato largo (tidy data) mediante una operación de tipo melt. La segunda fuente es el catálogo de División Político-Administrativa (DIVIPOLA) del Departamento Administrativo Nacional de Estadística (DANE), que proporciona coordenadas geográficas (latitud y longitud) y códigos municipales estandarizados para 1.060 municipios. La tercera fuente corresponde a los archivos raster de WorldClim, de los cuales se extraen tres variables climáticas mensuales — temperatura mínima, temperatura máxima y precipitación acumulada — con una resolución espacial aproximada de 1 km², así como la elevación sobre el nivel del mar como variable estática. La cuarta fuente son las anomalías de temperatura superficial del mar en la región Niño 3.4, que se utilizan para categorizar el fenómeno ENSO en cinco estados: Niño severo, Niño moderado, neutral, Niña moderada y Niña severa, siguiendo los umbrales establecidos por la Administración Nacional Oceánica y Atmosférica (NOAA). La quinta fuente incluye índices ambientales satelitales: el Índice de Vegetación de Diferencia Normalizada (NDVI), que captura la cobertura vegetal como proxy de disponibilidad de hábitat para el vector, y el punto de rocío (dew point) como indicador de humedad ambiental.

### 2.3 Normalización geográfica

La integración de fuentes con nomenclaturas distintas (SIVIGILA versus DANE) exigió un proceso de armonización de nombres. Todos los nombres de departamento y municipio se normalizan mediante la eliminación de acentos, conversión a minúsculas y remoción de prefijos numéricos comunes en la codificación DIVIPOLA. Se aplica además un diccionario de homologación manual para resolver discrepancias conocidas entre ambas fuentes, tales como la equivalencia entre "cali" y "santiago de cali" o entre "leguizamo" y "puerto leguizamo".

### 2.4 Expansión temporal y generación de rezagos

Para garantizar la consistencia de los análisis temporales, se construye una serie de tiempo continua mensual para cada municipio. Si un municipio no reporta datos en un mes determinado, se inserta un registro con valores nulos que será tratado por los métodos de imputación en la etapa posterior. Sobre esta serie continua, se generan variables de rezago (lags) temporales para cada variable climática y ambiental. El número de rezagos es un parámetro configurable; por defecto se utilizan tres meses, lo cual responde al intervalo documentado entre las condiciones climáticas favorables y la aparición de casos de dengue, que comprende el ciclo de incubación extrínseco del virus en el mosquito (10 a 14 días), el tiempo de generación de nuevas cohortes de vectores (2 a 4 semanas) y el período de incubación intrínseco en el humano (4 a 10 días).

### 2.5 Extracción de variables raster

La extracción de datos desde archivos raster se realiza mediante la biblioteca Rasterio. Para la elevación, se consulta una única vez la posición geográfica de cada municipio contra el archivo de elevación global de WorldClim. Para las variables climáticas dinámicas (temperatura y precipitación), se procesan los archivos mensuales correspondientes a cada combinación de año y mes dentro del rango temporal del dataset, incluyendo meses previos adicionales necesarios para calcular los rezagos sin introducir valores nulos artificiales.

### 2.6 Salida

El script produce un archivo CSV delimitado por tuberías (`|`), ubicado en `data/dengue_data_v2_ocur.csv`, que contiene la serie temporal completa con todas las variables base y sus rezagos.

---

## 3. S2: Imputación, Construcción del Canal Endémico y Feature Engineering

### 3.1 Objetivo

S2 cumple tres funciones: resolver los datos faltantes en la variable epidemiológica, construir la variable objetivo del modelo mediante el canal endémico, y generar las características de ingeniería de variables necesarias para el modelado.

### 3.2 Métodos de imputación del target

El dataset original presenta un 16.7% de valores faltantes en la serie temporal de casos confirmados, concentrados en municipios de categorías rurales dispersas y áreas con infraestructura de reporte limitada. Se implementaron cinco métodos de imputación diseñados para respetar la dinámica epidemiológica del dengue.

El método M1 (media histórica estacional) imputa cada valor faltante utilizando el promedio de los casos confirmados del mismo mes y municipio en los años anteriores, exigiendo un mínimo de cuatro años de historia disponible. El método M2 (mediana histórica estacional) replica la lógica de M1 pero utiliza la mediana en lugar de la media, lo que lo hace más robusto frente a brotes epidémicos atípicos que podrían sesgar el valor central. El método M3 (interpolación lineal) aplica una interpolación lineal dentro de la serie temporal de cada municipio, con un límite máximo de tres meses de brecha para evitar la fabricación de tendencias en periodos prolongados sin información. El método M4 (promedio móvil exponencial) utiliza un factor de suavizado alfa de 0.3 y requiere al menos 12 meses de datos previos, priorizando la tendencia reciente sobre la historia lejana. El método M5 (baseline) no aplica imputación alguna y elimina los registros nulos remanentes, sirviendo como grupo de control para evaluar si la ganancia en volumen de datos compensa el posible sesgo de imputación.

En todos los métodos se aplican dos reglas de consistencia: los valores calculados inferiores a 1 se redondean a cero (piso a cero) y los valores resultantes se redondean al entero más cercano, garantizando la interpretabilidad epidemiológica del dato como conteo de individuos.

### 3.3 Construcción del canal endémico

La variable objetivo del modelo no es el conteo de casos sino una categoría de riesgo ordinal derivada del canal endémico de Bortman, metodología adoptada por el Instituto Nacional de Salud de Colombia en sus Boletines Epidemiológicos Semanales. El canal endémico clasifica cada registro (municipio, mes, año) en función de la posición del valor observado de casos respecto a la media geométrica y el intervalo de confianza del 95% calculados sobre su propia distribución histórica.

El procedimiento opera de la siguiente manera. Para cada registro correspondiente a un municipio M, un año Y y un mes m, se seleccionan los valores de casos confirmados del mismo municipio y mes en los últimos siete años estrictamente anteriores a Y (ventana móvil de siete años). Esta restricción temporal es una medida deliberada de prevención de fuga de información (data leakage): los datos del año Y nunca participan en el cálculo de los umbrales que se utilizarán para clasificarlo.

Antes de calcular los umbrales, se aplica un filtro de exclusión de años epidémicos atípicos. Para cada municipio, se calcula el total anual de casos y se identifican como atípicos aquellos años cuyo total supera el límite superior del rango intercuartílico (Q3 + 1.5 × IQR). Esta exclusión evita que un brote excepcional infle los umbrales y haga que los límites sean demasiado permisivos en años normales.

Si después de la exclusión de outliers quedan al menos tres años con datos del mes m para ese municipio, se aplica la metodología de Bortman: (1) se transforman los valores con logaritmo natural: x_i = ln(casos_i + 1); (2) se calcula la media y desviación estándar de los logaritmos; (3) se obtiene la media geométrica MG = exp(x̄) - 1; (4) se calculan los límites del IC95%: LI = exp(x̄ - 1.96·s/√n) - 1 y LS = exp(x̄ + 1.96·s/√n) - 1. El valor actual se clasifica inicialmente en cuatro niveles (éxito, seguridad, alerta, epidemia) y posteriormente se colapsa a tres niveles operativos: éxito (casos < LI), en rango (LI ≤ casos < LS, fusionando seguridad y alerta) y epidemia (casos ≥ LS). La decisión de colapsar a tres clases se tomó tras verificar experimentalmente que las clases seguridad y alerta eran indistinguibles para los modelos de clasificación, con un F1-Score por clase de apenas 0.33 para alerta. La media geométrica (MG) se preserva como feature del modelo bajo el nombre `nivel_endemico`, proporcionando al modelo la escala de referencia del municipio. Los registros sin historia suficiente reciben un valor nulo y son excluidos del entrenamiento.

La decisión de utilizar una ventana de siete años responde a un equilibrio entre representatividad estadística y capacidad de adaptación. Una ventana más corta capturaría mejor los cambios recientes en la dinámica epidémica pero tendría mayor varianza en los umbrales; una ventana más larga sería más estable pero podría incluir datos de periodos con protocolos de reporte diferentes o dinámicas epidemiológicas ya superadas.

### 3.4 Sistema de checkpoints

La construcción del canal endémico es computacionalmente intensiva (aproximadamente una hora por método de imputación). Para evitar la pérdida de trabajo en caso de interrupciones, el script implementa un sistema de checkpoints: cada método se guarda a disco inmediatamente después de completar su canal endémico en la carpeta `data/processed/checkpoints/`. El parámetro `--skip_channel` permite re-ejecutar únicamente la fase de feature engineering sin recalcular los canales, lo cual es útil cuando se ajustan parámetros de rezago o transformaciones sin modificar la definición del target.

### 3.5 Ingeniería de características

Sobre cada dataset imputado y clasificado, se generan tres tipos de características adicionales. Primero, la codificación cíclica del mes: el mes (1 a 12) se transforma en dos variables mediante funciones trigonométricas (seno y coseno), resolviendo el problema de la discontinuidad del encoding entero donde diciembre (12) y enero (1) aparecen numéricamente distantes cuando son cronológicamente adyacentes. Segundo, los rezagos del conteo de casos (casos_lag1, casos_lag2, casos_lag3): estos capturan la magnitud histórica de la actividad epidémica y funcionan como features predictoras, no como target. Tercero, los rezagos del nivel de riesgo (riesgo_lag1, riesgo_lag2, riesgo_lag3): estos capturan la tendencia de la clasificación epidemiológica reciente.

Adicionalmente, se imputan los valores faltantes menores en las variables ambientales utilizando la media del municipio o, en su defecto, la media del departamento, para evitar que el proceso de limpieza final descarte registros donde el target fue recuperado exitosamente. Las variables categóricas del fenómeno ENSO se codifican mediante variables dummy con la técnica drop_first para prevenir multicolinealidad.

### 3.6 Salidas

S2 genera los siguientes archivos en `data/processed/`: cinco datasets finales (dengue_imputed_M1.csv hasta M5.csv), cada uno correspondiente a un método de imputación con la columna nivel_riesgo incluida; el archivo canal_endemico_log.csv con el detalle de cada clasificación (umbrales utilizados, años de historia, outliers excluidos) para auditoría; el archivo umbrales_canal_endemico.csv como referencia para el geovisor; el archivo municipios_coordenadas.csv con las combinaciones únicas de ubicación para restauración de metadatos en S4; el archivo cobertura_departamentos.csv con la representatividad por departamento; y los checkpoints intermedios en data/processed/checkpoints/.

---

## 4. S3: Diagnóstico de Variables Predictoras

### 4.1 Objetivo

S3 ejecuta un diagnóstico multidimensional sobre las variables predictoras de cada dataset generado por S2, con el fin de identificar redundancias, multicolinealidad, precedencia temporal y relevancia no lineal antes de que los datos lleguen al modelado.

### 4.2 Análisis de redundancia bivariada (correlación de Pearson)

Se calcula la matriz de correlación de Pearson entre todos los pares de variables predictoras. Los pares con coeficiente de correlación superior a 0.8 en valor absoluto se reportan como alerta de redundancia. Este umbral se justifica porque cuando dos variables comparten más del 64% de su varianza (r² > 0.64), mantener ambas incrementa la complejidad del modelo sin añadir poder predictivo significativo, y en modelos lineales genera inestabilidad en la estimación de los coeficientes.

### 4.3 Factor de inflación de la varianza (VIF)

El VIF se calcula como 1/(1 - R²ᵢ), donde R²ᵢ es el coeficiente de determinación obtenido al regresar la variable xᵢ contra todas las demás variables predictoras del conjunto. A diferencia de la correlación de Pearson (que solo evalúa pares), el VIF detecta la multicolinealidad múltiple: situaciones donde una variable puede reconstruirse casi perfectamente mediante una combinación lineal de varias otras. Un VIF superior a 10 indica que la varianza del coeficiente de esa variable está inflada en un factor de 10, lo cual compromete la fiabilidad de las estimaciones. Un VIF infinito señala singularidad de la matriz y exige la eliminación inmediata de la variable.

### 4.4 Prueba de causalidad de Granger

La prueba de Granger evalúa la hipótesis nula de que los valores pasados de una variable X no mejoran la predicción de la variable objetivo Y, comparando un modelo autorregresivo que solo usa rezagos de Y frente a otro que incorpora rezagos de X. Se prueba con rezagos de 1 a 6 meses y se reporta el p-valor mínimo. Un p-valor inferior a 0.05 indica que la variable tiene precedencia temporal validada, lo cual es particularmente relevante en el contexto del dengue dado que existe un desfase biológico documentado entre las condiciones climáticas y la aparición de casos.

### 4.5 Importancia por Random Forest

Se entrena un Random Forest con 100 estimadores para evaluar la importancia de cada variable mediante la reducción media de impureza (Mean Decrease in Impurity). Este análisis complementa los métodos lineales porque captura relaciones no lineales y de umbral que son frecuentes en la epidemiología del dengue (por ejemplo, la precipitación puede ser irrelevante por debajo de cierto nivel pero determinante por encima de él). Las variables con baja correlación lineal (Spearman inferior a 0.2) pero alta importancia en el Random Forest se reportan como candidatas con comportamiento no lineal.

### 4.6 Salida

S3 no genera archivos de datos modificados; su salida es exclusivamente diagnóstica y se registra en project_logs.log, incluyendo un bloque de recomendaciones compiladas que el analista utiliza para configurar las transformaciones de la etapa S3.5.

---

## 5. S3.5: Transformaciones Pre-modelado y Diagnóstico de Riesgo

### 5.1 Objetivo

S3.5 actúa como puente entre el diagnóstico (S3) y el modelado (S4). Aplica las transformaciones manuales definidas por el analista en respuesta a las alertas de S3, verifica la ausencia de fuga de información, y ejecuta un diagnóstico adaptado al target ordinal.

### 5.2 Transformaciones manuales (setup_pre_s4.JSON)

Las transformaciones se definen en un archivo de configuración JSON que soporta cuatro tipos de operaciones. La primera es la creación de promedios entre pares de variables: por ejemplo, la temperatura máxima y mínima se combinan en una temperatura promedio, eliminando la redundancia reportada por el diagnóstico de Pearson sin perder la señal térmica. La segunda es la eliminación selectiva de columnas: se descartan las variables identificadores de texto (departamento, municipio, fecha), la variable casosconfirmados (insumo directo del canal endémico cuya presencia como feature constituiría fuga de información) y los umbrales del canal endémico (umbral_p25, umbral_p50, umbral_p75) por la misma razón. La tercera operación es el renombramiento de columnas para estandarización. La cuarta es la reducción de dimensionalidad mediante Análisis de Componentes Principales (PCA): los grupos de variables con alta correlación interna (como los tres rezagos de dew_point, los tres rezagos de temperatura promedio o los tres rezagos de casos) se colapsan en un único componente principal que captura la mayor proporción de varianza del grupo. El PCA se aplica sobre datos previamente estandarizados con StandardScaler.

### 5.3 Prevención de fuga de información

El script implementa una capa de seguridad explícita contra data leakage. La función validate_leakage_prevention verifica que las columnas casosconfirmados, umbral_li y umbral_ls no estén presentes en el dataset final. Si alguna sobrevive a las transformaciones del JSON, se elimina automáticamente con una advertencia en el log. Esta protección es necesaria porque casosconfirmados es el insumo directo a partir del cual se calcula nivel_riesgo; su presencia como feature permitiría al modelo inferir la respuesta sin aprender las relaciones climáticas y ambientales que constituyen el objetivo del proyecto. La columna nivel_endemico (media geométrica histórica) sí se preserva intencionalmente como feature, ya que proporciona la escala de referencia del municipio sin revelar el target actual.

### 5.4 Diagnóstico adaptado al target ordinal

La clase DengueRiskAnalyst reemplaza el DengueFeatureAnalyst de S3 con tres adaptaciones. Primera, el Random Forest utilizado para evaluar la importancia de las variables es un clasificador (RandomForestClassifier) en lugar de un regresor, lo cual es apropiado para un target ordinal de cuatro clases. Segunda, se incorpora la prueba de Kruskal-Wallis para evaluar la separabilidad por clase: esta prueba no paramétrica determina si la distribución de cada variable predictora difiere significativamente entre al menos dos niveles de riesgo. Una variable con p-valor superior a 0.05 en Kruskal-Wallis no logra discriminar entre los niveles de riesgo, lo cual sugiere que su contribución al modelo será limitada. Tercera, la correlación de Spearman se mantiene porque es válida para evaluar la asociación monótona entre variables continuas y ordinales.

### 5.5 Salida

S3.5 genera en data/processed/ los archivos con sufijo _modified_ (por ejemplo, dengue_imputed_M1_modified_20260518_210000.csv), que son los únicos que S4 procesará para el entrenamiento. El diagnóstico completo se registra en project_logs.log incluyendo la distribución del target, el top 10 de features por importancia y las alertas de Kruskal-Wallis.

---

## 6. S4: Modelado y Validación Espacio-Temporal

### 6.1 Objetivo

S4 es la etapa final del pipeline y tiene como responsabilidad exclusiva el entrenamiento, la optimización de hiperparámetros y la evaluación de modelos de clasificación ordinal para predecir el nivel de riesgo epidémico de dengue.

### 6.2 Arquitecturas de modelos

Se evalúan cinco arquitecturas de clasificación. La regresión logística multinomial con regularización L2 sirve como baseline lineal; su formulación multinomial modela las probabilidades de cada clase simultáneamente, y el parámetro de regularización C controla el equilibrio entre ajuste y generalización. El Random Forest Classifier construye un ensamble de árboles de decisión que captura relaciones no lineales e interacciones entre variables sin requerir especificación explícita; la opción class_weight='balanced' ajusta automáticamente los pesos de las clases inversamente proporcional a su frecuencia para mitigar el desbalance entre niveles de riesgo. XGBoost Classifier implementa gradient boosting con árboles mediante la función objetivo multi:softmax, que optimiza directamente la clasificación multiclase; utiliza el método de construcción de árboles hist para eficiencia computacional y la métrica de evaluación mlogloss (log-loss multiclase). LightGBM Classifier utiliza una estrategia de crecimiento por hoja (leaf-wise) en lugar de por nivel, lo que le permite alcanzar menores errores con menor número de iteraciones en datasets grandes; también soporta class_weight='balanced'. CatBoost Classifier utiliza ordered boosting para reducir el sesgo de predicción y la función de pérdida MultiClass; su métrica de evaluación interna es TotalF1 con promedio macro, alineada con la métrica de selección del pipeline.

### 6.3 Estrategia de validación: SpatioTemporalSplit

La evaluación de modelos en problemas epidemiológicos espacio-temporales exige precauciones que la validación cruzada estándar (k-fold) no proporciona. Un k-fold aleatorio permitiría que datos de 2020 aparezcan en el entrenamiento mientras datos de 2018 se usan para validación, violando la causalidad temporal y produciendo métricas artificialmente optimistas.

La clase SpatioTemporalSplit implementa una validación cruzada que garantiza tres propiedades. Primera, la integridad temporal: cada fold utiliza años completos, nunca fracciones de año, para respetar la estacionalidad del dengue. Segunda, la causalidad: el conjunto de entrenamiento siempre contiene años cronológicamente anteriores al conjunto de validación. Tercera, la purga mediante un gap temporal de un año entre el último año de entrenamiento y el primer año de validación, lo cual previene la contaminación por autocorrelación temporal residual.

La configuración por defecto utiliza cinco folds, con tres años iniciales de entrenamiento, dos años de validación por fold y un año de gap. Adicionalmente, se reserva un conjunto de validación externa compuesto por los datos posteriores a 2020 (excluyendo 2024), que no participa en la validación cruzada y sirve como estimación independiente del desempeño en datos futuros.

### 6.4 Métricas de evaluación

La selección de métricas responde a la naturaleza ordinal del target y a los requisitos de interpretabilidad para salud pública.

El F1-score macro (métrica principal de selección) calcula el F1 de cada nivel de riesgo por separado y luego promedia sin ponderar por frecuencia. Esto obliga al modelo a desempeñarse bien en todas las clases, incluida la minoritaria (éxito, ~16%), que de otro modo sería ignorada por un modelo que maximice accuracy global prediciendo siempre las clases mayoritarias.

El F1-score ponderado calcula el mismo promedio pero ponderado por la frecuencia de cada clase, lo cual refleja el desempeño global del modelo respetando la distribución natural de los datos.

La exactitud (accuracy) mide la fracción de predicciones correctas. Es informativa pero insuficiente como métrica única en presencia de desbalance de clases: un modelo que prediga siempre éxito (la clase mayoritaria con más del 50% de los registros) obtendría una accuracy superior a 0.50 sin aportar valor predictivo real.

El MAE ordinal (Error Absoluto Medio Ordinal) es la métrica que distingue este problema de una clasificación nominal. Calcula la distancia promedio entre la categoría predicha y la real. Predecir éxito (0) cuando la realidad es epidemia (2) genera un error de 2, mientras que predecir en rango (1) cuando la realidad es epidemia (2) genera un error de 1. Esto refleja la realidad de salud pública: confundir un escenario de éxito con uno de epidemia tiene consecuencias operativas y sanitarias mucho más graves que confundir en rango con epidemia. Su rango va de 0 (predicción perfecta) a 2 (peor caso posible).

El Kappa de Cohen ponderado cuadráticamente mide la concordancia entre la clasificación predicha y la real, ajustada por la concordancia esperada por azar. La ponderación cuadrática penaliza más los desacuerdos grandes (éxito versus epidemia) que los pequeños (en rango versus epidemia), alineándose con la lógica ordinal del problema. Un kappa de 0 indica concordancia igual al azar; un kappa de 1 indica concordancia perfecta.

### 6.5 Búsqueda de hiperparámetros

La optimización se realiza mediante muestreo aleatorio (ParameterSampler) del espacio de hiperparámetros de cada modelo. El parámetro --n_iter controla el número de combinaciones evaluadas por modelo. Cada combinación se evalúa mediante la validación cruzada espacio-temporal completa y la validación externa, registrando todas las métricas en el log técnico. El mejor modelo se selecciona según el macro F1 promedio de la validación cruzada y se persiste a disco únicamente si su macro F1 en la validación externa supera el umbral de 0.55.

### 6.6 Desempeño regional

Para cada fold de la validación cruzada y para la validación externa, se calculan las métricas desglosadas por departamento. Este análisis es necesario porque el desempeño agregado puede enmascarar deficiencias territoriales: un modelo con macro F1 global de 0.65 podría estar prediciendo correctamente en departamentos con transmisión endémica estable pero fallando sistemáticamente en departamentos periféricos con series intermitentes. El reporte regional incluye la distribución de predicciones versus realidad para cada departamento, lo cual permite detectar sesgos sistemáticos (por ejemplo, si el modelo nunca predice epidemia en un departamento donde históricamente sí ocurre).

### 6.7 Modos de ejecución

S4 soporta dos modos de ejecución. El modo rápido (--fast) entrena un único modelo con parámetros base, sin búsqueda de hiperparámetros, y está diseñado para validación funcional rápida. El modo iterativo (por defecto) entrena los cinco modelos con búsqueda de hiperparámetros controlada por --n_iter.

### 6.8 Salidas

S4 genera los siguientes archivos en la carpeta processing/: experiment_technical_log.csv con el registro de cada iteración de entrenamiento incluyendo método de imputación, modelo, hiperparámetros, métricas de CV y métricas externas; regional_performance.csv con el desempeño desglosado por departamento; y archivos .pkl con los modelos serializados que superan el umbral de calidad, nombrados según el patrón {método}_{modelo}_best.pkl.

---

## 7. Instrucciones de Ejecución

### 7.1 Preparación del entorno

Se requiere Python 3.12 con las dependencias especificadas en requirements.txt:

```bash
pip install -r requirements.txt
```

### 7.2 Ejecución secuencial del pipeline

El pipeline se ejecuta en el siguiente orden estricto:

```bash
# Paso 1: Preparación de datos (solo si no se ha ejecutado previamente)
python S1_data_preparation_script.py

# Paso 2: Imputación, canal endémico y feature engineering
python S2_pre_processing_script.py --target_lags 3

# Paso 3: Diagnóstico de variables (opcional, solo genera logs)
python S3_feature_engineering.py

# Paso 4: Transformaciones pre-modelado y diagnóstico ordinal
python S3_5_testing_feature_engineering.py --moving_avg 3

# Paso 5: Entrenamiento — validación rápida
python S4_processing_script.py --fast xgb

# Paso 5 alternativo: Entrenamiento exhaustivo
python S4_processing_script.py --n_iter 10
```

### 7.3 Re-ejecución parcial con checkpoints

Si los canales endémicos ya fueron calculados y se desea modificar únicamente el feature engineering:

```bash
python S2_pre_processing_script.py --target_lags 3 --skip_channel
```

### 7.4 Parámetros disponibles

S2 acepta los siguientes parámetros: --target_lags (entero, por defecto 3) define el número de rezagos del target y de casos a generar; --history_years (entero, por defecto 7) establece la ventana de años para el cálculo del canal endémico; --no_exclude_outliers desactiva la exclusión de años epidémicos atípicos; y --skip_channel omite la construcción del canal endémico y utiliza checkpoints existentes.

S3.5 acepta: --moving_avg (entero, por defecto 0) que define la ventana de media móvil de casos como feature adicional.

S4 acepta: --fast (cadena, opciones: logistic, rf, xgb, lgbm, catboost) que activa el modo rápido para un modelo específico; y --n_iter (entero, por defecto 5) que define el número de combinaciones de hiperparámetros a evaluar por modelo en modo iterativo.

---

## 8. Archivos de Salida y Estructura del Proyecto

### 8.1 Descripción de archivos generados

El archivo data/dengue_data_v2_ocur.csv es el dataset consolidado producido por S1, conteniendo la serie temporal completa con variables climáticas, ambientales y sus rezagos, delimitado por tuberías.

Los archivos data/processed/dengue_imputed_M1.csv hasta M5.csv son los datasets finales de S2 con la columna nivel_riesgo (0-2: éxito, en rango, epidemia), la columna nivel_endemico (media geométrica histórica), variables cíclicas, rezagos de casos y rezagos de riesgo, listos para el diagnóstico de S3.5.

El archivo data/processed/canal_endemico_log.csv contiene el detalle de cada clasificación del canal endémico: municipio, año, mes, casos observados, umbrales Bortman (LI IC95%, media geométrica, LS IC95%) utilizados, nivel de riesgo asignado, número de años de historia empleados y número de años excluidos como outliers.

El archivo data/processed/umbrales_canal_endemico.csv almacena los umbrales del canal endémico del método M4 como referencia para el geovisor.

El archivo data/processed/municipios_coordenadas.csv contiene las combinaciones únicas de departamento, municipio, longitud y latitud, utilizado por S4 para restaurar los metadatos geográficos necesarios para el reporte regional.

Los archivos data/processed/checkpoints/checkpoint_M1_canal.csv hasta M5 son los checkpoints intermedios del canal endémico que permiten la re-ejecución parcial.

Los archivos data/processed/dengue_imputed_M*_modified_*.csv son los datasets transformados por S3.5 con PCA aplicado, variables de leakage eliminadas y diagnóstico ordinal ejecutado. Son los únicos archivos que S4 procesa.

El archivo processing/experiment_technical_log.csv registra cada iteración de entrenamiento con todas las métricas de clasificación, hiperparámetros y tiempo de ejecución.

El archivo processing/regional_performance.csv contiene el desempeño por departamento para cada fold y validación externa, incluyendo distribuciones de predicción versus realidad.

Los archivos processing/{método}_{modelo}_best.pkl son los modelos serializados que superaron el umbral de calidad y están listos para su uso en producción o en el geovisor.

El archivo project_logs.log contiene el historial cronológico completo de todas las etapas, incluyendo diagnósticos, alertas, matrices de confusión y metadatos de ejecución.

### 8.2 Estructura del proyecto

```
proyecto/
├── data/
│   ├── dengue_data_v2_ocur.csv                          # S1: dataset consolidado
│   ├── Base_deng.xlsx                                    # Entrada: casos SISPRO
│   ├── Listados_DIVIPOLA.xlsx                            # Entrada: coordenadas DANE
│   ├── nino34.long.anom.csv                              # Entrada: anomalías ENSO
│   ├── climaticas_aedes_colombia.csv                     # Entrada: NDVI y dew point
│   ├── worldclim/                                        # Entrada: archivos raster
│   └── processed/
│       ├── dengue_imputed_M1.csv ... M5.csv              # S2: datasets con nivel_riesgo
│       ├── dengue_imputed_M1_modified_*.csv ... M5       # S3.5: datasets transformados
│       ├── canal_endemico_log.csv                        # S2: trazabilidad del canal
│       ├── umbrales_canal_endemico.csv                   # S2: referencia para geovisor
│       ├── municipios_coordenadas.csv                    # S2: auxiliar geográfico
│       ├── cobertura_departamentos.csv                   # S2: representatividad
│       └── checkpoints/
│           └── checkpoint_M1_canal.csv ... M5            # S2: checkpoints intermedios
├── processing/
│   ├── experiment_technical_log.csv                      # S4: log de experimentos
│   ├── regional_performance.csv                          # S4: desempeño por depto
│   └── {método}_{modelo}_best.pkl                        # S4: modelos serializados
├── S1_data_preparation_script.py
├── S2_pre_processing_script.py
├── S3_feature_engineering.py
├── S3_5_testing_feature_engineering.py
├── S4_processing_script.py
├── setup_pre_s4.JSON
├── requirements.txt
└── project_logs.log                                      # Log general del proyecto
```
