# Documentación Metodológica: Pipeline de Preparación de Datos para la Clasificación del Riesgo Epidémico de Dengue

## 1. Descripción de la fuente de datos

La base de datos fundamental para este proyecto de modelado predictivo se sustenta en tres pilares de información oficial del Estado colombiano y fuentes biofísicas globales. El núcleo epidemiológico proviene del Sistema de Información de Salud y Protección Social (SISPRO) del Ministerio de Salud, abarcando un registro histórico consolidado desde enero de 2007 hasta diciembre de 2024. La granularidad de la información es mensual, agregada por municipio de ocurrencia del evento, lo cual permite capturar la dinámica epidémica en el punto geográfico donde se produce la infección, factor crítico para intervenciones de salud pública dirigidas.

El dataset integra variables de tres dominios distintos:
1. **Dominio Epidemiológico (Insumo para el Target):** La variable de casos confirmados de dengue (incluyendo dengue grave) se utiliza como insumo para construir la variable objetivo: el nivel de riesgo epidémico, derivado del canal endémico de Bortman. El riesgo epidémico mide la probabilidad de que los casos superen el patrón histórico esperado del municipio, a diferencia del riesgo de transmisión (que mide la aptitud del territorio para sostener la transmisión del virus).
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
El método EWM prioriza la tendencia reciente sobre la historia lejana, aplicando un factor de suavizado $\alpha = 0.3$. Requiere al menos 12 meses de datos previos. Metodológicamente, es el enfoque más "epidemiológico" en términos de brote, ya que reconoce que el riesgo actual depende fuertemente de la masa crítica de infectados en los meses inmediatamente anteriores, permitiendo que el valor imputado se adapte a cambios estructurales en la dinámica epidémica. M4 produjo consistentemente los mejores resultados en todos los modelos de clasificación, lo cual es coherente con la naturaleza temporal del dengue.

### M5: Baseline Experimental
Consiste en la eliminación estricta de nulos (listwise deletion). Sirve como control para evaluar si la ganancia en volumen de datos obtenida con M1-M4 compensa la posible introducción de sesgo de imputación.

### Reglas de consistencia (Floor-to-Zero y Redondeo)
En todos los métodos, se aplicó una regla de "piso a cero" para valores calculados inferiores a 1 y un redondeo al entero más cercano. Esto garantiza la interpretabilidad epidemiológica del dato: los casos son conteos de individuos, por lo cual no existen fracciones de contagio ni casos negativos.

## 4. Construcción del canal endémico de Bortman y la variable objetivo

### 4.1 Metodología
La variable objetivo del modelo no es el conteo de casos sino el **nivel de riesgo epidémico**, una categoría ordinal derivada del canal endémico de Bortman (medias geométricas con intervalo de confianza del 95%). Esta metodología fue adoptada por el INS de Colombia para la vigilancia de dengue y se verifica su uso en los Boletines Epidemiológicos Semanales (BES).

Para cada combinación municipio-mes, el canal se calcula con la siguiente secuencia:
1. **Transformación logarítmica**: $x_i = \ln(casos_i + 1)$
2. **Media y desviación estándar** de los logaritmos: $\bar{x}$ y $s$
3. **Media geométrica**: $MG = e^{\bar{x}} - 1$
4. **Límite inferior IC95%**: $LI = e^{\bar{x} - 1.96 \cdot s/\sqrt{n}} - 1$
5. **Límite superior IC95%**: $LS = e^{\bar{x} + 1.96 \cdot s/\sqrt{n}} - 1$

### 4.2 Clasificación en 3 niveles operativos
Inicialmente se implementaron 4 niveles (éxito, seguridad, alerta, epidemia). La evaluación experimental demostró que seguridad y alerta eran indistinguibles para los clasificadores (F1 por clase de 0.33 para alerta), por lo que se colapsaron en una sola clase:
- **Éxito (0):** Casos < LI. El municipio reporta menos casos de lo esperado.
- **En rango (1):** LI ≤ Casos < LS. Dentro del comportamiento histórico esperado.
- **Epidemia (2):** Casos ≥ LS. Supera significativamente su patrón histórico.

### 4.3 Prevención de data leakage
Se implementaron cuatro mecanismos: (1) solo datos de años estrictamente anteriores al año de clasificación; (2) ventana móvil de 7 años; (3) exclusión de años epidémicos atípicos por IQR; (4) mínimo de 3 años de historia para calcular umbrales.

### 4.4 Nivel endémico como feature
La media geométrica (MG) se preserva como feature del modelo bajo el nombre `nivel_endemico`, proporcionando la escala de referencia del municipio. Esta fue la decisión de diseño con mayor impacto: el Macro F1-Score subió de 0.468 a 0.668 (+20 puntos) al incluir esta variable, ya que resuelve la ambigüedad fundamental del problema ("10 casos es mucho para este municipio o poco para este otro").

## 5. Transformaciones aplicadas en el preprocesamiento

El pipeline ejecuta una secuencia de transformaciones diseñada para optimizar la capacidad de aprendizaje de los modelos:

1. **Codificación Cíclica del Tiempo:** El mes (1-12) se transforma mediante funciones trigonométricas ($\sin$ y $\cos$), resolviendo la discontinuidad entre diciembre (12) y enero (1).
2. **Construcción de Rezagos (Lags) Autoregresivos:** Se generan variables de retardo para casos confirmados y nivel de riesgo (1 a 3 meses), así como para variables ambientales (hasta 3 meses). Los rezagos permiten la anticipación temporal de 1 a 3 meses que plantea la pregunta problema.
3. **Compresión PCA:** Se aplica Análisis de Componentes Principales sobre 5 grupos de variables con alta correlación interna: punto de rocío (4→1), casos_lags (3→1), riesgo_lags (3→1), temperatura promedio (4→1) y ENSO (16→2). Esto reduce el dataset de 35 a 22 columnas, eliminando redundancia sin perder señal.
4. **Imputación Ambiental Selectiva:** Para evitar que la limpieza final descarte filas donde el target fue recuperado exitosamente, se imputan los nulos menores en variables climáticas usando la media del municipio o departamento.
5. **Orden de Operaciones y Prevención de Data Leakage:** El pipeline sigue un orden estricto: primero la imputación del target, luego el canal endémico (solo con datos pasados), después los lags, y finalmente la limpieza.

## 6. Resultados de Experimentación y Modelado

La fase de experimentación evaluó cinco arquitecturas de clasificación supervisada (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) sobre los datasets generados por los cinco métodos de imputación. El diseño experimental utilizó TimeSeriesSplit (5 folds) para validación cruzada y validación temporal hold-out (año 2024) para evaluación final.

### Desempeño Comparativo de Modelos (Dataset M4 — Mejor método)

| Modelo | Macro F1 CV | Macro F1 Val | Exactitud Val | MAE Ordinal |
|:---|:---:|:---:|:---:|:---:|
| **LightGBM** | 0.668 | **0.729** | 0.739 | 0.275 |
| **XGBoost** | 0.633 | 0.727 | 0.731 | 0.282 |
| **Random Forest** | 0.666 | 0.722 | 0.717 | 0.312 |
| **CatBoost** | 0.666 | 0.704 | 0.716 | 0.311 |
| **Logistic (baseline)** | 0.486 | 0.595 | 0.618 | 0.400 |

### Métricas por Clase — Mejor Modelo (LightGBM M4)

| Clase | Precision | Sensibilidad | F1-Score | Tasa FP |
|:---|:---:|:---:|:---:|:---:|
| Éxito | 0.571 | 0.697 | 0.627 | 0.112 |
| En rango | 0.674 | 0.728 | 0.700 | 0.231 |
| Epidemia | **0.917** | **0.766** | **0.835** | 0.052 |

### Perfil como Sistema de Alerta Epidémica (epidemia vs. no epidemia)

| Métrica | Valor | Interpretación |
|:---|:---:|:---|
| Sensibilidad | 76.6% | Detecta 77 de cada 100 epidemias reales |
| Especificidad | 94.8% | Descarta correctamente 95 de cada 100 no-epidemias |
| Valor Predictivo Positivo | 91.7% | Cuando alerta epidemia, acierta 92% de las veces |
| Valor Predictivo Negativo | 84.4% | Cuando dice "no epidemia", acierta 84% de las veces |
| Error catastrófico | 1.4% | Solo 1.4% de predicciones confunden éxito con epidemia |

## 7. Consideraciones y limitaciones metodológicas

Este pipeline opera bajo el supuesto de que el sistema de vigilancia (SISPRO/SIVIGILA) ha mantenido una definición de caso relativamente estable. Cambios mayores en los criterios de diagnóstico podrían introducir rupturas en la serie que los métodos de media histórica (M1/M2) no pueden detectar.

El modelo es viable para producción como sistema de apoyo a la decisión, no como sistema autónomo. Se recomienda un período de validación prospectiva de 3-6 meses en paralelo con la vigilancia existente antes de tomar decisiones operativas basadas en sus predicciones. Los municipios clasificados como epidemia se priorizan con alta confianza (VPP=91.7%); los clasificados como éxito o en rango deben cruzarse con la vigilancia rutinaria del SIVIGILA.
