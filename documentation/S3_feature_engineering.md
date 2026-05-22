# Documentación Técnica y Metodológica: S3_feature_engineering.py

## 1. Descripción General
El script `S3_feature_engineering.py` realiza un diagnóstico multidimensional de las variables predictoras. Aplica pruebas de redundancia, colinealidad, causalidad temporal y relevancia no lineal para asegurar que solo las variables con mayor valor informativo lleguen a la etapa de modelado. Al final de cada ejecución, genera un **Resumen Compilado** que facilita la toma de decisiones sobre el dataset.

**Nota:** Este script opera sobre el target ordinal `nivel_riesgo` (0=éxito, 1=en_rango, 2=epidemia). Las pruebas de separabilidad utilizan Kruskal-Wallis (no paramétrica) en lugar de correlación lineal con el target, dado que la relación entre las features y un target ordinal no es necesariamente lineal.

---

## 2. Profundización en Métodos Estadísticos

### A. Correlación de Pearson (Análisis de Redundancia Bivariada)
*   **Fundamento Matemático**: Mide el grado de relación lineal entre dos variables aleatorias continuas. Su valor oscila entre -1 y 1.
*   **¿Por qué se usa?**: Detecta **redundancia directa** entre pares de features. Si dos variables tienen correlación > 0.80, la varianza compartida es superior al 64%, indicando que una es mayormente un espejo de la otra.
*   **Umbral Crítico (0.8)**: Se seleccionó 0.8 como límite de alerta. Este umbral motivó la compresión PCA de las variables ENSO (correlaciones 0.80-0.84 entre lags consecutivos) y se aplica igualmente a los lags de NDVI en los métodos M3 y M5.

### B. Factor de Inflación de la Varianza (VIF - Multicolinealidad)
*   **Fundamento Matemático**: $VIF_i = \frac{1}{1 - R_i^2}$, donde $R_i^2$ es el coeficiente de determinación de la regresión de $x_i$ contra todas las demás variables.
*   **¿Por qué se usa?**: Detecta **multicolinealidad** donde una variable puede ser reconstruida mediante combinación lineal de múltiples otras variables.
*   **Escala de Interpretación**:
    *   **VIF = 1**: Ausencia de colinealidad.
    *   **5 < VIF < 10**: Colinealidad moderada. Aceptable para modelos de árbol.
    *   **VIF > 10**: Colinealidad severa.
    *   **VIF = inf**: Colinealidad perfecta. Acción obligatoria: eliminar.

### C. Prueba de Causalidad de Granger (Precedencia Temporal)
*   **Fundamento Matemático**: Basada en modelos de autorregresión vectorial (VAR). Evalúa si los valores pasados de $X$ mejoran la predicción de $Y$ respecto a usar solo los rezagos de $Y$.
*   **¿Por qué se usa?**: Valida si los rezagos (lags) de las variables ambientales tienen una conexión estadística de precedencia temporal con el nivel de riesgo epidémico. En el dengue, existe un retraso biológico entre las condiciones climáticas y la aparición de casos.
*   **Interpretación**:
    *   **P-valor < 0.05**: Precedencia temporal validada. Variable "estrella" para el pronóstico.
    *   **P-valor > 0.05**: Sin evidencia de que la variable ayude a predecir el riesgo futuro más allá de la propia historia del target.

### D. Prueba de Kruskal-Wallis (Separabilidad entre Clases)
*   **Fundamento**: Test no paramétrico que evalúa si las distribuciones de una variable difieren significativamente entre los 3 niveles de riesgo (éxito, en_rango, epidemia).
*   **¿Por qué se usa?**: Reemplaza la correlación lineal con el target (apropiada para regresión) por una prueba de separabilidad entre grupos ordinales.
*   **Interpretación**:
    *   **P-valor < 0.05**: La variable discrimina entre niveles de riesgo. Útil para la clasificación.
    *   **P-valor > 0.05**: "Sin poder discriminativo". La variable no distingue entre los grupos. Se reporta como alerta pero no se elimina automáticamente (puede ser útil en interacciones).

### E. Importancia por Random Forest Classifier (Relevancia No Lineal)
*   **Fundamento**: Utiliza *Mean Decrease in Impurity* (Gini Importance) de un Random Forest Classifier entrenado para clasificar `nivel_riesgo`.
*   **¿Por qué se usa?**: Captura relaciones no lineales y de umbral que los métodos lineales no detectan. El dengue presenta comportamientos de umbral donde una variable solo importa al superar cierto valor.
*   **Hallazgo clave**: `nivel_endemico` consistentemente ocupa el primer lugar (~25-30% de importancia), confirmando que la escala de referencia del municipio es la variable más informativa para la clasificación.

---

## 3. Guía de Lectura del Resumen Compilado

### Sección 1: Distribución del Target
Muestra la distribución de las 3 clases de `nivel_riesgo`. La distribución esperada para M4 es: éxito ~15%, en_rango ~39%, epidemia ~46%.

### Sección 2: Resumen de Redundancia Lineal (Pearson)
Pares de variables con correlación > 0.80. Tras la compresión PCA de ENSO, las únicas alertas restantes son los lags de NDVI en M3 y M5.

### Sección 3: Resumen de Diagnóstico por Atributo
Lista cada variable con sus resultados de Granger y Kruskal-Wallis.
-   **`GRANGER validado`**: Precedencia temporal confirmada. Variable valiosa para la predicción.
-   **`SIN PODER DISCRIMINATIVO`**: No distingue entre niveles de riesgo por sí sola (KW p>0.05).

### Sección 4: Top 10 Features por Importancia
Ranking de importancia del Random Forest Classifier. Orden esperado: nivel_endemico > pca_riesgo_lags > pca_casos_lags > pca_dew_point > NDVI > pca_enso.

---

## 4. Estrategia de Silenciamiento de Warnings
El script aplica tres capas de limpieza para garantizar legibilidad:
1.  **Filtro Global**: Ignora FutureWarnings.
2.  **Anulación de Warn**: Redirige `warnings.warn` para evitar impresiones de `statsmodels`.
3.  **Gestión de NumPy**: `np.seterr(all='ignore')` para errores de división por cero en VIF.
