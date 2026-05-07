# Documentación Técnica y Metodológica: S3_feature_engineering.py

## 1. Descripción General
El script `S3_feature_engineering.py` realiza un diagnóstico multidimensional de las variables predictoras. No se limita a una selección simple, sino que aplica pruebas de redundancia, colinealidad, causalidad temporal y relevancia no lineal para asegurar que solo las variables con mayor valor informativo lleguen a la etapa de modelado. Al final de cada ejecución, genera un **Resumen Compilado** que facilita la toma de decisiones quirúrgicas sobre el dataset.

---

## 2. Profundización Robusta en Métodos Estadísticos

### A. Correlación de Pearson (Análisis de Redundancia Bivariada)
*   **Fundamento Matemático**: Mide el grado de relación lineal entre dos variables aleatorias continuas. Su valor oscila entre -1 y 1, donde 1 es correlación positiva perfecta, -1 es correlación negativa perfecta y 0 es ausencia de relación lineal.
*   **¿Por qué se usa?**: Es el método más eficiente para detectar **redundancia directa**. Si dos variables (ej. `temp_min` y `temp_max`) tienen una correlación extremadamente alta, el modelo de regresión tendrá dificultades para asignar "crédito" a cada una, aumentando la inestabilidad de los coeficientes.
*   **Umbral Crítico (0.8)**: Se ha seleccionado 0.8 como límite de alerta. Por encima de este valor, la varianza compartida es superior al 64% ($r^2$), lo que indica que una de las variables es mayormente un espejo de la otra.
*   **Interpretación**: Una alerta de redundancia entre A y B sugiere que mantener ambas incrementará la complejidad del modelo sin añadir nuevo poder predictivo significativo.

### B. Factor de Inflación de la Varianza (VIF - Multicolinealidad)
*   **Fundamento Matemático**: Se calcula como $VIF_i = \frac{1}{1 - R_i^2}$, donde $R_i^2$ es el coeficiente de determinación obtenido al aplicar una regresión de la variable $x_i$ contra todas las demás variables predictoras del conjunto.
*   **¿Por qué se usa?**: A diferencia de Pearson (que solo ve pares), el VIF detecta la **Multicolinealidad**, un fenómeno donde una variable puede ser "reconstruida" casi perfectamente mediante una combinación lineal de *múltiples* otras variables.
*   **Escala de Interpretación**:
    *   **VIF = 1**: Ausencia total de colinealidad.
    *   **5 < VIF < 10**: Colinealidad moderada. Aceptable en contextos de predicción, pero arriesgado para inferencia.
    *   **VIF > 10**: Colinealidad severa. La varianza del coeficiente de esta variable está 10 veces "inflada", lo que hace que sus estimaciones sean poco fiables.
    *   **VIF = inf**: Colinealidad perfecta (singularidad de matriz). Ocurre cuando una variable es una suma o múltiplo exacto de otras. **Acción obligatoria: Eliminar.**

### C. Prueba de Causalidad de Granger (Precedencia Temporal)
*   **Fundamento Matemático**: Basada en modelos de autorregresión vectorial (VAR). La prueba evalúa la hipótesis nula ($H_0$) de que los valores pasados de $X$ no ayudan a predecir $Y$, comparando un modelo que solo usa rezagos de $Y$ frente a uno que usa rezagos de $Y$ y $X$.
*   **¿Por qué se usa?**: En el Dengue, existe un retraso biológico entre el clima (lluvia/temperatura) y la aparición de casos. Esta prueba valida si los rezagos (lags) que hemos creado realmente tienen una conexión estadística "causal" (en términos de información predictiva) con los casos confirmados.
*   **Interpretación de Resultados**:
    *   **P-valor < 0.05**: Se rechaza $H_0$. Confirmamos que la variable $X$ tiene **precedencia temporal** validada. Es una característica "estrella" para el pronóstico.
    *   **P-valor > 0.05**: No hay evidencia de que esta variable ayude a predecir el futuro de los casos más de lo que ya lo hace la propia historia de los casos.

### D. Importancia por Random Forest (Relevancia No Lineal)
*   **Fundamento Matemático**: Utiliza la métrica *Mean Decrease in Impurity* (Gini Importance). Mide cuánto reduce cada variable la incertidumbre (impureza) al dividir los datos en los nodos de los árboles del bosque.
*   **¿Por qué se usa?**: El Dengue a menudo presenta comportamientos de umbral (ej. la lluvia solo importa si supera los 50mm). Los métodos lineales (VIF/Pearson) pueden fallar en detectar estas relaciones. Random Forest, al ser un modelo no paramétrico de ensamble, captura interacciones complejas.
*   **Diagnóstico "NO LINEAL"**: Se activa cuando una variable tiene una importancia alta en el bosque pero una correlación lineal baja (Spearman < 0.2). 
*   **Valor**: Indica variables que son "diamantes en bruto" para modelos complejos pero que serían descartadas por una simple regresión lineal.

---

## 3. Guía de Lectura del Resumen Compilado

Al final de la ejecución, el script imprime un bloque titulado `=== RECOMENDACIONES COMPILADAS ===`. Aquí se explica cómo interpretarlo:

### Sección 1: Resumen de Redundancia Lineal (Pearson)
Muestra pares de variables que "se pisan" entre sí.
-   **Acción Recomendada**: Si ves `temp_min` y `temp_max` con correlación 1.00, **debes eliminar una de las dos**. Generalmente, se mantiene la que tenga un p-valor de Granger más bajo o la que tenga más sentido clínico/biológico.

### Sección 2: Resumen de Diagnóstico por Atributo
Lista cada variable y sus "pecados" o "virtudes" estadísticos.
-   **Si ves `VIF ALTO`**: La variable es estadísticamente redundante en el conjunto global. Evalúa eliminarla si el VIF > 10.
-   **Si ves `GRANGER validado`**: Es una variable ganadora. Incluso si tiene un VIF ligeramente alto (ej. 6-8), su poder de precedencia temporal sugiere que es valiosa para el modelo.
-   **Si ves `NO LINEAL`**: Esta variable solo brillará en modelos como XGBoost, LightGBM o Random Forest. Si planeas usar una Regresión Lineal o un GLM, esta variable probablemente no servirá.

---

## 4. Estrategia de Silenciamiento de Warnings
Para garantizar que el analista pueda leer estos resultados sin distracciones, el script aplica tres capas de limpieza:
1.  **Filtro Global**: Ignora avisos de depreciación (FutureWarnings).
2.  **Anulación de Warn**: Redirige la función `warnings.warn` para evitar impresiones de `statsmodels` (comunes en matrices singulares).
3.  **Gestión de NumPy**: Usa `np.seterr(all='ignore')` para ocultar errores de división por cero que ocurren naturalmente al calcular el VIF de variables perfectamente correlacionadas.
