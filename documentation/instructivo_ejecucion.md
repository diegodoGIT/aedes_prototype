# Instructivo de Ejecución Completa del Pipeline

## Estructura de Archivos Requerida

```
aedes_prototype/
├── data/
│   ├── Base_deng.xlsx                          ← Datos SISPRO
│   ├── Listados_DIVIPOLA.xlsx                  ← Códigos DANE
│   ├── nino34.long.anom.csv                    ← Anomalías ENSO
│   ├── climaticas_aedes_colombia.csv           ← NDVI y dew_point
│   ├── worldclim/                              ← Rasters .tif (tmin, tmax, prec, elev)
│   └── processed/                              ← Se genera automáticamente
├── processing/                                 ← Se genera automáticamente
├── S1_data_preparation_script.py               ← Original (no modificado)
├── S2_pre_processing_script.py                 ← VERSIÓN FINAL (con fix 0-casos)
├── S3_feature_engineering.py                   ← Original (no modificado)
├── S3_5_testing_feature_engineering.py          ← Original (no modificado)
├── S4_processing_script.py                     ← VERSIÓN FINAL (con fix .ravel())
├── setup_pre_s4.JSON                           ← Config A (PCA temperatura)
└── project_logs.log                            ← Se genera automáticamente
```

## Pasos de Ejecución

### Paso 1: S1 — Ingesta y Fusión de Datos (solo si no se ha ejecutado antes)

```powershell
python S1_data_preparation_script.py
```

**Qué hace:** Lee las fuentes de datos (SISPRO, WorldClim, NDVI, ENSO), las fusiona y genera un dataset unificado.

**Output:** `data/dengue_data_v2_ocur.csv` (202,176 registros, 34 columnas)

**Tiempo estimado:** ~30 minutos

**Cuándo re-ejecutar:** Solo si cambian los datos fuente. Si `data/dengue_data_v2_ocur.csv` ya existe, saltar este paso.

---

### Paso 2: S2 — Imputación + Canal Endémico Bortman

```powershell
python S2_pre_processing_script.py --target_lags 3
```

O si ya tienes checkpoints del canal endémico:

```powershell
python S2_pre_processing_script.py --target_lags 3 --skip_channel
```

**Qué hace:**
1. Aplica 5 métodos de imputación (M1-M5) sobre casos faltantes
2. Construye el canal endémico de Bortman por municipio-mes (solo con datos pasados)
3. Clasifica cada registro en 3 niveles: éxito(0), en_rango(1), epidemia(2)
4. Genera lags de casos y nivel de riesgo (1-3 meses)
5. Preserva nivel_endemico (MG histórica) como feature

**Fix incluido en esta versión:** Si un municipio tiene 0 casos confirmados, siempre se clasifica como éxito (0), independientemente de los umbrales. Esto corrige el bug donde municipios con 0 casos históricos y 0 casos actuales se clasificaban como epidemia porque LI=MG=LS=0.

**Outputs:**
- `data/processed/dengue_imputed_M1.csv` hasta `M5.csv`
- `data/processed/checkpoints/` (para re-ejecución rápida)
- `data/processed/municipios_coordenadas.csv`
- `data/processed/umbrales_canal_endemico.csv`
- `data/processed/cobertura_departamentos.csv`

**Tiempo estimado:** ~1 hora (primera vez), ~10 minutos (con --skip_channel)

---

### Paso 3: S3.5 — Ingeniería de Variables y Diagnóstico

```powershell
python S3_5_testing_feature_engineering.py --moving_avg 0
```

**Qué hace:**
1. Lee `setup_pre_s4.JSON` y aplica transformaciones:
   - Elimina columnas de metadatos y leakage (casosconfirmados, umbral_li, umbral_ls)
   - Promedia tempmax + tempmin en temp_avg
   - Aplica PCA sobre 5 grupos de variables (dew_point, casos_lags, riesgo_lags, temp_avg, ENSO)
2. Ejecuta diagnóstico estadístico (Pearson, Kruskal-Wallis, Granger, importancia RF)
3. Genera datasets `_modified_` listos para S4

**Output:** `data/processed/dengue_imputed_M*_modified_*.csv` (5 archivos, 23 columnas cada uno)

**Tiempo estimado:** ~3 minutos

---

### Paso 4: S4 — Entrenamiento y Evaluación

```powershell
# Modo completo (5 modelos × 10 iteraciones, búsqueda de hiperparámetros):
python S4_processing_script.py --n_iter 10

# Modo rápido (1 modelo, parámetros por defecto):
python S4_processing_script.py --fast lgbm
```

**Qué hace:**
1. Carga los datasets `_modified_` generados por S3.5
2. Separa el año 2024 como validación hold-out
3. Entrena clasificadores con RandomizedSearchCV y TimeSeriesSplit
4. Evalúa en hold-out 2024 con métricas por clase
5. Exporta predicciones con etiquetas textuales

**Fix incluido en esta versión:** `.ravel()` en la función MAE ordinal para evitar error de memoria con arrays 2D.

**Outputs:**
- `processing/experiment_technical_log.csv` (métricas de todas las iteraciones)
- `processing/regional_performance.csv` (desempeño por departamento)
- `processing/M4_lgbm_best.pkl` (mejor modelo serializado)
- `processing/M4_lgbm_predicciones.csv` (predicciones vs reales)

**Tiempo estimado:** ~15 minutos (modo rápido) / ~90 minutos (modo completo)

---

## Ejecución Completa desde Cero

Si es la primera vez o quieres regenerar todo:

```powershell
cd C:\Users\diego\Documents\aedes_project\aedes_prototype

# Limpiar resultados anteriores (CONSERVAR data/ fuentes)
Remove-Item data\processed\dengue_imputed_*.csv -ErrorAction SilentlyContinue
Remove-Item data\processed\checkpoints\* -ErrorAction SilentlyContinue
Remove-Item processing\*.csv -ErrorAction SilentlyContinue
Remove-Item processing\*.pkl -ErrorAction SilentlyContinue

# Paso 1: Solo si no existe data/dengue_data_v2_ocur.csv
python S1_data_preparation_script.py

# Paso 2: Canal endémico + imputación
python S2_pre_processing_script.py --target_lags 3

# Paso 3: Ingeniería de variables
python S3_5_testing_feature_engineering.py --moving_avg 0

# Paso 4: Entrenamiento completo
python S4_processing_script.py --n_iter 10
```

## Re-ejecución Parcial (más común)

Si solo cambió el JSON de configuración o quieres probar nuevos hiperparámetros:

```powershell
# Limpiar solo los _modified_ y resultados
Remove-Item data\processed\dengue_imputed_M*_modified_*.csv -ErrorAction SilentlyContinue
Remove-Item processing\experiment_technical_log.csv -ErrorAction SilentlyContinue
Remove-Item processing\*.pkl -ErrorAction SilentlyContinue

# Re-ejecutar desde S3.5
python S3_5_testing_feature_engineering.py --moving_avg 0
python S4_processing_script.py --n_iter 10
```

## Verificación de Resultados

Después de la ejecución, verificar que:

1. **No hay 0-casos como epidemia:**
```powershell
python -c "import pandas as pd; df=pd.read_csv('processing/M4_lgbm_predicciones.csv'); print(df[df['etiqueta_predicha']=='epidemia'].head())"
```

2. **Métricas esperadas (LGBM M4):**
   - Macro F1 Val ≥ 0.70
   - Precision epidemia ≥ 0.90
   - Sensibilidad epidemia ≥ 0.75
   - Error catastrófico < 2%

3. **Archivos generados:**
```powershell
Get-ChildItem processing\*.csv | Select-Object Name, Length
Get-ChildItem processing\*.pkl | Select-Object Name, Length
```

## Cambios Realizados en los Scripts

### S2_pre_processing_script.py (Línea ~354)
**Antes:** Si casosconfirmados = 0 y LI = MG = LS = 0, el bloque `else` clasificaba como epidemia.
**Después:** Regla explícita: si casosconfirmados == 0, siempre nivel_riesgo = 0 (éxito).

### S4_processing_script.py (Línea 69)
**Antes:** `np.array(y_true, dtype=int) - np.array(y_pred, dtype=int)` podía crear matriz n×n por broadcasting.
**Después:** `.ravel()` fuerza arrays 1D, evitando error de memoria de 4-8 GB.

### setup_pre_s4.JSON
Config A: promedia tempmax+tempmin, aplica PCA temperatura (1 componente), PCA ENSO (2 componentes), PCA dew_point/casos_lags/riesgo_lags (1 componente cada uno). Preserva nivel_endemico. Elimina casosconfirmados, umbrales y metadatos textuales.
