import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
import sys
import argparse
from datetime import datetime

# --- Configuración de Logging (Unificada con el proyecto) ---
log_filename = "project_logs.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Parámetros de Configuración Interna (Modificables en el script) ---
# Estas opciones permiten controlar transformaciones sin necesidad de argumentos de línea de comandos.
APPLY_DUMMIES = True  # Cambiar a True para habilitar la conversión a variables dummies
CATEGORICAL_COLS = ['fenomeno', 'fenomeno_lag1','fenomeno_lag2','fenomeno_lag3']  # Columnas a transformar si APPLY_DUMMIES es True
#* No es necesario incluir el mes porque ya esta compensado en las columnas de mes cíclico (mes_sin y mes_cos) que se generan
#* en la función feature_engineering, y de hecho incluirlo como dummy podría generar multicolinealidad con estas columnas cíclicas.
# ----------------------------------------------------------------------

class TargetImputer:
    def __init__(self, df, target_col='casosconfirmados', year_col='year', month_col='mes'):
        self.df_original = df.sort_values(['departamento', 'municipio', 'year', 'month_idx' if 'month_idx' in df.columns else month_col]).copy()
        self.target_col = target_col
        self.year_col = year_col
        self.month_col = month_col
        self.results = {}
        self.imputation_logs = []

    def _log_entry(self, idx, method, val_orig, val_calc, val_final, extra_info={}):
        row = self.df_original.loc[idx]
        log_data = {
            'index': idx,
            'departamento': row['departamento'],
            'municipio': row['municipio'],
            'year': row[self.year_col],
            'mes': row[self.month_col],
            'metodo': method,
            'valor_original': val_orig,
            'valor_calculado': val_calc,
            'valor_imputado': val_final
        }
        log_data.update(extra_info)
        self.imputation_logs.append(log_data)

    def fit_transform_m1_m2(self, use_median=False):
        """
        ### Resumen:
        Imputa valores nulos usando la media (M1) o mediana (M2)
        de los casos confirmados para el mismo mes y departamento, considerando solo años anteriores.
        
        ### Detalles:
        Para cada valor nulo, se filtran los datos históricos del mismo departamento y mes,
        pero solo de años anteriores al año del registro a imputar. Si hay al menos 4 años de datos disponibles,
         se calcula la media o mediana según el método seleccionado. Si el valor calculado es menor a 1, 
         se asigna 0 para evitar casos negativos o fracciones. Si no hay suficientes datos históricos, 
         el valor se mantiene nulo. Se registra un log detallado de cada imputación realizada, 
         incluyendo el número de años usados y si se aplicó el floor a cero.
        """
        method_name = "M2" if use_median else "M1"
        logger.info(f"Ejecutando {method_name}...")
        df = self.df_original.copy()
        
        #* Iterar por año para optimizar el cálculo de datos históricos
        for year in sorted(df[self.year_col].unique()):
            year_nulls = df[(df[self.year_col] == year) & (df[self.target_col].isnull())].index
            if year_nulls.empty:
                continue
            
            # Datos históricos: años estrictamente anteriores
            historical_data = self.df_original[self.df_original[self.year_col] < year]
            if historical_data.empty:
                continue
                
            # Calcular estadísticas agrupadas para este año
            stats = historical_data.groupby(['departamento', 'municipio', self.month_col])[self.target_col].agg(['mean', 'median', 'count'])
            
            for idx in year_nulls:
                row = df.loc[idx]
                key = (row['departamento'], row['municipio'], row[self.month_col])
                
                if key in stats.index:
                    stat_row = stats.loc[key]
                    if stat_row['count'] >= 4:
                        calc_val = stat_row['median'] if use_median else stat_row['mean']
                        floor_to_zero = calc_val < 1
                        final_val = 0 if floor_to_zero else int(round(calc_val))
                        
                        df.at[idx, self.target_col] = final_val
                        self._log_entry(idx, method_name, np.nan, calc_val, final_val, 
                                        {f'{method_name}_años_usados': int(stat_row['count']), 
                                         f'{method_name}_floor_to_zero': floor_to_zero})
        
        self.results[method_name] = df
        return df

    def fit_transform_m3(self, max_gap=3):
        """### Resumen:
        Imputa valores nulos usando interpolación lineal dentro de cada grupo de departamento y municipio.
        ### Detalles:
        Para cada grupo de departamento y municipio, se aplica interpolación lineal a la columna de casos
        confirmados. Solo se imputan valores nulos que estén dentro de una brecha máxima de 3 meses entre
        valores no nulos. Esto significa que si hay más de 3 meses consecutivos con valores nulos, esos
        valores no serán imputados. Se registra un log detallado de cada imputación realizada,
        incluyendo el valor calculado por la interpolación y el valor final asignado (redondeado al entero más cercano).

        La interpolacion lineal consiste en calcular el valor de un punto intermedio entre dos puntos conocidos
        (en este caso, meses con datos no nulos) asumiendo una relación lineal entre ellos. Por ejemplo, si en
        un departamento-municipio el mes 1 tiene 10 casos, el mes 2 es nulo, y el mes 3 tiene 20 casos, la
        interpolación lineal imputaría 15 casos para el mes 2, ya que es el punto medio entre 10 y 20. Sin embargo,
        si el mes 2 estuviera seguido por otros dos meses nulos (meses 3 y 4), entonces el método no imputaría el
        mes 2 debido a la brecha de más de 3 meses consecutivos sin datos.
    
        """
        logger.info("Ejecutando M3 (Interpolación)...")
        df = self.df_original.copy()
        
        #* Aplicar interpolación lineal dentro de cada grupo de departamento y municipio

        df[self.target_col] = df.groupby(['departamento', 'municipio'])[self.target_col].transform(
            lambda x: x.interpolate(method='linear', limit=max_gap, limit_area='inside')
        )
        #* Crear mascara con los indices que fueron imputados por interpolacion (nulos en original y no nulos en el df resultante)
        mask_imputed = self.df_original[self.target_col].isnull() & df[self.target_col].notnull()
        new_imputed_df = df[mask_imputed]
        #* Iterar solo sobre los indices que fueron imputados por interpolacion
        #* Para cada uno, registrar el valor calculado por la interpolación y el valor final asignado (redondeado al entero más cercano)
        for idx, row in new_imputed_df.iterrows():
            val_calc = row[self.target_col]
            val_final = int(round(val_calc))
            df.at[idx, self.target_col] = val_final
            self._log_entry(idx, "M3", np.nan, val_calc, val_final)
            
        self.results["M3"] = df
        return df

    def fit_transform_m4(self, alpha=0.3):
        """### Resumen:
        Imputa valores nulos usando el promedio móvil exponencial (EWM) dentro de cada grupo de departamento y municipio.
        ### Detalles:
        Para cada grupo de departamento y municipio, se calcula el promedio móvil exponencial (EWM) de la columna de casos
        confirmados. El EWM se calcula utilizando un factor de suavizado alpha de 0.3 y requiere un mínimo de 12 meses de
        datos anteriores para generar un valor confiable. Solo se imputan valores nulos que tengan al menos 12 meses de datos
        anteriores en el mismo departamento y municipio. El valor calculado por el EWM se redondea al entero más cercano.
        Se registra un log detallado de cada imputación realizada, incluyendo el valor calculado
        por el EWM, el número de meses anteriores utilizados para el cálculo, y si se aplicó el floor a cero.
        """
        logger.info("Ejecutando M4 (EWM)...")
        df = self.df_original.copy()
        #* creando datos con imputacion EWM en una columna temporal para luego asignar solo a 
        #* los indices nulos originales que tengan al menos 12 meses de datos anteriores
        df['ewm_temp'] = df.groupby(['departamento', 'municipio'])[self.target_col].transform(
            lambda x: x.shift(1).ewm(alpha=alpha, min_periods=12, adjust=False).mean()
        )
        
        mask_imputed = self.df_original[self.target_col].isnull() & df['ewm_temp'].notnull()
        null_indices = df[mask_imputed].index
        
        for idx in null_indices:
            calc_val = df.at[idx, 'ewm_temp']
            final_val = int(round(calc_val))
            df.at[idx, self.target_col] = final_val
            self._log_entry(idx, "M4", np.nan, calc_val, final_val)

        #* eliminar la columna temporal de EWM para evitar confusiones en los datasets finales    
        df.drop(columns=['ewm_temp'], inplace=True, errors='ignore')
        self.results["M4"] = df
        return df

    def fit_transform_m5(self):
        logger.info("Ejecutando M5 (Baseline)...")
        self.results["M5"] = self.df_original.copy()
        return self.results["M5"]

    def run_all(self):
        #* Ejecutar todos los métodos secuencialmente
        #* 1. M1 y M2 (media y mediana) - se ejecutan en una misma función
        self.fit_transform_m1_m2(use_median=False) # M1
        self.fit_transform_m1_m2(use_median=True)  # M2
        #* 2. M3 - Interpolación lineal
        self.fit_transform_m3()                   # M3
        #* 3. M4 - EWM (Exponential Weighted Mean) con alpha=0.3 y mínimo 12 meses de datos anteriores
        self.fit_transform_m4()                   # M4
        #* 4. M5 - Baseline (sin imputación, solo eliminación de nulos)
        self.fit_transform_m5()                   # M5
        return self.results

def feature_engineering(df, target_lags=3):
    """### Resumen:
    Agrega características de mes cíclico y rezagos de la variable objetivo.
    ### Detalles:
    Se crean dos nuevas columnas para representar el mes de forma cíclica utilizando funciones trigonométricas:
    'mes_sin' y 'mes_cos'. Esto permite que los modelos capturen la naturaleza cíclica de los
    meses del año. Además, se generan rezagos de la variable objetivo 'casosconfirmados' para el número de
    meses anteriores definido por 'target_lags' (por defecto 3) dentro de cada grupo de departamento y
    municipio. Los valores nulos resultantes de los rezagos se llenan con la mediana de cada grupo para esos
    rezagos, y cualquier valor nulo restante se llena con 0. Es importante destacar que no se realiza un dropna
    en esta función para permitir un análisis más granular en el reporte de cobertura por departamento, y solo se
    eliminarán los registros con valores nulos al final del proceso antes de la exportación."""
    
    df = df.copy()
    
    #* Imputar nulos en variables climáticas y de vegetación con la media del grupo (depto-muni)
    #* Esto evita que el dropna() final elimine filas donde la variable objetivo fue recuperada
    #* pero falta algún dato climático menor.
    ambient_cols = [c for c in df.columns if any(x in c for x in ['dew_point', 'ndvi', 'precipitacion', 'temp'])]
    if ambient_cols:
        logger.info(f"Imputando nulos en {len(ambient_cols)} columnas ambientales con la media por municipio...")
        for col in ambient_cols:
            df[col] = df.groupby(['departamento', 'municipio'])[col].transform(lambda x: x.fillna(x.mean()))
            #* Si aún quedan nulos (municipio sin ningún dato), usar la media del departamento
            if df[col].isnull().any():
                df[col] = df.groupby(['departamento'])[col].transform(lambda x: x.fillna(x.mean()))
            #* Relleno final con 0 para casos extremos sin datos en todo el departamento
            df[col] = df[col].fillna(0)

    #* crear columnas de mes cíclico
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    #* generar rezagos de la variable objetivo 'casosconfirmados' dentro de cada grupo de departamento y municipio
    df = df.sort_values(['departamento', 'municipio', 'year', 'mes'])
    lag_cols = []
    for l in range(1, target_lags + 1):
        col_name = f'target_lag{l}'
        df[col_name] = df.groupby(['departamento', 'municipio'])['casosconfirmados'].shift(l)
        lag_cols.append(col_name)
    
    if lag_cols:
        #* llenar los valores nulos resultantes de los rezagos con la mediana de cada grupo para esos rezagos
        df[lag_cols] = df.groupby(['departamento', 'municipio'])[lag_cols].transform(lambda x: x.fillna(x.median()))
        #* nulos adicionales en los rezagos de casosconfirmados se rellenan con 0
        df[lag_cols] = df[lag_cols].fillna(0)
    
    # IMPORTANTE: No hacemos dropna aquí para el reporte, solo al final de la exportación
    return df

def generate_coverage_report(df_orig, imputed_datasets, processed_dir, target_lags=3):
    """### Resumen:
    Genera un reporte de cobertura por departamento.
    ### Detalles:
    Para cada departamento, se calcula el número total de meses, el número de meses
    nulos en la columna de casos confirmados en el dataset original, el número de meses
    imputados por cada método (M1 a M4), y el número de meses eliminados en M5. Además,
    se determina si el departamento está incluido en los datasets finales después de aplicar
    feature engineering y dropna, y si no lo está, se registra la razón de exclusión basada
    en los criterios establecidos (datos insuficientes tras imputación y feature engineering).
    El reporte se guarda como un archivo CSV llamado 'cobertura_departamentos.csv' en el
    directorio de datos procesados. Este reporte es fundamental para entender la cobertura y
    representatividad de cada departamento en los datasets finales utilizados para modelado.
    """
    logger.info(f"Generando reporte de cobertura por departamento (Target Lags: {target_lags})...")
    report_rows = []
    depts = sorted(df_orig['departamento'].unique())
    
    #* Aplicar feature engineering a cada dataset imputado para luego verificar la inclusión de cada departamento en los datasets finales

    final_sets = {m: feature_engineering(ds, target_lags=target_lags).dropna() for m, ds in imputed_datasets.items()}
    #* iterar sobre departamento obteniendo numero de registros originales y numero de vacios
    for dept in depts:
        df_dept_orig = df_orig[df_orig['departamento'] == dept]
        total_meses = len(df_dept_orig)
        meses_nulos_orig = df_dept_orig['casosconfirmados'].isnull().sum()
        
        row = {
            'departamento': dept,
            'total_meses': total_meses,
            'meses_nulos_original': meses_nulos_orig
        }
        #* por cada metodo de imputacion, calcular los nulos originales y los no nulos tras imputacion para obtener el numero de meses imputados por cada metodo
        for method in ["M1", "M2", "M3", "M4"]:
            df_imp = imputed_datasets[method]
            mask_was_null = df_dept_orig['casosconfirmados'].isnull()
            mask_is_filled = df_imp.loc[df_dept_orig.index, 'casosconfirmados'].notnull()
            row[f'meses_imputados_{method}'] = (mask_was_null & mask_is_filled).sum()
            incluido = dept in final_sets[method]['departamento'].unique()
            row[f'incluido_en_{method}'] = incluido
            if not incluido:
                row[f'razon_exclusion_{method}'] = 'Datos insuficientes tras imputación y feature engineering'
            else:
                row[f'razon_exclusion_{method}'] = ''

        #* meses nulos originales
        row['meses_eliminados_M5'] = meses_nulos_orig
        
        report_rows.append(row)
        
    pd.DataFrame(report_rows).to_csv(processed_dir / 'cobertura_departamentos.csv', index=False)

def apply_categorical_encoding(df, columns, apply=False):
    """
    ### Resumen:
    Transforma variables categóricas seleccionadas en variables dummy.
    ### Detalles:
    Si 'apply' es True, utiliza pd.get_dummies para convertir las columnas especificadas
    en indicadores binarios. Se utiliza drop_first=True para evitar la trampa de la
    variable dummy (multicolinealidad). Esta transformación se aplica al final del
    procesamiento para no interferir con las agrupaciones por departamento/municipio.
    """
    if not apply or not columns:
        return df
    
    logger.info(f"Aplicando codificación dummy a las columnas: {columns}")
    # Filtramos solo las columnas que realmente existen en el DataFrame
    existing_cols = [c for c in columns if c in df.columns]
    if existing_cols:
        return pd.get_dummies(df, columns=existing_cols, drop_first=True, dtype=np.uint8)
    return df

def save_location_auxiliary(df, processed_dir):
    """
    ### Resumen:
    Guarda un archivo auxiliar con las combinaciones únicas de ubicación.
    ### Detalles:
    Extrae las columnas de departamento, municipio, longitud y latitud,
    obtiene las filas únicas y las guarda en 'municipios_coordenadas.csv'.
    """
    cols_interes = ['departamento', 'municipio', 'longitud', 'latitud']
    # Verificar cuáles de estas columnas existen realmente
    available_cols = [c for c in cols_interes if c in df.columns]
    
    if len(available_cols) > 0:
        logger.info(f"Generando archivo auxiliar de ubicaciones con columnas: {available_cols}")
        df_coords = df[available_cols].drop_duplicates().sort_values(['departamento', 'municipio'])
        output_path = processed_dir / 'municipios_coordenadas.csv'
        df_coords.to_csv(output_path, index=False)
        logger.info(f"Archivo de coordenadas guardado en: {output_path}")
    else:
        logger.warning("No se encontraron columnas de ubicación (departamento, municipio, longitud, latitud) para generar el archivo auxiliar.")

def main():
    parser = argparse.ArgumentParser(description="S2: Preprocesamiento e Imputación de Datos")
    parser.add_argument('--target_lags', type=int, default=3, help="Número de rezagos del target a generar")
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info(f"=== INICIO DEL PREPROCESAMIENTO Y GENERACIÓN DE DATASETS (S2) ===")
    logger.info(f"Configuración: Target Lags = {args.target_lags}")
    
    #* Creación de directorio para datos procesados
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    #* Carga de datos
    v2_path = Path('data/dengue_data_v2_ocur.csv')
    if v2_path.exists():
        df = pd.read_csv(v2_path, sep='|')
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # --- Identificación de Dimensiones y Lags ---
        rows, cols = df.shape
        lag_cols = [c for c in df.columns if '_lag' in c]
        logger.info(f"Dataset original cargado: {rows} filas y {cols} columnas.")
        
        if lag_cols:
            # Extraemos las raíces de las variables para un log más limpio
            base_vars_with_lags = sorted(list(set([c.split('_lag')[0] for c in lag_cols])))
            num_base_vars = len(base_vars_with_lags)
            total_lag_cols = len(lag_cols)
            
            # Inferir el número de lags por variable (Total de columnas / Num variables únicas)
            lags_per_var = total_lag_cols // num_base_vars
            
            logger.info(f"Variables con rezagos (lags) detectadas ({num_base_vars}): {', '.join(base_vars_with_lags)}")
            logger.info(f"Inferencia: Se detectaron {lags_per_var} rezagos (lags) por cada variable base.")
            logger.info(f"Total de columnas de rezago encontradas: {total_lag_cols}")
        else:
            logger.info("No se detectaron columnas de rezago (lags) en el dataset original.")
        # --------------------------------------------
        
        # --- Generar archivo auxiliar de ubicaciones ---
        save_location_auxiliary(df, processed_dir)
        # -----------------------------------------------

    else:
        logger.error("Base de datos no encontrada.")
        return
    #* Ejecución de imputaciones y generación de datasets finales
    imputer = TargetImputer(df)
    #* imputed_datasets es un diccionario con claves M1, M2, M3, M4, M5 y valores los dataframes imputados correspondientes
    imputed_datasets = imputer.run_all()
    #* Generación de reporte de cobertura por departamento
    generate_coverage_report(df, imputed_datasets, processed_dir, target_lags=args.target_lags)
    #* Exportación de datasets finales
    for method, df_imp in imputed_datasets.items():
        logger.info(f"Procesando dataset final para {method}...")
        df_final = feature_engineering(df_imp, target_lags=args.target_lags)
        #* Exportar solo después de feature engineering y dropna
        df_final = df_final.dropna()
        
        #* Codificación categórica (opcional según parámetros internos)
        df_final = apply_categorical_encoding(df_final, CATEGORICAL_COLS, apply=APPLY_DUMMIES)
        
        #* Guardar con metadatos para main_v2.py
        df_final.to_csv(processed_dir / f'dengue_imputed_{method}.csv', index=False)
        logger.info(f"Exportado: {method} con {len(df_final)} registros.")
    
    end_time = datetime.now()
    logger.info(f"=== S2 FINALIZADO EXITOSAMENTE en {end_time - start_time} ===")

if __name__ == "__main__":
    main()
