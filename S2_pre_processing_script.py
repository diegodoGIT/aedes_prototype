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
APPLY_DUMMIES = True
CATEGORICAL_COLS = ['fenomeno', 'fenomeno_lag1','fenomeno_lag2','fenomeno_lag3']

# --- Configuración del Canal Endémico ---
ENDEMIC_CHANNEL_YEARS = 7       # Años de historia para calcular percentiles
EXCLUDE_OUTLIER_YEARS = True    # Excluir años epidémicos atípicos del cálculo
RISK_LEVELS = {                 # Canal endémico Bortman (3 niveles operativos)
    0: 'exito',                 # Por debajo del LI IC95%
    1: 'en_rango',              # Entre LI y LS (seguridad + alerta colapsados)
    2: 'epidemia'               # Por encima del LS IC95%
}
RISK_LEVELS_4 = {               # Niveles originales (para log de trazabilidad)
    0: 'exito', 1: 'seguridad', 2: 'alerta', 3: 'epidemia'
}
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
        Imputa valores nulos usando la media (M1) o mediana (M2) de los casos confirmados
        para el mismo mes y municipio, considerando solo años anteriores.
        Requiere al menos 4 años de historia disponible.
        """
        method_name = "M2" if use_median else "M1"
        logger.info(f"Ejecutando {method_name}...")
        df = self.df_original.copy()
        
        for year in sorted(df[self.year_col].unique()):
            year_nulls = df[(df[self.year_col] == year) & (df[self.target_col].isnull())].index
            if year_nulls.empty:
                continue
            
            historical_data = self.df_original[self.df_original[self.year_col] < year]
            if historical_data.empty:
                continue
                
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
        """
        Imputa valores nulos usando interpolación lineal dentro de cada grupo de
        departamento y municipio, con límite de brecha de 3 meses.
        """
        logger.info("Ejecutando M3 (Interpolación)...")
        df = self.df_original.copy()
        
        df[self.target_col] = df.groupby(['departamento', 'municipio'])[self.target_col].transform(
            lambda x: x.interpolate(method='linear', limit=max_gap, limit_area='inside')
        )
        mask_imputed = self.df_original[self.target_col].isnull() & df[self.target_col].notnull()
        new_imputed_df = df[mask_imputed]

        for idx, row in new_imputed_df.iterrows():
            val_calc = row[self.target_col]
            val_final = int(round(val_calc))
            df.at[idx, self.target_col] = val_final
            self._log_entry(idx, "M3", np.nan, val_calc, val_final)
            
        self.results["M3"] = df
        return df

    def fit_transform_m4(self, alpha=0.3):
        """
        Imputa valores nulos usando el promedio móvil exponencial (EWM) con alpha=0.3.
        Requiere al menos 12 meses de datos anteriores en el mismo municipio.
        """
        logger.info("Ejecutando M4 (EWM)...")
        df = self.df_original.copy()

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

        df.drop(columns=['ewm_temp'], inplace=True, errors='ignore')
        self.results["M4"] = df
        return df

    def fit_transform_m5(self):
        logger.info("Ejecutando M5 (Baseline)...")
        self.results["M5"] = self.df_original.copy()
        return self.results["M5"]

    def run_all(self):
        self.fit_transform_m1_m2(use_median=False)  # M1
        self.fit_transform_m1_m2(use_median=True)   # M2
        self.fit_transform_m3()                      # M3
        self.fit_transform_m4()                      # M4
        self.fit_transform_m5()                      # M5
        return self.results


# =============================================================================
# NUEVA CLASE: Construcción del Canal Endémico y Clasificación de Riesgo
# =============================================================================
class EndemicChannelBuilder:
    """
    Construye el canal endémico por municipio-mes usando la metodología de
    medias geométricas de Bortman, tal como la aplica el INS de Colombia.
    
    Metodología (Bortman, Rev Panam Salud Publica, 1999):
    1. Para cada municipio-mes, se toman los casos de los N años anteriores.
    2. Se transforman a escala logarítmica: x_i = ln(casos + 1).
    3. Se calculan media (x̄) y desviación estándar (s) de los logaritmos.
    4. Media geométrica = exp(x̄) - 1
    5. Límite inferior IC95% = exp(x̄ - 1.96 × s/√n) - 1
    6. Límite superior IC95% = exp(x̄ + 1.96 × s/√n) - 1
    
    Clasificación (consistente con BES del INS):
    - 0 (éxito):     casos < límite inferior IC95%
    - 1 (seguridad):  límite inferior ≤ casos < media geométrica
    - 2 (alerta):     media geométrica ≤ casos < límite superior IC95%
    - 3 (epidemia):   casos ≥ límite superior IC95%
    
    Diseñado para prevenir data leakage: los umbrales de cada año se calculan
    usando SOLO datos de años estrictamente anteriores.
    
    Parámetros:
    -----------
    history_years : int
        Número máximo de años de historia a considerar (ventana móvil).
    exclude_outliers : bool
        Si True, excluye años epidémicos atípicos del cálculo usando IQR.
    min_years : int
        Años mínimos requeridos para calcular umbrales confiables.
    """
    
    def __init__(self, history_years=7, exclude_outliers=True, min_years=3):
        self.history_years = history_years
        self.exclude_outliers = exclude_outliers
        self.min_years = min_years
        self.channel_log = []
    
    def _identify_outlier_years(self, group_data, year_col='year', target_col='casosconfirmados'):
        """
        Identifica años epidémicos atípicos para un municipio usando el método IQR
        sobre el total anual de casos.
        
        Un año se considera atípico si su total de casos supera Q3 + 1.5 * IQR,
        lo cual indica un brote excepcional que distorsionaría los umbrales normales.
        """
        annual_totals = group_data.groupby(year_col)[target_col].sum()
        
        if len(annual_totals) < 4:
            return set()
        
        q1 = annual_totals.quantile(0.25)
        q3 = annual_totals.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        outlier_years = set(annual_totals[annual_totals > upper_bound].index)
        return outlier_years
    
    @staticmethod
    def _bortman_thresholds(values):
        """
        Calcula los umbrales del canal endémico usando media geométrica + IC95%.
        
        Parámetros:
            values: array-like de casos confirmados (enteros >= 0)
        
        Returns:
            tuple: (limite_inferior, media_geometrica, limite_superior)
                   Todos en escala original (casos). Mínimo 0.
        """
        values = np.array(values, dtype=float)
        n = len(values)
        
        # Transformación logarítmica (ln(casos + 1) para manejar ceros)
        log_values = np.log(values + 1)
        
        log_mean = np.mean(log_values)
        log_std = np.std(log_values, ddof=1)  # ddof=1 para muestra
        
        # Error estándar
        se = log_std / np.sqrt(n)
        
        # Umbrales en escala log, luego revertir
        mg = np.exp(log_mean) - 1              # Media geométrica
        li = np.exp(log_mean - 1.96 * se) - 1  # Límite inferior IC95%
        ls = np.exp(log_mean + 1.96 * se) - 1  # Límite superior IC95%
        
        # Asegurar mínimo 0 (no puede haber umbrales negativos)
        li = max(li, 0.0)
        mg = max(mg, 0.0)
        ls = max(ls, 0.0)
        
        return li, mg, ls
    
    def build_risk_levels(self, df, target_col='casosconfirmados', year_col='year', month_col='mes'):
        """
        Asigna nivel de riesgo a cada registro basándose en el canal endémico
        Bortman calculado con datos estrictamente anteriores al año del registro.
        
        Flujo para cada registro (municipio, año Y, mes M):
        1. Seleccionar datos del mismo municipio y mes M de los últimos N años antes de Y.
        2. Si exclude_outliers=True, remover años atípicos de esa ventana.
        3. Si quedan >= min_years datos, calcular media geométrica + IC95%.
        4. Clasificar el valor actual según esos umbrales.
        
        Returns:
            DataFrame con columna 'nivel_riesgo' (0-3) y columnas de umbrales Bortman.
        """
        logger.info(f"=== CONSTRUYENDO CANAL ENDÉMICO BORTMAN (Historia: {self.history_years} años, "
                     f"Excluir outliers: {self.exclude_outliers}) ===")
        
        df = df.copy()
        df['nivel_riesgo'] = np.nan
        df['umbral_li'] = np.nan    # Límite inferior IC95%
        df['umbral_mg'] = np.nan    # Media geométrica
        df['umbral_ls'] = np.nan    # Límite superior IC95%
        
        location_groups = df.groupby(['departamento', 'municipio'])
        
        total_locations = len(location_groups)
        classified_count = 0
        insufficient_count = 0
        outlier_years_total = 0
        
        for (dept, muni), group in location_groups:
            # Identificar años outlier para este municipio
            outlier_years = set()
            if self.exclude_outliers:
                outlier_years = self._identify_outlier_years(group, year_col, target_col)
                if outlier_years:
                    outlier_years_total += len(outlier_years)
                    logger.debug(f"[{dept}/{muni}] Años atípicos excluidos: {outlier_years}")
            
            for year in sorted(group[year_col].unique()):
                # --- PREVENCIÓN DE DATA LEAKAGE ---
                min_hist_year = year - self.history_years
                hist_mask = (
                    (group[year_col] >= min_hist_year) & 
                    (group[year_col] < year) &
                    (~group[year_col].isin(outlier_years)) &
                    (group[target_col].notnull())
                )
                historical = group[hist_mask]
                
                available_years = historical[year_col].nunique()
                
                if available_years < self.min_years:
                    insufficient_count += group[group[year_col] == year].shape[0]
                    continue
                
                year_month_mask = group[year_col] == year
                year_records = group[year_month_mask]
                
                for mes in year_records[month_col].unique():
                    hist_month_data = historical[historical[month_col] == mes][target_col]
                    
                    if len(hist_month_data) < self.min_years:
                        continue
                    
                    # Calcular umbrales Bortman
                    li, mg, ls = self._bortman_thresholds(hist_month_data.values)
                    
                    # Índices del registro actual
                    current_mask = (
                        (df['departamento'] == dept) & 
                        (df['municipio'] == muni) & 
                        (df[year_col] == year) & 
                        (df[month_col] == mes)
                    )
                    current_indices = df[current_mask].index
                    
                    for idx in current_indices:
                        current_val = df.at[idx, target_col]
                        
                        # Guardar umbrales para trazabilidad
                        df.at[idx, 'umbral_li'] = round(li, 2)
                        df.at[idx, 'umbral_mg'] = round(mg, 2)
                        df.at[idx, 'umbral_ls'] = round(ls, 2)
                        
                        if pd.isna(current_val):
                            continue
                        
                        # Clasificación según canal endémico Bortman (INS)
                        if current_val < li:
                            risk = 0  # Éxito (por debajo de lo esperado)
                        elif current_val < mg:
                            risk = 1  # Seguridad (dentro de lo esperado)
                        elif current_val < ls:
                            risk = 2  # Alerta
                        else:
                            risk = 3  # Epidemia (por encima de lo esperado)
                        
                        df.at[idx, 'nivel_riesgo'] = risk
                        classified_count += 1
                        
                        # Log de trazabilidad
                        self.channel_log.append({
                            'departamento': dept,
                            'municipio': muni,
                            'year': year,
                            'mes': mes,
                            'casos': current_val,
                            'li_ic95': round(li, 2),
                            'media_geometrica': round(mg, 2),
                            'ls_ic95': round(ls, 2),
                            'nivel_riesgo': risk,
                            'nivel_label': RISK_LEVELS_4[risk],
                            'años_historia_usados': available_years,
                            'años_outlier_excluidos': len(outlier_years)
                        })
        
        # Mantener como float para compatibilidad con operaciones de mediana en lags
        df['nivel_riesgo'] = df['nivel_riesgo'].astype(float)
        
        # --- Reporte de Resultados ---
        total_records = len(df)
        classified_pct = (classified_count / total_records * 100) if total_records > 0 else 0
        
        logger.info(f"Canal endémico Bortman construido:")
        logger.info(f"  Registros clasificados: {classified_count}/{total_records} ({classified_pct:.1f}%)")
        logger.info(f"  Registros sin historia suficiente: {insufficient_count}")
        if self.exclude_outliers:
            logger.info(f"  Total de años-municipio excluidos como outliers: {outlier_years_total}")
        
        # Distribución de niveles de riesgo
        risk_dist = df['nivel_riesgo'].value_counts().sort_index()
        for level, count in risk_dist.items():
            if pd.notna(level):
                label = RISK_LEVELS_4.get(int(level), 'desconocido')
                pct = count / classified_count * 100 if classified_count > 0 else 0
                logger.info(f"  Nivel {int(level)} ({label}): {count} registros ({pct:.1f}%)")
        
        logger.info(f"=== CANAL ENDÉMICO BORTMAN FINALIZADO ===")
        
        return df
    
    def save_channel_log(self, output_dir):
        """Guarda el log detallado del canal endémico para auditoría."""
        if self.channel_log:
            log_df = pd.DataFrame(self.channel_log)
            log_path = output_dir / 'canal_endemico_log.csv'
            log_df.to_csv(log_path, index=False)
            logger.info(f"Log del canal endémico guardado: {log_path}")


# =============================================================================
# Feature Engineering (Actualizado para target de riesgo)
# =============================================================================
def feature_engineering(df, target_lags=3):
    """
    Agrega características de mes cíclico, rezagos de casos confirmados (como features)
    y rezagos del nivel de riesgo (nuevo target).
    
    IMPORTANTE: 'casosconfirmados' se mantiene como feature (rezagado) pero el target
    para modelado es 'nivel_riesgo'. Los rezagos de casosconfirmados capturan la magnitud
    histórica, mientras que los rezagos de nivel_riesgo capturan la tendencia de la
    clasificación epidemiológica.
    """
    df = df.copy()
    
    # --- Imputación de variables ambientales (preservado del original) ---
    ambient_cols = [c for c in df.columns if any(x in c for x in ['dew_point', 'ndvi', 'precipitacion', 'temp'])]
    if ambient_cols:
        logger.info(f"Imputando nulos en {len(ambient_cols)} columnas ambientales con la media por municipio...")
        for col in ambient_cols:
            df[col] = df.groupby(['departamento', 'municipio'])[col].transform(lambda x: x.fillna(x.mean()))
            if df[col].isnull().any():
                df[col] = df.groupby(['departamento'])[col].transform(lambda x: x.fillna(x.mean()))
            df[col] = df[col].fillna(0)

    # --- Variables cíclicas del mes ---
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    # --- Ordenamiento para lags consistentes ---
    df = df.sort_values(['departamento', 'municipio', 'year', 'mes'])
    
    # --- Rezagos de casosconfirmados (COMO FEATURE, no como target) ---
    # Estos capturan la magnitud histórica de la transmisión
    casos_lag_cols = []
    for l in range(1, target_lags + 1):
        col_name = f'casos_lag{l}'
        df[col_name] = df.groupby(['departamento', 'municipio'])['casosconfirmados'].shift(l)
        casos_lag_cols.append(col_name)
    
    if casos_lag_cols:
        df[casos_lag_cols] = df.groupby(['departamento', 'municipio'])[casos_lag_cols].transform(
            lambda x: x.fillna(x.median())
        )
        df[casos_lag_cols] = df[casos_lag_cols].fillna(0)
    
    # --- Rezagos del nivel de riesgo (capturan tendencia de la clasificación) ---
    if 'nivel_riesgo' in df.columns:
        # Convertir a float para evitar conflicto con Int64 al calcular medianas
        df['nivel_riesgo'] = df['nivel_riesgo'].astype(float)
        risk_lag_cols = []
        for l in range(1, target_lags + 1):
            col_name = f'riesgo_lag{l}'
            df[col_name] = df.groupby(['departamento', 'municipio'])['nivel_riesgo'].shift(l)
            risk_lag_cols.append(col_name)
        
        if risk_lag_cols:
            df[risk_lag_cols] = df.groupby(['departamento', 'municipio'])[risk_lag_cols].transform(
                lambda x: x.fillna(x.median())
            )
            df[risk_lag_cols] = df[risk_lag_cols].fillna(0)
    
    return df


def generate_coverage_report(df_orig, imputed_datasets, processed_dir, target_lags=3):
    """
    Genera un reporte de cobertura por departamento evaluando la
    representatividad de cada departamento en los datasets finales.
    Actualizado para incluir cobertura del canal endémico.
    """
    logger.info(f"Generando reporte de cobertura por departamento (Target Lags: {target_lags})...")
    report_rows = []
    depts = sorted(df_orig['departamento'].unique())
    
    final_sets = {m: feature_engineering(ds, target_lags=target_lags).dropna() for m, ds in imputed_datasets.items()}

    for dept in depts:
        df_dept_orig = df_orig[df_orig['departamento'] == dept]
        total_meses = len(df_dept_orig)
        meses_nulos_orig = df_dept_orig['casosconfirmados'].isnull().sum()
        
        row = {
            'departamento': dept,
            'total_meses': total_meses,
            'meses_nulos_original': meses_nulos_orig
        }

        for method in ["M1", "M2", "M3", "M4"]:
            df_imp = imputed_datasets[method]
            mask_was_null = df_dept_orig['casosconfirmados'].isnull()
            mask_is_filled = df_imp.loc[df_dept_orig.index, 'casosconfirmados'].notnull()
            row[f'meses_imputados_{method}'] = (mask_was_null & mask_is_filled).sum()
            
            incluido = dept in final_sets[method]['departamento'].unique()
            row[f'incluido_en_{method}'] = incluido
            
            # Cobertura del canal endémico (registros con nivel_riesgo asignado)
            if 'nivel_riesgo' in final_sets[method].columns:
                dept_final = final_sets[method][final_sets[method]['departamento'] == dept]
                risk_assigned = dept_final['nivel_riesgo'].notna().sum()
                row[f'registros_con_riesgo_{method}'] = risk_assigned
                row[f'pct_cobertura_riesgo_{method}'] = round(risk_assigned / len(dept_final) * 100, 1) if len(dept_final) > 0 else 0
            
            if not incluido:
                row[f'razon_exclusion_{method}'] = 'Datos insuficientes tras imputación y feature engineering'
            else:
                row[f'razon_exclusion_{method}'] = ''

        row['meses_eliminados_M5'] = meses_nulos_orig
        report_rows.append(row)
        
    pd.DataFrame(report_rows).to_csv(processed_dir / 'cobertura_departamentos.csv', index=False)


def apply_categorical_encoding(df, columns, apply=False):
    """
    Transforma variables categóricas seleccionadas en variables dummy.
    Usa drop_first=True para evitar multicolinealidad.
    """
    if not apply or not columns:
        return df
    
    logger.info(f"Aplicando codificación dummy a las columnas: {columns}")
    existing_cols = [c for c in columns if c in df.columns]
    if existing_cols:
        return pd.get_dummies(df, columns=existing_cols, drop_first=True, dtype=np.uint8)
    return df


def save_location_auxiliary(df, processed_dir):
    """
    Guarda un archivo auxiliar con las combinaciones únicas de ubicación
    (departamento, municipio, longitud, latitud) para restauración en S4.
    """
    cols_interes = ['departamento', 'municipio', 'longitud', 'latitud']
    available_cols = [c for c in cols_interes if c in df.columns]
    
    if len(available_cols) > 0:
        logger.info(f"Generando archivo auxiliar de ubicaciones con columnas: {available_cols}")
        df_coords = df[available_cols].drop_duplicates().sort_values(['departamento', 'municipio'])
        output_path = processed_dir / 'municipios_coordenadas.csv'
        df_coords.to_csv(output_path, index=False)
        logger.info(f"Archivo de coordenadas guardado en: {output_path}")
    else:
        logger.warning("No se encontraron columnas de ubicación para generar el archivo auxiliar.")


def main():
    parser = argparse.ArgumentParser(description="S2: Preprocesamiento, Imputación y Clasificación de Riesgo")
    parser.add_argument('--target_lags', type=int, default=3, help="Número de rezagos del target a generar")
    parser.add_argument('--history_years', type=int, default=ENDEMIC_CHANNEL_YEARS, 
                        help="Años de historia para el canal endémico")
    parser.add_argument('--no_exclude_outliers', action='store_true', 
                        help="Desactivar exclusión de años epidémicos atípicos")
    parser.add_argument('--skip_channel', action='store_true',
                        help="Saltar la construcción del canal endémico y usar checkpoints existentes")
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info(f"=== INICIO S2: PREPROCESAMIENTO, IMPUTACIÓN Y CLASIFICACIÓN DE RIESGO ===")
    logger.info(f"Configuración: Target Lags={args.target_lags}, "
                f"Historia Canal={args.history_years} años, "
                f"Excluir Outliers={not args.no_exclude_outliers}, "
                f"Skip Canal={args.skip_channel}")
    
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = processed_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    v2_path = Path('data/dengue_data_v2_ocur.csv')
    if v2_path.exists():
        df = pd.read_csv(v2_path, sep='|')
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        rows, cols = df.shape
        lag_cols = [c for c in df.columns if '_lag' in c]
        logger.info(f"Dataset original cargado: {rows} filas y {cols} columnas.")
        
        if lag_cols:
            base_vars_with_lags = sorted(list(set([c.split('_lag')[0] for c in lag_cols])))
            num_base_vars = len(base_vars_with_lags)
            total_lag_cols = len(lag_cols)
            lags_per_var = total_lag_cols // num_base_vars
            
            logger.info(f"Variables con rezagos detectadas ({num_base_vars}): {', '.join(base_vars_with_lags)}")
            logger.info(f"Inferencia: {lags_per_var} rezagos por variable base.")
            logger.info(f"Total de columnas de rezago: {total_lag_cols}")
        else:
            logger.info("No se detectaron columnas de rezago en el dataset original.")
        
        save_location_auxiliary(df, processed_dir)

    else:
        logger.error("Base de datos no encontrada.")
        return
    
    # =====================================================================
    # FASE 1: Imputación + Canal Endémico (pesada, ~5 horas)
    # Se guarda checkpoint después de cada método para no perder trabajo.
    # Con --skip_channel se salta esta fase y carga los checkpoints.
    # =====================================================================
    
    methods = ['M1', 'M2', 'M3', 'M4', 'M5']
    imputed_datasets = {}
    
    if args.skip_channel:
        # --- Cargar checkpoints existentes ---
        logger.info("=== CARGANDO CHECKPOINTS DEL CANAL ENDÉMICO ===")
        all_found = True
        for method in methods:
            ckpt_path = checkpoint_dir / f'checkpoint_{method}_canal.csv'
            if ckpt_path.exists():
                imputed_datasets[method] = pd.read_csv(ckpt_path)
                logger.info(f"  {method}: cargado desde checkpoint ({len(imputed_datasets[method])} registros)")
            else:
                logger.error(f"  {method}: checkpoint no encontrado en {ckpt_path}")
                all_found = False
        
        if not all_found:
            logger.error("Faltan checkpoints. Ejecute sin --skip_channel primero.")
            return
        logger.info("=== CHECKPOINTS CARGADOS EXITOSAMENTE ===")
    
    else:
        # --- Imputación ---
        logger.info("=== FASE 1A: IMPUTACIÓN DEL TARGET ===")
        imputer = TargetImputer(df)
        imputed_datasets = imputer.run_all()
        
        # --- Canal Endémico con checkpoints ---
        logger.info("=== FASE 1B: CONSTRUCCIÓN DEL CANAL ENDÉMICO ===")
        channel_builder = EndemicChannelBuilder(
            history_years=args.history_years,
            exclude_outliers=not args.no_exclude_outliers,
            min_years=3
        )
        
        for method in methods:
            if method not in imputed_datasets:
                continue
            
            # Verificar si ya existe checkpoint para este método
            ckpt_path = checkpoint_dir / f'checkpoint_{method}_canal.csv'
            if ckpt_path.exists():
                logger.info(f"--- {method}: checkpoint encontrado, cargando en lugar de recalcular ---")
                imputed_datasets[method] = pd.read_csv(ckpt_path)
                continue
            
            logger.info(f"--- Construyendo canal endémico para {method} ---")
            imputed_datasets[method] = channel_builder.build_risk_levels(imputed_datasets[method])
            
            # CHECKPOINT: guardar inmediatamente después de construir el canal
            imputed_datasets[method].to_csv(ckpt_path, index=False)
            logger.info(f"  Checkpoint guardado: {ckpt_path}")
        
        # Guardar log del canal endémico para auditoría
        channel_builder.save_channel_log(processed_dir)
    
    # =====================================================================
    # FASE 1C: Colapso a 3 Clases + Feature de Nivel Endémico
    # =====================================================================
    logger.info("=== FASE 1C: COLAPSO A 3 CLASES OPERATIVAS ===")
    COLLAPSE_MAP = {0: 0, 1: 1, 2: 1, 3: 2}  # seguridad + alerta → en_rango
    
    for method in list(imputed_datasets.keys()):
        df_m = imputed_datasets[method]
        
        # Renombrar umbral_mg → nivel_endemico (media geométrica histórica como feature)
        if 'umbral_mg' in df_m.columns:
            df_m.rename(columns={'umbral_mg': 'nivel_endemico'}, inplace=True)
        
        # Colapsar 4 niveles → 3 niveles operativos
        if 'nivel_riesgo' in df_m.columns:
            df_m['nivel_riesgo'] = df_m['nivel_riesgo'].map(COLLAPSE_MAP)
        
        imputed_datasets[method] = df_m
        
        # Reportar nueva distribución
        risk_dist = df_m['nivel_riesgo'].dropna().value_counts().sort_index()
        dist_str = " | ".join([f"{RISK_LEVELS.get(int(k), '?')}:{int(v)}" for k, v in risk_dist.items()])
        logger.info(f"  {method} (3 clases): [{dist_str}]")
    
    # =====================================================================
    # FASE 2: Feature Engineering y Exportación (liviana, ~minutos)
    # =====================================================================
    logger.info("=== FASE 2: FEATURE ENGINEERING Y EXPORTACIÓN ===")
    
    for method, df_imp in imputed_datasets.items():
        logger.info(f"Procesando dataset final para {method}...")
        try:
            df_final = feature_engineering(df_imp, target_lags=args.target_lags)
            
            # Eliminar registros sin nivel de riesgo asignado (historia insuficiente)
            before_drop = len(df_final)
            df_final = df_final.dropna(subset=['nivel_riesgo'])
            df_final = df_final.dropna()
            after_drop = len(df_final)
            logger.info(f"  Registros eliminados por datos faltantes: {before_drop - after_drop}")
            
            # Convertir columnas de riesgo a entero (seguro después de dropna)
            risk_cols = ['nivel_riesgo'] + [c for c in df_final.columns if c.startswith('riesgo_lag')]
            for col in risk_cols:
                if col in df_final.columns:
                    df_final[col] = df_final[col].round().astype(int)
            
            # Codificación categórica
            df_final = apply_categorical_encoding(df_final, CATEGORICAL_COLS, apply=APPLY_DUMMIES)
            
            # Agregar columna de etiqueta textual del nivel de riesgo
            df_final['nivel_riesgo_label'] = df_final['nivel_riesgo'].map(RISK_LEVELS)
            
            # Exportar
            output_path = processed_dir / f'dengue_imputed_{method}.csv'
            df_final.to_csv(output_path, index=False)
            
            # Resumen de distribución del target por método
            risk_dist = df_final['nivel_riesgo'].value_counts().sort_index()
            dist_str = " | ".join([f"{RISK_LEVELS.get(int(k), '?')}:{v}" for k, v in risk_dist.items()])
            logger.info(f"  Exportado: {method} con {len(df_final)} registros. Distribución: [{dist_str}]")
        
        except Exception as e:
            logger.error(f"  Error en feature engineering para {method}: {e}", exc_info=True)
            continue
    
    # --- Guardar umbrales del canal endémico como referencia ---
    umbral_cols = ['departamento', 'municipio', 'year', 'mes', 'casosconfirmados', 
                   'nivel_riesgo', 'umbral_li', 'nivel_endemico', 'umbral_ls']
    if 'M4' in imputed_datasets:
        ref_df = imputed_datasets['M4']
        available_umbral_cols = [c for c in umbral_cols if c in ref_df.columns]
        ref_umbrales = ref_df[available_umbral_cols].dropna(subset=['nivel_riesgo'])
        ref_umbrales.to_csv(processed_dir / 'umbrales_canal_endemico.csv', index=False)
        logger.info(f"Umbrales del canal endémico guardados ({len(ref_umbrales)} registros).")
    
    # --- Reporte de cobertura (al final, no bloquea exportación) ---
    try:
        generate_coverage_report(df, imputed_datasets, processed_dir, target_lags=args.target_lags)
    except Exception as e:
        logger.warning(f"Error generando reporte de cobertura (no crítico): {e}")
    
    end_time = datetime.now()
    logger.info(f"=== S2 FINALIZADO EXITOSAMENTE en {end_time - start_time} ===")


if __name__ == "__main__":
    main()
