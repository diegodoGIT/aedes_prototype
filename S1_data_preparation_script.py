import pandas as pd
import numpy as np
import rasterio
import logging
import sys
from datetime import datetime
from unidecode import unidecode
from pathlib import Path
from thefuzz import process, fuzz

# --- Configuración de Logging ---
# Se configura para imprimir en consola y guardar en 'project_logs.log' como log general del proyecto
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

# --- Utilidades de Procesamiento ---
def normalize_str(s):
    """
    Normaliza cadenas de texto: quita acentos, convierte a minúsculas,
    elimina prefijos numéricos comunes en DIVIPOLA y espacios extra.
    """
    if not isinstance(s, str): return s
    # El split('-')[-1] ayuda a manejar casos como '05-medellin' -> 'medellin'
    return unidecode(str(s).split('-')[-1].strip().lower())

def lag_features(df, group_cols, target_cols, lags):
    """
    Crea variables de rezago (lags) temporales.
    group_cols: Columnas para agrupar (ej. municipio) y asegurar que el lag sea consistente.
    target_cols: Columnas a las que se les aplicará el rezago.
    lags: Rango o lista de meses de retraso.
    """
    for target in target_cols:
        for lag in lags:
            lag_col_name = f"{target}_lag{lag}"
            if group_cols:
                df[lag_col_name] = df.groupby(group_cols)[target].shift(lag)
            else:
                df[lag_col_name] = df[target].shift(lag)
    return df

# --- Funciones de Carga y Transformación ---
def load_db(f_path, sheet_name):
    """
    Carga el Excel de SISPRO, limpia nombres de columnas y transforma 
    de formato ancho (meses como columnas) a formato largo (tidy data).
    """
    logger.info(f"Cargando base de datos desde {f_path}, hoja: {sheet_name}...")
    df = pd.read_excel(f_path, sheet_name=sheet_name)
    df.fillna(0, inplace=True)
    df.columns = [normalize_str(c) for c in df.columns]
    df.rename(columns={'ano epidemiologico': 'year'}, inplace=True)
    
    # Filtrado de registros inválidos
    df = df[~df['municipio'].str.contains('Sin Informacion', na=False)]
    df = df[df['pais'] != 'no definido']
    
    for col in ['pais', 'departamento', 'municipio']:
        df[col] = df[col].apply(normalize_str)

    # Transformación Melt para normalizar la serie de tiempo
    df = df.melt(
        id_vars=['pais', 'departamento', 'municipio', 'year'],
        var_name='mes', value_name='casosconfirmados'
    )
    
    month_names = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 
                   'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    mes_mapping = {name: i+1 for i, name in enumerate(month_names)}
    df['mes'] = df['mes'].map(mes_mapping)
    return df

def fill_location_timeseries(df):
    """
    Asegura que cada municipio tenga una serie de tiempo continua, 
    rellenando meses faltantes con NaN para evitar saltos en los lags.
    """
    logger.info("Completando series de tiempo por ubicación...")
    df['date'] = pd.to_datetime(pd.DataFrame({'year': df['year'], 'month': df['mes'], 'day': 1}))
    location_cols = ['departamento', 'municipio']
    
    df_complete = (
        df.set_index('date')
        .groupby(location_cols)
        .resample('MS')
        .asfreq()
        .drop(columns=location_cols, errors='ignore')
        .reset_index()
    )
    df_complete['year'] = df_complete['date'].dt.year
    df_complete['mes'] = df_complete['date'].dt.month
    return df_complete.sort_values(location_cols + ['date']).reset_index(drop=True)

# --- Integración de Datos Ambientales ---
def add_worldclim(df, lags=8):
    """
    Extrae datos biofísicos de archivos Raster (.tif):
    1. Elevación (estática por coordenadas).
    2. Temperatura y Precipitación mensual (dinámica por año/mes).
    """
    base_path = Path('data/worldclim')
    elev_path = base_path / 'wc2.1_2.5m_elev/wc2.1_2.5m_elev.tif'
    climate_folders = {
        'tmin': [base_path / f'wc2.1_cruts4.09_2.5m_tmin_{r}' for r in ['2000-2009', '2010-2019', '2020-2024']],
        'tmax': [base_path / f'wc2.1_cruts4.09_2.5m_tmax_{r}' for r in ['2000-2009', '2010-2019', '2020-2024']],
        'prec': [base_path / f'wc2.1_cruts4.09_2.5m_prec_{r}' for r in ['2000-2009', '2010-2019', '2020-2024']]
    }

    # Carga y limpieza de DIVIPOLA para georreferenciación
    logger.info("Cargando DIVIPOLA y armonizando nombres de municipios...")
    divipola = pd.read_excel('data/Listados_DIVIPOLA.xlsx', sheet_name='Municipios', skiprows=10, skipfooter=10)
    divipola.columns = ['cod_dept', 'departamento', 'cod_muni', 'municipio', 'tipo', 'longitud', 'latitud', 'nota']
    divipola = divipola[['departamento', 'municipio', 'cod_muni', 'tipo', 'longitud', 'latitud']]
    for col in ['departamento', 'municipio']: divipola[col] = divipola[col].apply(normalize_str)

    # Diccionario de homologación para cruce exitoso entre SIVIGILA y DANE
    renamer = {
        'putumayo': {'leguizamo': 'puerto leguizamo'},
        'cauca': {'sotara': 'sotara paispamba', 'piendamo': 'piendamo - tunia', 'lopez': 'lopez de micay'},
        'valle del cauca': {'cali': 'santiago de cali'}
    }
    for dept, m_map in renamer.items():
        df.loc[df['departamento'] == dept, 'municipio'] = df.loc[df['departamento'] == dept, 'municipio'].replace(m_map)

    df['loc_key'] = df['departamento'] + "_" + df['municipio']
    divipola['loc_key'] = divipola['departamento'] + "_" + divipola['municipio']
    
    df = df[df['loc_key'].isin(divipola['loc_key'])]
    df = df.merge(divipola.drop(columns=['departamento', 'municipio']), on='loc_key', how='left')

    unique_locs = df[['loc_key', 'longitud', 'latitud']].drop_duplicates().dropna()
    coords = list(zip(unique_locs['longitud'], unique_locs['latitud']))

    # Extracción de Elevación mediante Rasterio
    if elev_path.exists():
        logger.info("Extrayendo elevación de archivos Raster...")
        with rasterio.open(elev_path) as src:
            unique_locs['elevacion'] = [val[0] for val in src.sample(coords)]
        df = df.merge(unique_locs[['loc_key', 'elevacion']], on='loc_key', how='left')

    # Procesamiento dinámico de Clima con Lags temporales
    logger.info(f"Procesando variables climáticas mensuales y generando {lags} lags...")
    all_dates = pd.date_range(start=df['date'].min() - pd.DateOffset(months=lags), end=df['date'].max(), freq='MS')
    climate_blocks = []
    
    for dt in all_dates:
        y, m = dt.year, dt.month
        idx = 0 if y <= 2009 else (1 if y <= 2019 else 2)
        month_data = {'loc_key': unique_locs['loc_key'].tolist(), 'year': y, 'mes': m}
        for var, folders in climate_folders.items():
            folder = folders[idx]
            f_base = folder.name.rsplit('_', 1)[0]
            f_name = f"{f_base}_{y}-{m:02d}.tif"
            f_path = folder / f_name
            if f_path.exists():
                with rasterio.open(f_path) as src:
                    month_data[var] = [val[0] for val in src.sample(coords)]
            else:
                month_data[var] = [np.nan] * len(unique_locs)
        climate_blocks.append(pd.DataFrame(month_data))

    climate_df = pd.concat(climate_blocks).sort_values(['loc_key', 'year', 'mes'])
    climate_df = lag_features(climate_df, ['loc_key'], ['prec', 'tmax', 'tmin'], range(1, lags + 1))
    
    final_df = df.merge(climate_df, on=['loc_key', 'year', 'mes'], how='left')
    
    # Estandarización de nombres de columnas de clima
    rename_dict = {'prec': 'Precipitacion', 'tmax': 'TempMax', 'tmin': 'TempMin'}
    for lag in range(1, lags + 1):
        rename_dict.update({f'prec_lag{lag}': f'Precipitacion_lag{lag}', 
                           f'tmax_lag{lag}': f'TempMax_lag{lag}', 
                           f'tmin_lag{lag}': f'TempMin_lag{lag}'})
    
    final_df.rename(columns=rename_dict, inplace=True)
    return final_df

def add_enso_data(df, lags=8):
    """
    Integra datos del Niño/Niña (ENSO) desde anomalías de temperatura superficial del mar.
    Categoriza la intensidad del fenómeno y genera lags.
    """
    logger.info("Integrando datos del fenómeno ENSO (Niño/Niña)...")
    df_nino = pd.read_csv('data/nino34.long.anom.csv')
    df_nino.columns = ['date', 'gravedad']
    df_nino['date'] = pd.to_datetime(df_nino['date'])
    df_nino['gravedad'] = df_nino['gravedad'].replace(-99.99, np.nan)
    
    # Marcación categórica según umbrales estándar de la NOAA
    df_nino['fenomeno'] = df_nino['gravedad'].apply(
        lambda x: 'nino_sev' if x >= 1.5 else ('nino_mod' if x >= 0.5 else 
                  ('neutral' if x > -0.5 else ('nina_mod' if x > -1.5 else 'nina_sev')))
    )
    df_nino = lag_features(df_nino, None, ['fenomeno'], range(1, lags + 1))
    return df.merge(df_nino.drop(columns='gravedad'), on='date', how='left')

def add_additional_env_vars(df, lags=8):
    """
    Integra variables ambientales adicionales:
    - NDVI: Índice de vegetación.
    - Dew Point: Punto de rocío (proxy de humedad).
    """
    logger.info("Integrando variables NDVI y Dew Point...")
    vars_amb = pd.read_csv('data/climaticas_aedes_colombia.csv')
    vars_amb.rename(columns={'ADM2_NAME': 'municipio', 'ADM1_NAME': 'departamento', 'month':'mes'}, inplace=True)
    vars_amb = vars_amb[(vars_amb['year'] >= 2006) & (vars_amb['year'] <= 2024)]
    vars_amb.columns = [normalize_str(c) for c in vars_amb.columns]
    
    for col in ['departamento', 'municipio']: vars_amb[col] = vars_amb[col].apply(normalize_str)
    
    # Armonización específica para Bogotá
    vars_amb.loc[vars_amb['municipio'] == 'santafe de bogota d.c.', 'departamento'] = 'bogota, d.c.'
    
    vars_amb['key'] = vars_amb['departamento'] + "_" + vars_amb['municipio']
    # Correcciones manuales para asegurar el cruce
    manual_map = {
        'valle del cauca_cali': 'valle del cauca_santiago de cali', 
        'bogota, d.c._santafe de bogota d.c.': 'bogota, d.c._bogota, d.c.'
    }
    vars_amb['key'] = vars_amb['key'].replace(manual_map)
    
    vars_amb.drop(columns=['lst', 'temp_air', 'precip'], inplace=True, errors='ignore')
    # Imputación por media local para evitar vacíos en dew_point
    vars_amb['dew_point'] = vars_amb.groupby('key')['dew_point'].transform(lambda x: x.fillna(x.mean()))
    vars_amb = lag_features(vars_amb, ['key'], ['ndvi', 'dew_point'], range(1, lags + 1))
    
    return df.merge(vars_amb.drop(columns=['departamento', 'municipio']), on=['key', 'year', 'mes'], how='left')

# --- Ejecución del Pipeline ---
if __name__ == "__main__":
    start_time = datetime.now()
    HOJA = 'ocur' # Opciones: 'ocur' (ocurrencia), 'res' (residencia)
    LAG_SIZE = 3
    
    logger.info(f"=== INICIO DEL PROCESAMIENTO DE DATOS (Modo: {HOJA}) ===")
    
    try:
        # 1. Carga de casos y expansión temporal
        df = load_db('data/Base_deng.xlsx', sheet_name=HOJA)
        df = fill_location_timeseries(df)
        
        # 2. Integración WorldClim (Raster)
        df = add_worldclim(df, lags=LAG_SIZE)
        
        # 3. Integración ENSO
        df = add_enso_data(df, lags=LAG_SIZE)
        
        # 4. Integración NDVI y Humedad
        df['key'] = df['departamento'] + "_" + df['municipio']
        df = add_additional_env_vars(df, lags=LAG_SIZE)
        
        # Limpieza final de claves auxiliares
        df.drop(columns=['loc_key', 'key', 'pais'], inplace=True, errors='ignore')
        
        # 5. Almacenamiento y Metadatos
        output_path = f'data/dengue_data_v2_{HOJA}.csv'
        df.to_csv(output_path, index=False, sep='|')
        
        # Registro de información crítica sobre el archivo generado
        last_data_date = df['date'].max().strftime('%Y-%m-%d')
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=== PROCESAMIENTO FINALIZADO EXITOSAMENTE ===")
        logger.info(f"Archivo generado: {output_path}")
        logger.info(f"Última fecha de dengue disponible en el archivo: {last_data_date}")
        logger.info(f"Duración total: {duration}")
        logger.info(f"Registros procesados: {len(df)}")

    except Exception as e:
        logger.error(f"Error crítico durante el procesamiento: {str(e)}", exc_info=True)
        sys.exit(1)
