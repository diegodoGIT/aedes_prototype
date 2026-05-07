import pandas as pd
import numpy as np
import logging
import sys
import json
import warnings
import argparse
from datetime import datetime
from pathlib import Path

# Librerías estadísticas para diagnóstico (Copiadas de S3)
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.ensemble import RandomForestRegressor

# Silenciar warnings
warnings.filterwarnings("ignore")

# --- Configuración de Logs ---
LOG_FILENAME = "project_logs.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Clase de Diagnóstico (Copiada de S3_feature_engineering.py) ---
class DengueFeatureAnalyst:
    """
    Analizador de variables para modelos de Dengue.
    Valida: Multicolinealidad (VIF), Precedencia Temporal (Granger) y Relevancia No Lineal (RF).
    """
    def __init__(self, dataframe, method_label, target='casosconfirmados', max_lag=6):
        # Seleccionar solo datos numéricos y asegurar tipo float64
        self.df = dataframe.select_dtypes(include=[np.number]).dropna().astype(float)
        self.method = method_label
        self.target = target
        self.max_lag = max_lag
        
        # Eliminar constantes para evitar errores en VIF
        self.df = self.df.loc[:, self.df.std() > 0]

    def analyze_pearson_redundancy(self, features):
        """Identifica pares de variables con correlación superior a 0.8."""
        corr_matrix = self.df[features].corr(method='pearson', numeric_only=True)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        alerts = []
        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if abs(val) > 0.8:
                    msg = f"Las variables {row} y {col} tienen una correlación de {val:.2f}. Se recomienda eliminar una."
                    logger.info(f"[ALERTA DE REDUNDANCIA]: {msg}")
                    alerts.append(msg)
        return alerts

    def run_full_diagnostic(self):
        logger.info(f"=== INICIANDO DIAGNÓSTICO: Dataset {self.method} ===")
        
        features = [c for c in self.df.columns if c != self.target and c not in ['year', 'mes']]
        if not features:
            logger.warning("No hay suficientes variables para el análisis.")
            return

        # 0. Análisis de Correlación de Pearson (Redundancia)
        pearson_alerts = self.analyze_pearson_redundancy(features)

        # 1. Análisis de Multicolinealidad (VIF)
        try:
            X = add_constant(self.df[features])
            vif_series = pd.Series([variance_inflation_factor(X.values, i) 
                                   for i in range(X.shape[1])], index=X.columns)
        except Exception as e:
            logger.error(f"Error en cálculo de VIF: {e}")
            vif_series = pd.Series()

        # 2. Relevancia No Lineal (Random Forest)
        try:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(self.df[features], self.df[self.target])
            importances = dict(zip(features, model.feature_importances_))
        except Exception as e:
            logger.error(f"Error en Random Forest: {e}")
            importances = {}

        # 3. Evaluación de Variables
        dataset_recoms = {}
        for col in features:
            rho, _ = spearmanr(self.df[col], self.df[self.target])
            vif_val = vif_series.get(col, 0)
            rf_val = importances.get(col, 0)
            
            recoms = []
            
            # Criterio 1: Redundancia
            if vif_val > 10:
                recoms.append(f"VIF ALTO ({vif_val:.1f}): Variable redundante.")
            
            # Criterio 2: Causalidad Temporal (Granger)
            try:
                g_test = grangercausalitytests(self.df[[self.target, col]], maxlag=self.max_lag, verbose=False)
                min_p = min([g_test[i][0]['ssr_chi2test'][1] for i in range(1, self.max_lag + 1)])
                if min_p < 0.05:
                    recoms.append(f"GRANGER (p={min_p:.4f}): Precedencia temporal validada.")
            except: pass

            # Criterio 3: No Linealidad
            if abs(rho) < 0.2 and rf_val > (1.5 / len(features)):
                recoms.append(f"NO LINEAL: Baja correlación lineal pero alta importancia en RF ({rf_val:.3f}).")

            if recoms:
                dataset_recoms[col] = recoms
                logger.info(f"Diagnóstico [{col}]: " + " | ".join(recoms))

        # --- RESUMEN FINAL COMPILADO ---
        logger.info(f"=== RECOMENDACIONES COMPILADAS: Dataset {self.method} ===")
        if not pearson_alerts and not dataset_recoms:
            logger.info("Resultado: Dataset óptimo. No se detectaron redundancias ni problemas significativos.")
        else:
            if pearson_alerts:
                logger.info("--- Resumen de Redundancia Lineal (Pearson) ---")
                for alert in pearson_alerts:
                    logger.info(f" > {alert}")
            
            if dataset_recoms:
                logger.info("--- Resumen de Diagnóstico por Atributo (VIF/Granger/RF) ---")
                for col, recs in dataset_recoms.items():
                    logger.info(f" > [{col}]: " + " | ".join(recs))
        logger.info(f"=== FIN DEL RESUMEN: Dataset {self.method} ===\n")

# --- Funciones de Transformación (Movidas de S4_processing_script.py) ---

def apply_setup_preprocessing(df):
    """Aplica transformaciones manuales definidas en setup_pre_s4.JSON."""
    setup_path = Path("setup_pre_s4.JSON")
    if not setup_path.exists():
        logger.warning("No se encontró setup_pre_s4.JSON. Saltando pre-procesamiento manual.")
        return df

    try:
        with open(setup_path, 'r', encoding='utf-8') as f:
            setup = json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar setup_pre_s4.JSON: {e}")
        return df

    df_clean = df.copy()

    # 1. Promediar pares de columnas
    for pair in setup.get('average_pairs', []):
        new_col = pair.get('new_column')
        cols = pair.get('columns', [])
        if all(c in df_clean.columns for c in cols):
            logger.info(f"Aplicando promedio: {cols} -> {new_col}")
            df_clean[new_col] = df_clean[cols].mean(axis=1)
            if pair.get('drop_originals', False):
                df_clean = df_clean.drop(columns=cols)
        else:
            missing = [c for c in cols if c not in df_clean.columns]
            logger.warning(f"No se pudo promediar {cols}. Faltan: {missing}")

    # 2. Renombrar columnas
    renames = setup.get('rename_columns', {})
    if renames:
        df_clean = df_clean.rename(columns=renames)
        logger.info(f"Columnas renombradas: {renames}")

    # 3. Eliminar columnas
    drop_cols = setup.get('drop_columns', [])
    existing_drops = [c for c in drop_cols if c in df_clean.columns]
    if existing_drops:
        df_clean = df_clean.drop(columns=existing_drops)
        logger.info(f"Columnas eliminadas por setup: {existing_drops}")

    return df_clean

def main():
    parser = argparse.ArgumentParser(description="S3.5: Testing Feature Engineering")
    parser.add_argument('--target_lags', type=int, default=3, help="Número de rezagos del target")
    parser.add_argument('--moving_avg', type=int, default=3, help="Ventana de media móvil del target")
    args = parser.parse_args()

    input_path = Path("data/processed")
    files = list(input_path.glob("dengue_imputed_M*.csv"))
    
    # Filtrar archivos que ya son modified para evitar doble procesamiento
    files = [f for f in files if "_modified_" not in f.name]

    if not files:
        logger.error("No se encontraron datasets base para procesar.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== INICIO S3.5: Ingeniería de Variables del Target y Pre-procesamiento ===")

    for f_path in files:
        method = f_path.stem.split('_')[-1]
        logger.info(f"Procesando dataset: {f_path.name}")
        
        try:
            df = pd.read_csv(f_path)
            
            # 1. Aplicar transformaciones de setup_pre_s4.JSON
            df = apply_setup_preprocessing(df)
            
            # 2. Rezagos dinámicos y Media Móvil al Target
            df = df.sort_values(['departamento', 'municipio', 'year', 'mes'])
            
            # Rezagos
            for l in range(1, args.target_lags + 1):
                col = f'target_lag{l}'
                df[col] = df.groupby(['departamento', 'municipio'])['casosconfirmados'].shift(l)
                df[col] = df.groupby(['departamento', 'municipio'])[col].transform(lambda x: x.fillna(x.median())).fillna(0)
            
            # Media Móvil
            if args.moving_avg > 0:
                ma_col = f'target_ma{args.moving_avg}'
                df[ma_col] = df.groupby(['departamento', 'municipio'])['casosconfirmados'].transform(
                    lambda x: x.rolling(window=args.moving_avg, min_periods=1).mean()
                ).shift(1) # Shift 1 para evitar fuga de información
                df[ma_col] = df.groupby(['departamento', 'municipio'])[ma_col].transform(lambda x: x.fillna(x.median())).fillna(0)

            # 3. Ejecutar Diagnóstico de S3 sobre el nuevo dataset
            analyst = DengueFeatureAnalyst(df, f"{method}_modified")
            analyst.run_full_diagnostic()
            
            # 4. Guardar dataset modificado
            new_filename = f"{f_path.stem}_modified_{timestamp}.csv"
            output_path = input_path / new_filename
            df.to_csv(output_path, index=False)
            logger.info(f"Dataset guardado exitosamente: {new_filename}")

        except Exception as e:
            logger.error(f"Error procesando {f_path.name}: {e}")

    logger.info("=== S3.5 FINALIZADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()
