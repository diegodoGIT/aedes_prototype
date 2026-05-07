import warnings
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Estrategia robusta para silenciar warnings
warnings.filterwarnings("ignore")
def action_ignore_warnings(*args, **kwargs):
    pass
warnings.warn = action_ignore_warnings
np.seterr(all='ignore')

# Librerías estadísticas (asumiendo downgrade exitoso de pandas, numpy y statsmodels)
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.ensemble import RandomForestRegressor

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

def main():
    logger.info("=== S3: FEATURE SELECTION & DIAGNOSTIC START ===")
    
    input_path = Path("data/processed")
    files = list(input_path.glob("dengue_imputed_M*.csv"))
    
    if not files:
        logger.error("No se encontraron archivos en data/processed/")
        return

    for f in files:
        method = f.stem.split('_')[-1]
        logger.info(f"Analizando dataset: {f.name}")
        try:
            df = pd.read_csv(f)
            if 'casosconfirmados' in df.columns:
                analyst = DengueFeatureAnalyst(df, method)
                analyst.run_full_diagnostic()
        except Exception as e:
            logger.error(f"Error procesando {f.name}: {e}")

    logger.info("=== S3: FINALIZADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()
