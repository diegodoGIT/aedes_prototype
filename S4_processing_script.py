import pandas as pd
import numpy as np
import logging
import sys
import os
import time
import argparse
import joblib
from datetime import datetime
from pathlib import Path

# Modelos
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Herramientas de ML
from sklearn.model_selection import ParameterSampler, BaseCrossValidator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

# --- Configuración Global ---
RMSE_THRESHOLD = 15.0
LOG_FILENAME = "project_logs.log"
RESULTS_DIR = Path("processing")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Validador Espacio-Temporal ---
class SpatioTemporalSplit(BaseCrossValidator):
    """Garantiza Años íntegros, Gap de purga y Causalidad temporal."""
    def __init__(self, n_splits=5, train_period=3, val_period=2, gap=1):
        self.n_splits = n_splits
        self.train_period = train_period
        self.val_period = val_period
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None: raise ValueError("Requiere groups (columna year).")
        unique_years = sorted(np.unique(groups))
        for i in range(self.n_splits):
            train_end_idx = self.train_period + (i * self.val_period)
            gap_end_idx = train_end_idx + self.gap
            val_end_idx = gap_end_idx + self.val_period
            if val_end_idx > len(unique_years): break
            train_years = unique_years[:train_end_idx]
            val_years = unique_years[gap_end_idx:val_end_idx]
            indices = np.arange(len(groups))
            yield indices[np.isin(groups, train_years)], indices[np.isin(groups, val_years)]

# --- Funciones de Reporte y Configuración ---

def get_model_configs():
    """Define los estimadores y sus grillas de parámetros equilibradas (~5000 combinaciones)."""
    def get_tt(base):
        return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)
    
    return {
        'ridge': {
            'model': get_tt(Ridge(random_state=42)),
            'params': {
                'regressor__alpha': np.logspace(-3, 5, 10)
            }
        },
        'rf': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 400, 600, 800, 1000],
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
            }
        },
        'xgb': {
            'model': XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [2, 3, 4, 6],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'reg_lambda': [1, 10, 100],
                'reg_alpha': [0, 10],
                'gamma': [0, 0.1]
            }
        },
        'lgbm': {
            'model': LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [-1, 5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 63, 127, 255],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 10],
                'reg_lambda': [0, 10],
                'min_child_samples': [10, 20]
            }
        },
        'catboost': {
            'model': CatBoostRegressor(random_seed=42, verbose=False, allow_writing_files=False),
            'params': {
                'iterations': [100, 200, 500],
                'depth': [4, 6, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5, 7, 9, 15],
                'border_count': [64, 128, 254],
                'bagging_temperature': [0, 0.5, 1],
                'random_strength': [1, 10]
            }
        }
    }

def calculate_departmental_metrics(meta, y_true, y_pred, method, model_key, mode, fold_id, global_metrics):
    """[PUNTO CLAVE: DESEMPEÑO POR DEPARTAMENTO]"""
    df_eval = meta.copy()
    df_eval['y_true'] = y_true.values
    df_eval['y_pred'] = y_pred
    
    regional_stats = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for dept in df_eval['departamento'].unique():
        subset = df_eval[df_eval['departamento'] == dept]
        if len(subset) == 0: continue
        try:
            rmse = np.sqrt(mean_squared_error(subset['y_true'], subset['y_pred']))
            mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
            r2 = r2_score(subset['y_true'], subset['y_pred'])
        except:
            rmse, mae, r2 = np.nan, np.nan, np.nan

        regional_stats.append({
            'timestamp': ts,
            'fold': fold_id,
            'departamento': dept,
            'metodo_imputacion': method,
            'modelo': model_key,
            'modo_ejecucion': mode,
            'rmse_regional': round(rmse, 4),
            'mae_regional': round(mae, 4),
            'r2_regional': round(r2, 4),
            'rmse_global_ext': round(global_metrics.get('rmse_ext', 0), 4),
            'mae_global_ext': round(global_metrics.get('mae_ext', 0), 4),
            'r2_global_ext': round(global_metrics.get('r2_ext', 0), 4),
            'n_muestras': len(subset)
        })
    return pd.DataFrame(regional_stats)

def save_regional_results(df_regional):
    csv_path = RESULTS_DIR / "regional_performance.csv"
    file_exists = csv_path.exists()
    df_regional.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def log_iteration_to_csv(method, model_name, mode, params, metrics, duration):
    csv_path = RESULTS_DIR / "experiment_technical_log.csv"
    file_exists = csv_path.exists()
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'imputation_method': method, 'model': model_name, 'mode': mode,
        'rmse_cv_mean': round(metrics.get('rmse_cv', 0), 4),
        'mae_cv_mean': round(metrics.get('mae_cv', 0), 4),
        'r2_cv_mean': round(metrics.get('r2_cv', 0), 4),
        'rmse_external': round(metrics.get('rmse_ext', 0), 4),
        'mae_external': round(metrics.get('mae_ext', 0), 4),
        'r2_external': round(metrics.get('r2_ext', 0), 4),
        'duration_sec': round(duration, 2), 'params': str(params)
    }
    pd.DataFrame([log_data]).to_csv(csv_path, mode='a', header=not file_exists, index=False)

# --- Entrenamiento ---

def run_experiment(method, model_key, config, data_cv, is_fast=False, n_iter=5):
    (X_pool, y_pool, meta_pool), (X_val_ext, y_val_ext, meta_ext) = data_cv
    mode_label = "fast" if is_fast else "iterative"
    cv_strategy = SpatioTemporalSplit(n_splits=5, train_period=3, val_period=2, gap=1)
    
    params_to_try = [config['model'].get_params()] if is_fast else \
                    list(ParameterSampler(config['params'], n_iter=n_iter, random_state=42))
    
    best_rmse_cv = float('inf')
    best_overall_regional = [] 
    
    for params in params_to_try:
        start_time = time.time()
        model = config['model'].set_params(**params)
        
        fold_rmses = []
        fold_maes = []
        fold_r2s = []
        current_iter_regional = []
        
        # Evaluación Final Externa preliminar para métricas globales
        model.fit(X_pool, y_pool)
        preds_ext_full = model.predict(X_val_ext)
        g_metrics = {
            'rmse_ext': np.sqrt(mean_squared_error(y_val_ext, preds_ext_full)),
            'mae_ext': mean_absolute_error(y_val_ext, preds_ext_full),
            'r2_ext': r2_score(y_val_ext, preds_ext_full)
        }

        # BUCLE MANUAL DE CV PARA REGIONES
        for f_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_pool, groups=X_pool['year'])):
            X_tr, y_tr = X_pool.iloc[train_idx], y_pool.iloc[train_idx]
            X_v, y_v = X_pool.iloc[val_idx], y_pool.iloc[val_idx]
            m_v = meta_pool.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            p_v = model.predict(X_v)
            
            fold_rmses.append(np.sqrt(mean_squared_error(y_v, p_v)))
            fold_maes.append(mean_absolute_error(y_v, p_v))
            fold_r2s.append(r2_score(y_v, p_v))
            
            current_iter_regional.append(calculate_departmental_metrics(m_v, y_v, p_v, method, model_key, mode_label, f"fold_{f_idx+1}", g_metrics))

        mean_rmse_cv = np.mean(fold_rmses)
        mean_mae_cv = np.mean(fold_maes)
        mean_r2_cv = np.mean(fold_r2s)
        
        # Agregar desempeño regional de la VALIDACIÓN EXTERNA
        current_iter_regional.append(calculate_departmental_metrics(meta_ext, y_val_ext, preds_ext_full, method, model_key, mode_label, "validacion_externa", g_metrics))

        log_iteration_to_csv(method, model_key, mode_label, params, 
                            {
                                'rmse_cv': mean_rmse_cv, 
                                'mae_cv': mean_mae_cv, 
                                'r2_cv': mean_r2_cv,
                                'rmse_ext': g_metrics['rmse_ext'],
                                'mae_ext': g_metrics['mae_ext'],
                                'r2_ext': g_metrics['r2_ext']
                            }, 
                            time.time() - start_time)
        
        if mean_rmse_cv < best_rmse_cv:
            best_rmse_cv = mean_rmse_cv
            best_overall_regional = current_iter_regional
            if g_metrics['rmse_ext'] < RMSE_THRESHOLD:
                joblib.dump(model, RESULTS_DIR / f"{method}_{model_key}_best.pkl")

    if best_overall_regional:
        pd.concat(best_overall_regional, ignore_index=True).pipe(save_regional_results)

    return best_rmse_cv

def main():
    parser = argparse.ArgumentParser(description="S4: Entrenamiento Espacio-Temporal")
    parser.add_argument('--fast', type=str, choices=['ridge', 'rf', 'xgb', 'lgbm', 'catboost'])
    parser.add_argument('--n_iter', type=int, default=5)
    args = parser.parse_args()

    processed_dir = Path('data/processed')
    # Buscar solo archivos modificados por S3.5
    files = list(processed_dir.glob('dengue_imputed_M*_modified_*.csv'))
    
    if not files:
        logger.error("No se encontraron datasets procesados por S3.5 (_modified_).")
        return

    logger.info(f"=== INICIO S4: MODO {'FAST [' + args.fast + ']' if args.fast else 'ITERATIVO'} ===")

    for f_path in files:
        # Extraer método (ej: M1 de dengue_imputed_M1_modified_...)
        parts = f_path.stem.split('_')
        method = parts[2] if len(parts) > 2 else f_path.stem
        logger.info(f"Procesando: {method}")
        df = pd.read_csv(f_path)
        
        df = df[df['year'] < 2024]
        
        # [PUNTO CLAVE: SPLIT POOL CV Y VAL EXTERNA]
        df_cv = df[df['year'] <= 2020].copy()
        df_ext = df[df['year'] > 2020].copy()
        
        def split(data):
            X = data.select_dtypes(include=[np.number]).drop(columns=['casosconfirmados'], errors='ignore')
            return X, data['casosconfirmados'], data[['departamento', 'municipio', 'year']]

        data_cv = (split(df_cv), split(df_ext))
        
        scaler = StandardScaler()
        cols_to_scale = [c for c in data_cv[0][0].columns if c != 'year']
        data_cv[0][0][cols_to_scale] = scaler.fit_transform(data_cv[0][0][cols_to_scale])
        data_cv[1][0][cols_to_scale] = scaler.transform(data_cv[1][0][cols_to_scale])

        configs = get_model_configs()
        models = [args.fast] if args.fast else configs.keys()
        for m_key in models:
            run_experiment(method, m_key, configs[m_key], data_cv, args.fast is not None, args.n_iter)

    logger.info(f"=== S4 FINALIZADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()
