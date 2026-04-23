import argparse
import time
import logging
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Scikit-Learn tools
from sklearn.model_selection import TimeSeriesSplit, ParameterSampler, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor

# Models — Scikit-Learn
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

# Models — External (Boosting avanzado)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# log para regresor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

# ==========================================
# CONFIGURATION & SETUP
# ==========================================

RMSE_THRESHOLD = 50.0 

logging.basicConfig(
    filename='training_execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_metrics_to_csv(model_name, best_params, mae, mse, rmse, r2, elapsed_time):
    csv_file = 'metrics_log.csv'
    file_exists = os.path.isfile(csv_file)
    data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'model': [model_name],
        'mae': [round(mae, 4)],
        'mse': [round(mse, 4)],
        'rmse': [round(rmse, 4)],
        'r2_score': [round(r2, 4)],
        'time_seconds': [round(elapsed_time, 2)],
        'best_params': [str(best_params)]
    }
    pd.DataFrame(data).to_csv(csv_file, mode='a', header=not file_exists, index=False)


# ==========================================
# GENERADOR DE MODELOS (PIPELINES)
# ==========================================
def get_models(preprocessor):
    """
    Genera el diccionario de modelos incrustando el preprocesador dinámico.
    Todos los hiperparámetros llevan el prefijo 'model__' porque el modelo
    es el segundo paso dentro de un Pipeline.
    
    Modelos disponibles:
      - Lineales: ridge, elasticnet
      - Kernel:   svr
      - Árboles:  rf, gb, hist_gb
      - Boosting avanzado: xgb, catboost, lgbm
      - Ensemble: stacked (meta-modelo)
    """
    #* Aplicar log transformación a la variable objetivo para modelos sensibles a la distribución de errores
    def make_linear_pipe(model_obj):
        # Applies log(y + 1) during fit and exp(y) - 1 during predict
        regressor = TransformedTargetRegressor(
            regressor=model_obj,
            func=np.log1p, 
            inverse_func=np.expm1
        )
        return Pipeline([('preprocessor', preprocessor), ('model', regressor)])

    return {
        # ────────────────────────────────────────
        # MODELOS LINEALES
        # ────────────────────────────────────────
        'ridge': {
            'estimator': make_linear_pipe(Ridge()), # Wrapped for Log-Transform
            'params': {
                'model__regressor__alpha': np.logspace(-5, 5, 200), # Expanded
                'model__regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
            }
        },
        'lasso': { # NEW MODEL ADDITION
            'estimator': make_linear_pipe(Lasso(random_state=42)),
            'params': {
                'model__regressor__alpha': np.logspace(-5, 2, 200),
                'model__regressor__max_iter': [1000, 5000]
            }
        },
        'elasticnet': {
            'estimator': make_linear_pipe(ElasticNet(random_state=42)),
            'params': {
                'model__regressor__alpha': np.logspace(-5, 4, 150),
                'model__regressor__l1_ratio': np.linspace(0.0, 1.0, 50),
                'model__regressor__max_iter': [2000, 5000]
            }
        },

        # ────────────────────────────────────────
        # MODELO KERNEL
        # ────────────────────────────────────────
        'svr': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', SVR())]),
            'params': {
                'model__kernel': ['linear', 'rbf'],
                'model__C': np.logspace(-2, 2, 30),
                'model__gamma': ['scale', 'auto', 0.01, 0.1]
            }
        },

        # ────────────────────────────────────────
        # MODELOS BASADOS EN ÁRBOLES (Scikit-Learn)
        # ────────────────────────────────────────
        'rf': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))]),
            'params': {
                'model__n_estimators': [100, 300, 500],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        },
        'gb': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', GradientBoostingRegressor(random_state=42))]),
            'params': {
                'model__n_estimators': [100, 300, 500],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.7, 0.8, 1.0],
                'model__min_samples_split': [2, 5, 10]
            }
        },
        'hist_gb': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', HistGradientBoostingRegressor(random_state=42))]),
            'params': {
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_iter': [100, 300, 500],
                'model__max_depth': [None, 10, 20],
                'model__l2_regularization': [0.0, 0.1, 1.0]
            }
        },

        # ────────────────────────────────────────
        # BOOSTING AVANZADO (Externos)
        # ────────────────────────────────────────
        # Ref: Ong et al. (2023) Sci. Rep. 13:19129 — XGBoost mejor AUC/F1 para dengue
        'xgb': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', XGBRegressor(
                random_state=42, 
                tree_method='hist',      # Rápido, similar a HistGB
                verbosity=0
            ))]),
            'params': {
                'model__n_estimators': [300, 500, 800],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [4, 6, 8],
                'model__subsample': [0.7, 0.8, 1.0],
                'model__colsample_bytree': [0.7, 0.8, 1.0],
                'model__reg_alpha': [0, 0.1, 1.0],
                'model__reg_lambda': [1.0, 5.0, 10.0]
            }
        },

        # Ref: Sebastianelli et al. (2024) Sci. Rep. 14:3807 — CatBoost en ensemble para dengue Brasil/Perú
        'catboost': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', CatBoostRegressor(
                random_state=42,
                verbose=0,
                allow_writing_files=False  # Evita crear carpetas de log
            ))]),
            'params': {
                'model__iterations': [300, 500, 800],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__depth': [4, 6, 8],
                'model__l2_leaf_reg': [1, 3, 5, 10],
                'model__subsample': [0.7, 0.8, 1.0]
            }
        },

        # LightGBM — Eficiencia computacional superior para iteración rápida
        'lgbm': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', LGBMRegressor(
                random_state=42,
                verbose=-1,
                n_jobs=-1
            ))]),
            'params': {
                'model__n_estimators': [300, 500, 800],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [4, 6, 8, -1],
                'model__num_leaves': [31, 63, 127],
                'model__subsample': [0.7, 0.8, 1.0],
                'model__reg_alpha': [0, 0.1, 1.0],
                'model__reg_lambda': [0, 1.0, 5.0]
            }
        },

        # ────────────────────────────────────────
        # META-MODELO STACKED ENSEMBLE
        # ────────────────────────────────────────
        # Ref: Da Re et al. (2025) Sci. Rep. 15:3750 — Stacked ML para Aedes albopictus
        # Ref: Sebastianelli et al. (2024) — Fusión CatBoost+SVM+LSTM con RF meta-learner
        'stacked': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', StackingRegressor(
                estimators=[
                    ('gb', GradientBoostingRegressor(
                        n_estimators=500, learning_rate=0.05, max_depth=5, 
                        subsample=0.8, random_state=42
                    )),
                    ('xgb', XGBRegressor(
                        n_estimators=500, learning_rate=0.05, max_depth=6, 
                        subsample=0.8, verbosity=0, random_state=42
                    )),
                    ('rf', RandomForestRegressor(
                        n_estimators=500, max_depth=20, 
                        min_samples_split=5, random_state=42
                    )),
                ],
                final_estimator=Ridge(alpha=1.0),
                cv=TimeSeriesSplit(n_splits=3),
                n_jobs=-1
            ))]),
            'params': {
                'model__final_estimator__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        }
    }


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train ML Time Series Models — Dengue Prediction Lab")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['ridge', 'lasso', 'elasticnet', 'svr', 'rf', 'gb', 'hist_gb',
                                 'xgb', 'catboost', 'lgbm', 'stacked'])
    parser.add_argument('--n_iter', type=int, default=50, help="Number of param candidates to try")
    args = parser.parse_args()
    
    model_name = args.model
    
    logging.info(f"=== INICIANDO ENTRENAMIENTO: {model_name.upper()} ===")
    print(f"--- Entrenando {model_name.upper()} ---")

    # ---------------------------------------------------------
    # 1. CARGA DE DATOS 
    # ---------------------------------------------------------
    np.random.seed(42)
    df = pd.read_csv('data/dengue_data_vfnl_ocur.csv', sep='|')
    #* filter year <=2023
    df = df[df['year'] <= 2023].reset_index(drop=True)
    
    # ---------------------------------------------------------------

    # ---------------------------------------------------------
    # 2. DEFINICIÓN DINÁMICA DEL SCALER 
    # ---------------------------------------------------------
    vars_to_scale = [
        i for i in df.columns 
        if i not in ['longitud', 'latitud', 'casosconfirmados', 'year'] 
        and not i.startswith('year_') 
        and not i.startswith('mes_') 
        and not i.startswith('fenomeno_')
    ]
    
    print(f"Variables seleccionadas para escalar automáticamente: {vars_to_scale}")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), vars_to_scale)
        ],
        remainder='passthrough'
    )

    MODELS = get_models(preprocessor)
    config = MODELS[model_name]

    # ---------------------------------------------------------
    # 3. EXTRACCIÓN Y PARTICIÓN TEMPORAL ESTRICTA
    # ---------------------------------------------------------
    X_full = df.drop(columns=['casosconfirmados']) 
    y_full = df['casosconfirmados']

    split_index = int(len(df) * 0.85)
    
    X_train, X_test = X_full.iloc[:split_index], X_full.iloc[split_index:]
    y_train, y_test = y_full.iloc[:split_index], y_full.iloc[split_index:]
    
    print(f"Total datos: {len(df)} | Train: {len(X_train)} (Pasado) | Test: {len(X_test)} (Futuro Estricto)")

    # ---------------------------------------------------------
    # 4. CONFIGURACIÓN DE LA VENTANA DESLIZANTE
    # ---------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=5)
    param_list = list(ParameterSampler(config['params'], n_iter=args.n_iter, random_state=42))
    
    best_rmse = float('inf')
    best_params = None
    start_total_time = time.time()
    
    # ---------------------------------------------------------
    # 5. BUCLE DE BÚSQUEDA (CON LOGGING POR ITERACIÓN)
    # ---------------------------------------------------------
    for i, params in enumerate(param_list):
        iter_start_time = time.time()
        model = config['estimator'].set_params(**params)
        
        try:
            cv_results = cross_validate(
                model, X_train, y_train, 
                cv=tscv, 
                scoring={'mse': 'neg_mean_squared_error', 
                         'mae': 'neg_mean_absolute_error', 
                         'r2': 'r2'},
                n_jobs=-1
            )
            
            iter_mse = -cv_results['test_mse'].mean()
            iter_mae = -cv_results['test_mae'].mean()
            iter_rmse = np.sqrt(iter_mse)
            iter_r2 = cv_results['test_r2'].mean()
            iter_time = time.time() - iter_start_time
            
            log_msg = (f"Iter {i+1}/{args.n_iter} | Tiempo: {iter_time:.2f}s | "
                       f"RMSE: {iter_rmse:.2f} | MAE: {iter_mae:.2f} | "
                       f"R2: {iter_r2:.4f} | Params: {params}")
            
            logging.info(log_msg)
            print(log_msg) 
            
            if iter_rmse < best_rmse:
                best_rmse = iter_rmse
                best_params = params
                
        except Exception as e:
            err_msg = f"Fallo en Iter {i+1} con params {params}. Error: {str(e)}"
            logging.error(err_msg)
            print(f"[ERROR] {err_msg}")
            continue

    total_time = time.time() - start_total_time
    
    # ---------------------------------------------------------
    # 6. EVALUACIÓN FINAL SOBRE EL FUTURO PURO (TEST SET)
    # ---------------------------------------------------------
    logging.info(f"=== BÚSQUEDA TERMINADA. Entrenando modelo final ===")
    print(f"\nEntrenando modelo final con mejores hiperparámetros: {best_params}")
    
    best_model = config['estimator'].set_params(**best_params)
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    
    final_mae = mean_absolute_error(y_test, y_pred)
    final_mse = mean_squared_error(y_test, y_pred)
    final_rmse = np.sqrt(final_mse)
    final_r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("MÉTRICAS FINALES SOBRE DATOS FUTUROS (TEST SET)")
    print("="*40)
    print(f"RMSE : {final_rmse:.4f}")
    print(f"MAE  : {final_mae:.4f}")
    print(f"MSE  : {final_mse:.4f}")
    print(f"R2   : {final_r2:.4f}")
    print("="*40 + "\n")
    
    # ---------------------------------------------------------
    # 7. GUARDADO DE RESULTADOS
    # ---------------------------------------------------------
    log_metrics_to_csv(model_name, best_params, final_mae, final_mse, final_rmse, final_r2, total_time)
    
    if final_rmse < RMSE_THRESHOLD:
        filename = f"{model_name}_best_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        joblib.dump(best_model, filename)
        logging.info(f"Modelo guardado: {filename} (Test RMSE: {final_rmse:.4f})")
        print(f"[ÉXITO] El modelo superó el umbral. Guardado como: {filename}")
    else:
        logging.warning(f"Modelo descartado. Test RMSE {final_rmse:.4f} > {RMSE_THRESHOLD}")
        print(f"[DESCARTADO] El modelo no superó el umbral de rendimiento.")

if __name__ == "__main__":
    main()
