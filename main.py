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

# Models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

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
        'mae_test': [round(mae, 4)],
        'mse_test': [round(mse, 4)],
        'rmse_test': [round(rmse, 4)],
        'r2_test': [round(r2, 4)],
        'total_time_sec': [round(elapsed_time, 2)],
        'best_params': [str(best_params)]
    }
    pd.DataFrame(data).to_csv(csv_file, mode='a', header=not file_exists, index=False)


# ==========================================
# GENERADOR DE MODELOS (PIPELINES)
# ==========================================
def get_models(preprocessor):
    """
    Genera el diccionario de modelos incrustando el preprocesador dinámico.
    Nota: Todos los hiperparámetros llevan el prefijo 'model__' porque ahora
    el modelo es el segundo paso dentro de un Pipeline.
    """
    return {
        'ridge': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', Ridge())]),
            'params': {
                'model__alpha': np.logspace(-4, 4, 100),
                'model__solver': ['auto', 'svd', 'cholesky', 'sag']
            }
        },
        'elasticnet': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', ElasticNet(random_state=42))]),
            'params': {
                'model__alpha': np.logspace(-4, 4, 50),
                'model__l1_ratio': np.linspace(0.01, 1.0, 20),
                'model__max_iter': [1000, 5000]
            }
        },
        'svr': {
            'estimator': Pipeline([('preprocessor', preprocessor), ('model', SVR())]),
            'params': {
                'model__kernel': ['linear', 'rbf'],
                'model__C': np.logspace(-2, 2, 30),
                'model__gamma': ['scale', 'auto', 0.01, 0.1]
            }
        },
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
                'model__max_depth': [3, 5, 7]
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
        }
    }


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train ML Time Series Models")
    # Usamos una lista temporal para parsear, ya que el dict real se crea después de cargar datos
    parser.add_argument('--model', type=str, required=True, 
                        choices=['ridge', 'elasticnet', 'svr', 'rf', 'gb', 'hist_gb'])
    parser.add_argument('--n_iter', type=int, default=50, help="Number of param candidates to try")
    args = parser.parse_args()
    
    model_name = args.model
    
    logging.info(f"=== INICIANDO ENTRENAMIENTO: {model_name.upper()} ===")
    print(f"--- Entrenando {model_name.upper()} ---")

    # ---------------------------------------------------------
    # 1. CARGA DE DATOS 
    # ---------------------------------------------------------
    # Aquí cargarías tu dataset real:
    # df = pd.read_csv('tus_datos.csv')
    # df = df.sort_values(by=['year', 'month']).reset_index(drop=True)
    
    # --- Simulación de tu estructura de datos para que el script corra ---
    np.random.seed(42)
    df = pd.read_csv('data/dengue_data_vfnl_ocur.csv', sep='|')
    #* filter year <=2022
    df = df[df['year'] <= 2022].reset_index(drop=True)
    
    # ---------------------------------------------------------------

    # ---------------------------------------------------------
    # 2. DEFINICIÓN DINÁMICA DEL SCALER 
    # ---------------------------------------------------------
    # Identificar variables a escalar según tu lógica exacta
    vars_to_scale = [
        i for i in df.columns 
        if i not in ['longitud', 'latitud', 'casosconfirmados', 'year'] 
        and not i.startswith('year_') 
        and not i.startswith('mes_') 
        and not i.startswith('fenomeno_')
    ]
    
    print(f"Variables seleccionadas para escalar automáticamente: {vars_to_scale}")

    # Crear el transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), vars_to_scale)
        ],
        remainder='passthrough' # 'passthrough' deja latitud, longitud y dummies tal como están
    )

    # Ahora que tenemos el preprocesador, generamos el registro de modelos
    MODELS = get_models(preprocessor)
    config = MODELS[model_name]

    # ---------------------------------------------------------
    # 3. EXTRACCIÓN Y PARTICIÓN TEMPORAL ESTRICTA
    # ---------------------------------------------------------
    X_full = df.drop(columns=['casosconfirmados']) 
    y_full = df['casosconfirmados']

    # Cortamos el último 15% como Test Set PURO
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
    
    # Entrenar en TODO el histórico. El pipeline escalará usando SOLO estos datos.
    best_model = config['estimator'].set_params(**best_params)
    best_model.fit(X_train, y_train)
    
    # Predecir futuro. El pipeline aplicará el scaler aprendido en Train a este Test set.
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