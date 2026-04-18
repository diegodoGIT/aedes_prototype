import argparse
import time
import logging
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Scikit-Learn tools
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import make_regression # For generating dummy data
from sklearn.model_selection import train_test_split
# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ==========================================
# CONFIGURATION & SETUP
# ==========================================

# Set a threshold for "good performance" (e.g., maximum acceptable RMSE)
# Models performing better (lower) than this threshold will be saved.
RMSE_THRESHOLD = 30.0 

# Setup Logging
logging.basicConfig(
    filename='training_execution.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_metrics_to_csv(model_name, best_params, mae, mse, rmse, r2, elapsed_time):
    """Appends model performance metrics to a CSV file."""
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
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file, mode='a', header=not file_exists, index=False)


# ==========================================
# MODEL REGISTRY & HYPERPARAMETER GRIDS
# ==========================================

# Dictionary mapping model names to their instances and parameter search spaces
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

# ==========================================
# EXPANDED MODEL REGISTRY & HYPERPARAMETER GRIDS
# ==========================================

MODELS = {
    'linear': {
        'estimator': LinearRegression(),
        'params': {
            'fit_intercept': [True, False],
            'positive': [True, False] # Forces positive coefficients (useful for some physical/economic models)
        }
    },
    'ridge': {
        'estimator': Ridge(),
        'params': {
            'alpha': np.logspace(-4, 4, 200),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'tol': [1e-4, 1e-3, 1e-2]
        }
    },
    'lasso': {
        'estimator': Lasso(),
        'params': {
            'alpha': np.logspace(-4, 4, 200),
            'selection': ['cyclic', 'random'],
            'max_iter': [1000, 5000, 10000],
            'tol': [1e-4, 1e-3]
        }
    },
    'elasticnet': { # NEW: Combines L1 and L2 penalties
        'estimator': ElasticNet(),
        'params': {
            'alpha': np.logspace(-4, 4, 200),
            'l1_ratio': np.linspace(0.01, 1.0, 50), # 1.0 is equivalent to Lasso, near 0 is Ridge
            'max_iter': [1000, 5000, 10000]
        }
    },
    'svr': {
        'estimator': SVR(),
        'params': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': np.logspace(-3, 3, 100),
            'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
            'degree': [2, 3, 4, 5], # Only used for 'poly' kernel
            'epsilon': [0.01, 0.1, 0.2, 0.5, 1.0]
        }
    },
    'rf': {
        'estimator': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 300, 500, 1000],
            'max_depth': [None, 10, 20, 30, 50, 100],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 10],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    },
    'gb': {
        'estimator': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 300, 500, 1000],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, 12],
            'subsample': [0.5, 0.7, 0.8, 1.0],
            'min_samples_split': [2, 5, 10, 20],
            'max_features': ['sqrt', 'log2', None]
        }
    },
    'hist_gb': { # NEW: Much faster for large datasets (> 10k samples)
        'estimator': HistGradientBoostingRegressor(random_state=42),
        'params': {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_iter': [100, 300, 500, 1000],
            'max_leaf_nodes': [15, 31, 63, 127],
            'max_depth': [None, 10, 20, 30],
            'min_samples_leaf': [10, 20, 50],
            'l2_regularization': [0.0, 0.1, 1.0, 10.0]
        }
    }
}
# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train and tune ML Regression Models")
    parser.add_argument('--model', type=str, required=True, choices=MODELS.keys(),
                        help="Select the model to train from the available registry")
    args = parser.parse_args()
    
    model_name = args.model
    config = MODELS[model_name]
    
    logging.info(f"Starting training process for model: {model_name.upper()}")
    print(f"--- Training {model_name.upper()} ---")

    # 1. Load Data
    # Replace this block with your actual data loading logic (e.g., pd.read_csv)
    # Using make_regression here to allow the script to run out-of-the-box
    df = pd.read_csv('data/dengue_data_vfnl_.csv', sep='|')
    X, y = df.drop(columns=['casosconfirmados']), df['casosconfirmados']

    
    # 2. Setup Cross-Validation (Folds)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Setup Hyperparameter Tuning
    # RandomizedSearchCV walks through the parameter space efficiently
    search = RandomizedSearchCV(
        estimator=config['estimator'],
        param_distributions=config['params'],
        n_iter=20, # Number of parameter settings that are sampled
        scoring='neg_mean_squared_error',
        cv=kf,
        verbose=1,
        random_state=42,
        n_jobs=-1 # Uses all available cores on the current PC
    )
    
    # 4. Train Model and Track Time
    start_time = time.time()
    search.fit(X, y)
    elapsed_time = time.time() - start_time
    
    # 5. Extract Best Model and Predict
    best_model = search.best_estimator_
    y_pred = best_model.predict(X) 
    # Note: In a strict MLOps pipeline, evaluate on a held-out test set, 
    # not the full training set X used in CV.
    
    # 6. Calculate Metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = best_model.score(X, y) # R2 Score
    
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    print(f"Best Params: {search.best_params_}")
    print(f"MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f}")
    
    # 7. Logging
    log_metrics_to_csv(model_name, search.best_params_, mae, mse, rmse, r2, elapsed_time)
    logging.info(f"{model_name.upper()} finished in {elapsed_time:.2f}s. RMSE: {rmse:.4f}")
    
    # 8. Save Model conditionally based on performance
    if rmse < RMSE_THRESHOLD:
        filename = f"{model_name}_best_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        joblib.dump(best_model, filename)
        logging.info(f"Model saved as {filename} (RMSE {rmse:.4f} < Threshold {RMSE_THRESHOLD})")
        print(f"[SUCCESS] Model saved as {filename}")
    else:
        logging.warning(f"Model NOT saved. RMSE {rmse:.4f} exceeded threshold of {RMSE_THRESHOLD}")
        print(f"[SKIPPED] Model not saved. Performance threshold not met.")

if __name__ == "__main__":
    main()