import pandas as pd
import numpy as np
import logging
import sys
import os
import time
import argparse
import joblib
import warnings
from datetime import datetime
from pathlib import Path
from copy import deepcopy

# Silenciar warnings no críticos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
os.environ["PYTHONWARNINGS"] = "ignore"

# Modelos (Clasificadores)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Herramientas de ML
from sklearn.model_selection import ParameterSampler, BaseCrossValidator
from sklearn.metrics import (
    f1_score, accuracy_score, cohen_kappa_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler

# --- Configuración Global ---
TARGET_COL = 'nivel_riesgo'
RISK_LABELS = {0: 'exito', 1: 'en_rango', 2: 'epidemia'}
SAVE_THRESHOLD_F1 = 0.55          # Guardar modelo si macro_f1 >= este umbral
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


# =============================================================================
# Métrica Ordinal: MAE Ordinal
# =============================================================================
def ordinal_mae(y_true, y_pred):
    """
    Error Absoluto Medio entre categorías ordinales.
    
    A diferencia de la accuracy (que trata todos los errores igual),
    el MAE ordinal penaliza más las predicciones lejanas:
    - Predecir 'éxito' (0) cuando era 'epidemia' (2) = error 2
    - Predecir 'en_rango' (1) cuando era 'epidemia' (2) = error 1
    
    Rango: 0 (perfecto) a 2 (peor caso posible).
    """
    return np.mean(np.abs(np.array(y_true, dtype=int).ravel() - np.array(y_pred, dtype=int).ravel()))


def calculate_classification_metrics(y_true, y_pred):
    """
    Calcula métricas completas de clasificación ordinal.
    
    Incluye métricas globales y por clase:
    - Precision por clase: de los que el modelo predijo como nivel X, ¿cuántos realmente eran X?
    - Recall por clase: de los que realmente eran nivel X, ¿cuántos detectó el modelo?
    - FPR por clase: de los que NO eran nivel X, ¿cuántos clasificó erróneamente como X?
    """
    y_true_int = np.array(y_true, dtype=int)
    y_pred_int = np.array(y_pred, dtype=int)
    labels = [0, 1, 2]
    
    # Métricas globales
    metrics = {
        'macro_f1': f1_score(y_true_int, y_pred_int, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true_int, y_pred_int, average='weighted', zero_division=0),
        'accuracy': accuracy_score(y_true_int, y_pred_int),
        'ordinal_mae': ordinal_mae(y_true_int, y_pred_int),
        'kappa': cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic'),
        'macro_precision': precision_score(y_true_int, y_pred_int, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true_int, y_pred_int, average='macro', zero_division=0),
    }
    
    # Métricas por clase
    prec_per_class = precision_score(y_true_int, y_pred_int, average=None, labels=labels, zero_division=0)
    rec_per_class = recall_score(y_true_int, y_pred_int, average=None, labels=labels, zero_division=0)
    
    # FPR por clase: FP / (FP + TN) para cada clase
    cm = confusion_matrix(y_true_int, y_pred_int, labels=labels)
    for i, label in enumerate(labels):
        level_name = RISK_LABELS[label]
        metrics[f'precision_{level_name}'] = round(prec_per_class[i], 4)
        metrics[f'recall_{level_name}'] = round(rec_per_class[i], 4)
        
        # FPR: falsos positivos de esta clase / total de negativos reales de esta clase
        fp = cm[:, i].sum() - cm[i, i]  # columna i menos verdaderos positivos
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        metrics[f'fpr_{level_name}'] = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    
    return metrics


# =============================================================================
# Validador Espacio-Temporal (sin cambios en lógica)
# =============================================================================
class SpatioTemporalSplit(BaseCrossValidator):
    """Garantiza años íntegros, gap de purga y causalidad temporal."""
    def __init__(self, n_splits=5, train_period=3, val_period=2, gap=1):
        self.n_splits = n_splits
        self.train_period = train_period
        self.val_period = val_period
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Requiere groups (columna year).")
        unique_years = sorted(np.unique(groups))
        for i in range(self.n_splits):
            train_end_idx = self.train_period + (i * self.val_period)
            gap_end_idx = train_end_idx + self.gap
            val_end_idx = gap_end_idx + self.val_period
            if val_end_idx > len(unique_years):
                break
            train_years = unique_years[:train_end_idx]
            val_years = unique_years[gap_end_idx:val_end_idx]
            indices = np.arange(len(groups))
            yield indices[np.isin(groups, train_years)], indices[np.isin(groups, val_years)]


# =============================================================================
# Configuración de Modelos (Clasificadores)
# =============================================================================
def get_model_configs():
    """
    Define clasificadores y sus grillas de hiperparámetros.
    
    Cambios respecto a la versión de regresión:
    - Ridge -> LogisticRegression (multinomial con regularización)
    - Todos los modelos usan variantes Classifier
    - XGBoost usa objective='multi:softmax' para clasificación multiclase
    - Se eliminó TransformedTargetRegressor (no aplica a clasificación)
    """
    return {
        'logistic': {
            'model': LogisticRegression(
                random_state=42, max_iter=5000
            ),
            'params': {
                'C': np.logspace(-3, 3, 10),
                'solver': ['lbfgs', 'saga']
            }
        },
        'rf': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 400, 600, 800, 1000],
                'max_depth': [None, 5, 10, 20, 30],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }
        },
        'xgb': {
            'model': XGBClassifier(
                random_state=42, n_jobs=-1, tree_method='hist',
                objective='multi:softmax', num_class=3, eval_metric='mlogloss'
            ),
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
            'model': LGBMClassifier(
                random_state=42, n_jobs=-1, verbosity=-1,
                objective='multiclass', num_class=3
            ),
            'params': {
                'n_estimators': [100, 200, 500],
                'max_depth': [-1, 5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 63, 127, 255],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 10],
                'reg_lambda': [0, 10],
                'min_child_samples': [10, 20],
                'class_weight': [None, 'balanced']
            }
        },
        'catboost': {
            'model': CatBoostClassifier(
                random_seed=42, verbose=False, allow_writing_files=False,
                loss_function='MultiClass', eval_metric='TotalF1:average=Macro'
            ),
            'params': {
                'iterations': [100, 200, 500],
                'depth': [4, 6, 8, 10, 12],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5, 7, 9, 15],
                'border_count': [64, 128, 254],
                'bagging_temperature': [0, 0.5, 1],
                'random_strength': [1, 10],
                'auto_class_weights': ['SqrtBalanced', 'Balanced']
            }
        }
    }


# =============================================================================
# Métricas Regionales (Adaptadas para clasificación)
# =============================================================================
def calculate_departmental_metrics(meta, y_true, y_pred, method, model_key, mode, fold_id, global_metrics):
    """Calcula métricas de clasificación desglosadas por departamento."""
    df_eval = meta.copy()
    df_eval['y_true'] = np.array(y_true, dtype=int)
    df_eval['y_pred'] = np.array(y_pred, dtype=int)
    
    regional_stats = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for dept in df_eval['departamento'].unique():
        subset = df_eval[df_eval['departamento'] == dept]
        if len(subset) == 0:
            continue
        
        try:
            dept_metrics = calculate_classification_metrics(subset['y_true'], subset['y_pred'])
        except Exception:
            dept_metrics = {
                'macro_f1': np.nan, 'weighted_f1': np.nan,
                'accuracy': np.nan, 'ordinal_mae': np.nan, 'kappa': np.nan
            }
        
        # Distribución de predicciones vs real para este departamento
        true_dist = subset['y_true'].value_counts().sort_index().to_dict()
        pred_dist = subset['y_pred'].value_counts().sort_index().to_dict()
        
        regional_stats.append({
            'timestamp': ts,
            'fold': fold_id,
            'departamento': dept,
            'metodo_imputacion': method,
            'modelo': model_key,
            'modo_ejecucion': mode,
            # Métricas regionales
            'macro_f1_regional': round(dept_metrics['macro_f1'], 4),
            'weighted_f1_regional': round(dept_metrics['weighted_f1'], 4),
            'accuracy_regional': round(dept_metrics['accuracy'], 4),
            'ordinal_mae_regional': round(dept_metrics['ordinal_mae'], 4),
            'kappa_regional': round(dept_metrics['kappa'], 4),
            # Métricas globales de referencia
            'macro_f1_global': round(global_metrics.get('macro_f1_ext', 0), 4),
            'ordinal_mae_global': round(global_metrics.get('ordinal_mae_ext', 0), 4),
            # Distribución
            'dist_real': str(true_dist),
            'dist_pred': str(pred_dist),
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
        'metodo_imputacion': method,
        'modelo': model_name,
        'modo': mode,
        # --- Métricas globales - Validación Cruzada (CV) ---
        'macro_f1_score_cv': round(metrics.get('macro_f1_cv', 0), 4),
        'weighted_f1_score_cv': round(metrics.get('weighted_f1_cv', 0), 4),
        'exactitud_cv': round(metrics.get('accuracy_cv', 0), 4),
        'mae_ordinal_cv': round(metrics.get('ordinal_mae_cv', 0), 4),
        'kappa_cohen_cv': round(metrics.get('kappa_cv', 0), 4),
        'macro_precision_cv': round(metrics.get('macro_precision_cv', 0), 4),
        'macro_sensibilidad_cv': round(metrics.get('macro_recall_cv', 0), 4),
        # --- Métricas globales - Validación Externa (Temporal Hold-Out) ---
        'macro_f1_score_val': round(metrics.get('macro_f1_ext', 0), 4),
        'weighted_f1_score_val': round(metrics.get('weighted_f1_ext', 0), 4),
        'exactitud_val': round(metrics.get('accuracy_ext', 0), 4),
        'mae_ordinal_val': round(metrics.get('ordinal_mae_ext', 0), 4),
        'kappa_cohen_val': round(metrics.get('kappa_ext', 0), 4),
        'macro_precision_val': round(metrics.get('macro_precision_ext', 0), 4),
        'macro_sensibilidad_val': round(metrics.get('macro_recall_ext', 0), 4),
        # --- Precision por clase (Validación Externa) ---
        'precision_exito_val': round(metrics.get('precision_exito_ext', 0), 4),
        'precision_en_rango_val': round(metrics.get('precision_en_rango_ext', 0), 4),
        'precision_epidemia_val': round(metrics.get('precision_epidemia_ext', 0), 4),
        # --- Sensibilidad (Recall) por clase (Validación Externa) ---
        'sensibilidad_exito_val': round(metrics.get('recall_exito_ext', 0), 4),
        'sensibilidad_en_rango_val': round(metrics.get('recall_en_rango_ext', 0), 4),
        'sensibilidad_epidemia_val': round(metrics.get('recall_epidemia_ext', 0), 4),
        # --- Tasa de Falsos Positivos (FPR) por clase (Validación Externa) ---
        'tasa_fp_exito_val': round(metrics.get('fpr_exito_ext', 0), 4),
        'tasa_fp_en_rango_val': round(metrics.get('fpr_en_rango_ext', 0), 4),
        'tasa_fp_epidemia_val': round(metrics.get('fpr_epidemia_ext', 0), 4),
        # --- Meta ---
        'duracion_seg': round(duration, 2),
        'parametros': str(params)
    }
    pd.DataFrame([log_data]).to_csv(csv_path, mode='a', header=not file_exists, index=False)


def log_confusion_matrix(y_true, y_pred, method, model_key, fold_label):
    """Registra la matriz de confusión en el log para diagnóstico."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    logger.info(f"  Matriz de Confusión [{method}/{model_key}/{fold_label}]:")
    header = "           " + "  ".join([f"{RISK_LABELS[i]:>10s}" for i in range(3)])
    logger.info(f"  {header}")
    for i in range(3):
        row_str = "  ".join([f"{cm[i][j]:>10d}" for j in range(3)])
        logger.info(f"  {RISK_LABELS[i]:>10s}  {row_str}")


# =============================================================================
# Entrenamiento (Adaptado para clasificación)
# =============================================================================
def run_experiment(method, model_key, config, data_cv, is_fast=False, n_iter=5):
    (X_pool, y_pool, meta_pool), (X_val_ext, y_val_ext, meta_ext) = data_cv
    mode_label = "fast" if is_fast else "iterative"
    cv_strategy = SpatioTemporalSplit(n_splits=5, train_period=3, val_period=2, gap=1)
    
    params_to_try = [config['model'].get_params()] if is_fast else \
                    list(ParameterSampler(config['params'], n_iter=n_iter, random_state=42))
    
    best_macro_f1_cv = -1.0
    best_overall_regional = []
    
    for params in params_to_try:
        start_time = time.time()
        
        # Crear instancia nueva para cada iteración (CatBoost no permite set_params tras fit)
        model = deepcopy(config['model'])
        try:
            model.set_params(**params)
        except Exception:
            # Si set_params falla (ej: CatBoost fitted), crear instancia fresca
            model = type(config['model'])(**{**config['model'].get_params(), **params})
        
        fold_metrics = {k: [] for k in [
            'macro_f1', 'weighted_f1', 'accuracy', 'ordinal_mae', 'kappa',
            'macro_precision', 'macro_recall'
        ]}
        current_iter_regional = []
        
        # --- Evaluación Externa preliminar ---
        model.fit(X_pool, y_pool)
        preds_ext_full = model.predict(X_val_ext)
        ext_metrics = calculate_classification_metrics(y_val_ext, preds_ext_full)
        
        g_metrics = {f'{k}_ext': v for k, v in ext_metrics.items()}

        # --- CV Espacio-Temporal ---
        for f_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_pool, groups=X_pool['year'])):
            X_tr, y_tr = X_pool.iloc[train_idx], y_pool.iloc[train_idx]
            X_v, y_v = X_pool.iloc[val_idx], y_pool.iloc[val_idx]
            m_v = meta_pool.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            p_v = model.predict(X_v)
            
            fold_m = calculate_classification_metrics(y_v, p_v)
            for k in fold_metrics:
                fold_metrics[k].append(fold_m[k])
            
            current_iter_regional.append(
                calculate_departmental_metrics(
                    m_v, y_v, p_v, method, model_key, mode_label,
                    f"fold_{f_idx+1}", g_metrics
                )
            )

        # Promedios de CV
        cv_means = {f'{k}_cv': np.mean(v) for k, v in fold_metrics.items()}
        
        # Regional de validación externa
        current_iter_regional.append(
            calculate_departmental_metrics(
                meta_ext, y_val_ext, preds_ext_full, method, model_key,
                mode_label, "validacion_externa", g_metrics
            )
        )

        # Combinar métricas para log
        all_metrics = {**cv_means, **g_metrics}
        log_iteration_to_csv(method, model_key, mode_label, params, all_metrics, time.time() - start_time)
        
        # Log resumido con nomenclatura estándar
        rec_epi = ext_metrics.get('recall_epidemia', 0)
        prec_epi = ext_metrics.get('precision_epidemia', 0)
        logger.info(
            f"  [{model_key}] F1-Score={cv_means['macro_f1_cv']:.3f} | "
            f"MAE-Ordinal={cv_means['ordinal_mae_cv']:.3f} | "
            f"Precision={cv_means.get('macro_precision_cv', 0):.3f} | "
            f"Sensibilidad={cv_means.get('macro_recall_cv', 0):.3f} | "
            f"F1-Score_val={ext_metrics['macro_f1']:.3f} | "
            f"Sensib_epidemia={rec_epi:.3f} | "
            f"Precis_epidemia={prec_epi:.3f}"
        )
        
        # Selección del mejor por macro F1 de CV
        if cv_means['macro_f1_cv'] > best_macro_f1_cv:
            best_macro_f1_cv = cv_means['macro_f1_cv']
            best_overall_regional = current_iter_regional
            
            # Guardar modelo si supera umbral
            if ext_metrics['macro_f1'] >= SAVE_THRESHOLD_F1:
                model_path = RESULTS_DIR / f"{method}_{model_key}_best.pkl"
                joblib.dump(model, model_path)
                logger.info(f"  Modelo guardado: {model_path} (macro_f1_val={ext_metrics['macro_f1']:.3f})")
                
                # Guardar predicciones con etiquetas de texto
                pred_df = meta_ext.copy()
                pred_df['nivel_riesgo_real'] = y_val_ext.values
                pred_df['nivel_riesgo_predicho'] = preds_ext_full
                pred_df['etiqueta_real'] = pred_df['nivel_riesgo_real'].map(RISK_LABELS)
                pred_df['etiqueta_predicha'] = pred_df['nivel_riesgo_predicho'].map(RISK_LABELS)
                pred_path = RESULTS_DIR / f"{method}_{model_key}_predicciones.csv"
                pred_df.to_csv(pred_path, index=False)
                logger.info(f"  Predicciones guardadas: {pred_path}")
                
                # Matriz de confusión del mejor modelo
                log_confusion_matrix(y_val_ext, preds_ext_full, method, model_key, "val_best")

    if best_overall_regional:
        pd.concat(best_overall_regional, ignore_index=True).pipe(save_regional_results)

    return best_macro_f1_cv


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="S4: Entrenamiento Espacio-Temporal (Clasificación de Riesgo)")
    parser.add_argument('--fast', type=str, choices=['logistic', 'rf', 'xgb', 'lgbm', 'catboost'])
    parser.add_argument('--n_iter', type=int, default=5)
    args = parser.parse_args()

    processed_dir = Path('data/processed')
    files = list(processed_dir.glob('dengue_imputed_M*_modified_*.csv'))
    
    if not files:
        logger.error("No se encontraron datasets procesados por S3.5 (_modified_).")
        return

    logger.info(f"=== INICIO S4: CLASIFICACIÓN DE RIESGO — "
                f"MODO {'FAST [' + args.fast + ']' if args.fast else 'ITERATIVO'} ===")
    logger.info(f"Target: {TARGET_COL} | Niveles: {RISK_LABELS}")

    for f_path in files:
        parts = f_path.stem.split('_')
        method = parts[2] if len(parts) > 2 else f_path.stem
        logger.info(f"Procesando: {method} ({f_path.name})")
        df = pd.read_csv(f_path)
        
        # --- Verificar presencia del target ---
        if TARGET_COL not in df.columns:
            logger.error(f"'{TARGET_COL}' no encontrado en {f_path.name}. Saltando.")
            continue
        
        # --- Recuperación de Metadatos Geográficos ---
        aux_path = processed_dir / 'municipios_coordenadas.csv'
        if aux_path.exists() and ('departamento' not in df.columns or 'municipio' not in df.columns):
            logger.info(f"Restaurando metadatos geográficos desde {aux_path}")
            df_aux = pd.read_csv(aux_path)
            for col in ['longitud', 'latitud']:
                if col in df.columns and col in df_aux.columns:
                    df[col] = df[col].astype(float)
                    df_aux[col] = df_aux[col].astype(float)
            df = df.merge(df_aux, on=['longitud', 'latitud'], how='left')

        # --- Filtrado temporal ---
        df = df[df['year'] < 2024]
        
        # Eliminar registros sin target válido
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)
        
        # --- Distribución del target ---
        dist = df[TARGET_COL].value_counts().sort_index()
        logger.info(f"  Distribución del target:")
        for level, count in dist.items():
            pct = count / len(df) * 100
            label = RISK_LABELS.get(level, '?')
            logger.info(f"    {level} ({label}): {count} ({pct:.1f}%)")
        
        # Agregar columna de texto para el nivel de riesgo
        df['nivel_riesgo_label'] = df[TARGET_COL].map(RISK_LABELS)
        
        # --- Split CV Pool vs Validación Externa ---
        df_cv = df[df['year'] <= 2020].copy()
        df_ext = df[df['year'] > 2020].copy()
        
        if len(df_cv) == 0 or len(df_ext) == 0:
            logger.warning(f"Split vacío para {method}. CV: {len(df_cv)}, Ext: {len(df_ext)}. Saltando.")
            continue
        
        def split(data):
            """Separa features numéricas, target ordinal y metadatos."""
            drop_cols = [TARGET_COL, 'nivel_riesgo_label']
            X = data.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')
            y = data[TARGET_COL].astype(int)
            
            # Metadatos para reporte regional
            meta_cols = ['departamento', 'municipio', 'year']
            available_meta = [c for c in meta_cols if c in data.columns]
            meta = data[available_meta] if available_meta else pd.DataFrame(index=data.index)
            
            return X, y, meta

        data_cv = (split(df_cv), split(df_ext))
        
        # --- Escalado (solo features, no target) ---
        scaler = StandardScaler()
        cols_to_scale = [c for c in data_cv[0][0].columns if c != 'year']
        
        if cols_to_scale:
            data_cv[0][0][cols_to_scale] = scaler.fit_transform(data_cv[0][0][cols_to_scale])
            data_cv[1][0][cols_to_scale] = scaler.transform(data_cv[1][0][cols_to_scale])

        # --- Entrenamiento ---
        configs = get_model_configs()
        models_to_run = [args.fast] if args.fast else configs.keys()
        
        for m_key in models_to_run:
            logger.info(f"--- Entrenando: {m_key} ---")
            best_f1 = run_experiment(
                method, m_key, configs[m_key], data_cv,
                args.fast is not None, args.n_iter
            )
            logger.info(f"  Mejor macro_f1_cv para {m_key}: {best_f1:.4f}")

    logger.info(f"=== S4 FINALIZADO EXITOSAMENTE ===")


if __name__ == "__main__":
    main()
