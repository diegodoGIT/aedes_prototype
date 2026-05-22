import pandas as pd
import numpy as np
import logging
import sys
import json
import warnings
import argparse
from datetime import datetime
from pathlib import Path

# Librerías estadísticas para diagnóstico
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

# --- Columnas que NO deben ser features (prevención de data leakage) ---
# casosconfirmados: insumo directo del canal endémico que genera nivel_riesgo
# umbral_p*: son los percentiles calculados para clasificar, filtrarían la respuesta
LEAKAGE_COLS = ['casosconfirmados', 'umbral_p25', 'umbral_p50', 'umbral_p75']
# Columnas excluidas del diagnóstico pero no necesariamente eliminadas del dataset
EXCLUDE_FROM_DIAGNOSTIC = ['year', 'mes', 'latitud', 'longitud'] + LEAKAGE_COLS


# =============================================================================
# Clase de Diagnóstico (Adaptada para target ordinal)
# =============================================================================
class DengueRiskAnalyst:
    """
    Analizador de variables para modelos de Riesgo de Dengue (target ordinal).
    
    Cambios respecto a DengueFeatureAnalyst (S3):
    - RandomForestClassifier en lugar de Regressor (target es ordinal 0-3).
    - Spearman se mantiene (válido para correlación ordinal vs continua).
    - Granger se mantiene con precaución: el target ordinal se trata como
      serie numérica para evaluar precedencia temporal. Los resultados son
      orientativos, no definitivos, para variables ordinales.
    - Se agrega análisis de separabilidad por clase (Kruskal-Wallis) para
      evaluar si cada feature discrimina entre niveles de riesgo.
    """
    def __init__(self, dataframe, method_label, target='nivel_riesgo', max_lag=6):
        # Seleccionar solo datos numéricos
        self.df = dataframe.select_dtypes(include=[np.number]).dropna().astype(float)
        self.method = method_label
        self.target = target
        self.max_lag = max_lag
        
        # Eliminar constantes para evitar errores en VIF
        self.df = self.df.loc[:, self.df.std() > 0]
        
        # Verificar que el target existe y tiene varianza
        if self.target not in self.df.columns:
            logger.error(f"Target '{self.target}' no encontrado en el dataset.")
            self.df = pd.DataFrame()

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

    def analyze_class_separability(self, features):
        """
        Evalúa si cada feature discrimina entre niveles de riesgo usando Kruskal-Wallis.
        
        Una feature con p-valor < 0.05 tiene distribuciones significativamente distintas
        entre al menos dos niveles de riesgo, lo que indica poder discriminativo.
        Features con p > 0.05 aportan poco para distinguir entre categorías.
        """
        from scipy.stats import kruskal
        
        separability_results = {}
        risk_levels = sorted(self.df[self.target].unique())
        
        if len(risk_levels) < 2:
            logger.warning("Menos de 2 niveles de riesgo presentes. No se puede evaluar separabilidad.")
            return separability_results
        
        for col in features:
            groups = [self.df[self.df[self.target] == level][col].values for level in risk_levels]
            # Filtrar grupos vacíos
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) < 2:
                continue
            
            try:
                stat, p_val = kruskal(*groups)
                separability_results[col] = {'statistic': stat, 'p_value': p_val}
                if p_val > 0.05:
                    logger.info(f"[BAJA SEPARABILIDAD] {col}: p={p_val:.4f} — No discrimina entre niveles de riesgo.")
            except Exception:
                pass
        
        return separability_results

    def run_full_diagnostic(self):
        logger.info(f"=== INICIANDO DIAGNÓSTICO (TARGET ORDINAL): Dataset {self.method} ===")
        
        if self.df.empty:
            logger.warning("Dataset vacío o sin target. Saltando diagnóstico.")
            return
        
        features = [c for c in self.df.columns 
                     if c != self.target and c not in EXCLUDE_FROM_DIAGNOSTIC]
        if not features:
            logger.warning("No hay suficientes variables para el análisis.")
            return

        # 0. Análisis de Correlación de Pearson (Redundancia entre features)
        pearson_alerts = self.analyze_pearson_redundancy(features)

        # 1. Análisis de Multicolinealidad (VIF)
        try:
            X = add_constant(self.df[features])
            vif_series = pd.Series([variance_inflation_factor(X.values, i) 
                                   for i in range(X.shape[1])], index=X.columns)
        except Exception as e:
            logger.error(f"Error en cálculo de VIF: {e}")
            vif_series = pd.Series()

        # 2. Relevancia No Lineal (Random Forest Classifier)
        try:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(self.df[features], self.df[self.target].astype(int))
            importances = dict(zip(features, model.feature_importances_))
        except Exception as e:
            logger.error(f"Error en Random Forest Classifier: {e}")
            importances = {}

        # 3. Separabilidad por Clase (Kruskal-Wallis)
        separability = self.analyze_class_separability(features)

        # 4. Evaluación de Variables
        dataset_recoms = {}
        for col in features:
            rho, _ = spearmanr(self.df[col], self.df[self.target])
            vif_val = vif_series.get(col, 0)
            rf_val = importances.get(col, 0)
            
            recoms = []
            
            # Criterio 1: Redundancia
            if vif_val > 10:
                recoms.append(f"VIF ALTO ({vif_val:.1f}): Variable redundante.")
            
            # Criterio 2: Causalidad Temporal (Granger — orientativo para target ordinal)
            try:
                g_test = grangercausalitytests(
                    self.df[[self.target, col]], maxlag=self.max_lag, verbose=False
                )
                min_p = min([g_test[i][0]['ssr_chi2test'][1] for i in range(1, self.max_lag + 1)])
                if min_p < 0.05:
                    recoms.append(f"GRANGER (p={min_p:.4f}): Precedencia temporal validada.")
            except:
                pass

            # Criterio 3: No Linealidad
            if abs(rho) < 0.2 and rf_val > (1.5 / len(features)):
                recoms.append(f"NO LINEAL: Baja correlación lineal pero alta importancia en RF ({rf_val:.3f}).")
            
            # Criterio 4: Baja separabilidad entre clases
            if col in separability and separability[col]['p_value'] > 0.05:
                recoms.append(f"SIN PODER DISCRIMINATIVO (KW p={separability[col]['p_value']:.4f}): "
                              f"No distingue entre niveles de riesgo.")

            if recoms:
                dataset_recoms[col] = recoms
                logger.info(f"Diagnóstico [{col}]: " + " | ".join(recoms))

        # --- RESUMEN FINAL COMPILADO ---
        logger.info(f"=== RECOMENDACIONES COMPILADAS: Dataset {self.method} ===")
        
        # Distribución del target
        risk_dist = self.df[self.target].value_counts().sort_index()
        logger.info(f"--- Distribución del Target (nivel_riesgo) ---")
        for level, count in risk_dist.items():
            pct = count / len(self.df) * 100
            logger.info(f"  Nivel {int(level)}: {count} registros ({pct:.1f}%)")
        
        if not pearson_alerts and not dataset_recoms:
            logger.info("Resultado: Dataset óptimo. No se detectaron redundancias ni problemas significativos.")
        else:
            if pearson_alerts:
                logger.info("--- Resumen de Redundancia Lineal (Pearson) ---")
                for alert in pearson_alerts:
                    logger.info(f" > {alert}")
            
            if dataset_recoms:
                logger.info("--- Resumen de Diagnóstico por Atributo (VIF/Granger/RF/KW) ---")
                for col, recs in dataset_recoms.items():
                    logger.info(f" > [{col}]: " + " | ".join(recs))
        
        # Top 10 features por importancia RF
        if importances:
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info("--- Top 10 Features por Importancia (Random Forest Classifier) ---")
            for fname, fval in top_features:
                logger.info(f"  {fname}: {fval:.4f}")
        
        logger.info(f"=== FIN DEL RESUMEN: Dataset {self.method} ===\n")


# =============================================================================
# Funciones de Transformación
# =============================================================================

def apply_setup_preprocessing(df):
    """
    Aplica transformaciones manuales definidas en setup_pre_s4.JSON.
    
    IMPORTANTE para el target de riesgo:
    - 'casosconfirmados' se elimina aquí (definido en drop_columns del JSON)
      para prevenir data leakage — es el insumo directo del canal endémico.
    - Los umbrales (umbral_p25/p50/p75) también se eliminan.
    - 'nivel_riesgo' NUNCA se toca en esta función.
    """
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

    # 3. Eliminar columnas (protegiendo nivel_riesgo)
    drop_cols = setup.get('drop_columns', [])
    # Seguro: nunca eliminar el target
    drop_cols = [c for c in drop_cols if c != 'nivel_riesgo']
    existing_drops = [c for c in drop_cols if c in df_clean.columns]
    if existing_drops:
        df_clean = df_clean.drop(columns=existing_drops)
        logger.info(f"Columnas eliminadas por setup: {existing_drops}")

    # 4. Aplicar PCA a grupos de variables
    for pca_cfg in setup.get('pca_groups', []):
        cols = pca_cfg.get('columns', [])
        n_comp = pca_cfg.get('n_components', 1)
        prefix = pca_cfg.get('prefix', 'pca')
        drop_orig = pca_cfg.get('drop_originals', False)

        available_cols = [c for c in cols if c in df_clean.columns]
        if len(available_cols) >= n_comp and n_comp > 0:
            logger.info(f"Aplicando PCA: {available_cols} -> {n_comp} componentes (Prefijo: {prefix})")
            
            pca_data = df_clean[available_cols].fillna(df_clean[available_cols].median())
            scaler = StandardScaler()
            pca_input_scaled = scaler.fit_transform(pca_data)
            
            pca = PCA(n_components=n_comp)
            pca_results = pca.fit_transform(pca_input_scaled)
            
            for i in range(n_comp):
                df_clean[f"{prefix}_{i+1}"] = pca_results[:, i]
            
            if drop_orig:
                df_clean = df_clean.drop(columns=available_cols)
                logger.info(f"Variables originales eliminadas tras PCA: {available_cols}")
        elif cols:
            logger.warning(f"No se pudo aplicar PCA a {cols}. Columnas disponibles: {len(available_cols)}, n_components: {n_comp}")

    return df_clean


def validate_leakage_prevention(df):
    """
    Verifica que no queden columnas de leakage en el dataset final.
    Si quedan, las elimina y registra una advertencia.
    """
    remaining_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    if remaining_leakage:
        logger.warning(f"[LEAKAGE] Columnas de fuga detectadas y eliminadas: {remaining_leakage}. "
                       f"Verifique setup_pre_s4.JSON para incluirlas en drop_columns.")
        df = df.drop(columns=remaining_leakage)
    return df


def main():
    parser = argparse.ArgumentParser(description="S3.5: Feature Engineering y Diagnóstico para Riesgo de Dengue")
    parser.add_argument('--moving_avg', type=int, default=0, 
                        help="Ventana de media móvil de casos (como feature adicional)")
    args = parser.parse_args()

    input_path = Path("data/processed")
    files = list(input_path.glob("dengue_imputed_M*.csv"))
    
    # Filtrar archivos que ya son modified para evitar doble procesamiento
    files = [f for f in files if "_modified_" not in f.name]

    if not files:
        logger.error("No se encontraron datasets base para procesar.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== INICIO S3.5: Ingeniería de Variables y Diagnóstico (TARGET: nivel_riesgo) ===")
    logger.info(f"Configuración: Moving Avg={args.moving_avg}")

    for f_path in files:
        method = f_path.stem.split('_')[-1]
        logger.info(f"Procesando dataset: {f_path.name}")
        
        try:
            df = pd.read_csv(f_path)
            
            # Verificar que nivel_riesgo existe (generado por S2)
            if 'nivel_riesgo' not in df.columns:
                logger.error(f"'{f_path.name}' no contiene 'nivel_riesgo'. "
                             f"Ejecute S2 actualizado primero.")
                continue
            
            # 1. Aplicar transformaciones de setup_pre_s4.JSON
            #    Esto elimina casosconfirmados, umbrales, y aplica PCA
            df = apply_setup_preprocessing(df)
            
            # 2. Media Móvil de casos como feature adicional (opcional)
            #    NOTA: Los lags de casos y riesgo ya vienen de S2.
            #    Aquí solo se agrega la media móvil si se solicita.
            if args.moving_avg > 0 and 'casosconfirmados' in df.columns:
                df = df.sort_values(['latitud', 'longitud', 'year', 'mes'])
                ma_col = f'casos_ma{args.moving_avg}'
                df[ma_col] = df.groupby(['latitud', 'longitud'])['casosconfirmados'].transform(
                    lambda x: x.rolling(window=args.moving_avg, min_periods=1).mean()
                ).shift(1)  # Shift 1 para evitar fuga de información
                df[ma_col] = df.groupby(['latitud', 'longitud'])[ma_col].transform(
                    lambda x: x.fillna(x.median())
                ).fillna(0)
                logger.info(f"Media móvil de casos generada: {ma_col}")
            
            # 3. Validación de leakage (red de seguridad)
            df = validate_leakage_prevention(df)
            
            # 4. Ejecutar Diagnóstico adaptado para target ordinal
            analyst = DengueRiskAnalyst(df, f"{method}_modified", target='nivel_riesgo')
            analyst.run_full_diagnostic()
            
            # --- Mostrar Estructura Final ---
            logger.info(f"--- Estructura Final del Dataset ({method}_modified) ---")
            types_summary = df.dtypes.to_string()
            for line in types_summary.split('\n'):
                logger.info(f" > {line}")
            logger.info(f"Total de columnas: {len(df.columns)}")
            
            # Verificación explícita de que el target está presente
            if 'nivel_riesgo' in df.columns:
                logger.info(f"Target 'nivel_riesgo' presente. Valores únicos: {sorted(df['nivel_riesgo'].dropna().unique())}")
            else:
                logger.error("ERROR CRÍTICO: 'nivel_riesgo' no está en el dataset final.")
            
            # 5. Guardar dataset modificado
            new_filename = f"{f_path.stem}_modified_{timestamp}.csv"
            output_path = input_path / new_filename
            df.to_csv(output_path, index=False)
            logger.info(f"Dataset guardado exitosamente: {new_filename}")

        except Exception as e:
            logger.error(f"Error procesando {f_path.name}: {e}", exc_info=True)

    logger.info("=== S3.5 FINALIZADO EXITOSAMENTE ===")


if __name__ == "__main__":
    main()
