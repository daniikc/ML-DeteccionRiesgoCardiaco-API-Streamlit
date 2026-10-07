

"""Configuración central del proyecto: rutas y parámetros."""
from pathlib import Path

# =====================================================
# RUTAS
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

# Datos
DATA_RAW = ROOT / "data" / "raw" / "Medicaldataset.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
CLEAN_PATH = DATA_PROCESSED / "clean_heart.csv"
TRAIN_PATH = DATA_PROCESSED / "train.csv"
TEST_PATH = DATA_PROCESSED / "test.csv"

# Modelos y resultados
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "xgboost_heart.joblib"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_PATH = ROOT / "reports" / "metrics.json"     # ← añadir
# =====================================================
# DATOS
# =====================================================
TARGET = "target"
CAT_COLS = ["Gender"]

# =====================================================
# REPRODUCIBILIDAD
# =====================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# =====================================================
# MODELO FINAL (resultado del GridSearchCV del notebook 04)
# =====================================================
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
}