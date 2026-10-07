"""Entrenamiento del modelo final con los hiperparámetros elegidos en el notebook 04.

Uso (desde la raíz del proyecto):
    python -m src.models.train
"""
import json
from datetime import datetime

import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.data.load_data import load_splits
from src.features.preprocessing import build_preprocessor


def build_model(X):
    """Crea el Pipeline completo: preprocesador + XGBoost."""
    return Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", XGBClassifier(
            **config.XGB_PARAMS,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
        )),
    ])


def evaluate(model, X, y):
    """Calcula las métricas principales sobre un conjunto de datos."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
    }


def save_model(model, metrics):
    """Guarda el pipeline entrenado y sus métricas."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "params": config.XGB_PARAMS,
        "test_metrics": metrics,
    }
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    X_train, X_test, y_train, y_test = load_splits()
    print(f"Datos cargados -> train: {X_train.shape} | test: {X_test.shape}")

    model = build_model(X_train)
    model.fit(X_train, y_train)
    print("Modelo entrenado")

    metrics = evaluate(model, X_test, y_test)
    print("\nMétricas en test:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.3f}")
    print("\n" + classification_report(y_test, model.predict(X_test)))

    save_model(model, metrics)
    print(f"Modelo guardado en: {config.MODEL_PATH}")
    print(f"Métricas guardadas en: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()