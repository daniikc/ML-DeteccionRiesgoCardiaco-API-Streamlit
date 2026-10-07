"""Carga del modelo entrenado y predicción sobre datos nuevos."""
import joblib
import pandas as pd

from src import config

# Umbrales para traducir la probabilidad a un nivel de riesgo
LOW_RISK = 0.3
HIGH_RISK = 0.6


def load_model():
    """Carga el pipeline entrenado desde models/."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo en {config.MODEL_PATH}. "
            "Entrénalo primero con: python -m src.models.train"
        )
    return joblib.load(config.MODEL_PATH)


def check_columns(model, df):
    """Comprueba que el DataFrame tiene las columnas con las que se entrenó el modelo."""
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    return df[expected]


def risk_level(proba):
    """Traduce una probabilidad a un nivel de riesgo legible."""
    if proba < LOW_RISK:
        return "Bajo"
    if proba < HIGH_RISK:
        return "Moderado"
    return "Alto"


def predict_one(model, patient):
    """Predice el riesgo de un único paciente.

    patient: diccionario {nombre_columna: valor}.
    """
    df = check_columns(model, pd.DataFrame([patient]))
    proba = float(model.predict_proba(df)[0, 1])
    return {
        "probabilidad": proba,
        "prediccion": int(proba >= 0.5),
        "riesgo": risk_level(proba),
    }


def predict_batch(model, df):
    """Predice el riesgo de varios pacientes. Devuelve el DataFrame con columnas añadidas."""
    X = check_columns(model, df)
    result = df.copy()
    result["probabilidad"] = model.predict_proba(X)[:, 1]
    result["prediccion"] = (result["probabilidad"] >= 0.5).astype(int)
    result["riesgo"] = result["probabilidad"].apply(risk_level)
    return result


if __name__ == "__main__":
    from src.data.load_data import load_splits

    model = load_model()
    _, X_test, _, y_test = load_splits()

    # Un paciente: la primera fila del test, convertida en diccionario
    patient = X_test.iloc[0].to_dict()
    print("Paciente:", patient)
    print("Predicción:", predict_one(model, patient))
    print("Valor real:", y_test.iloc[0])

    # Varios pacientes
    print("\nPrimeros 5 pacientes del test:")
    print(predict_batch(model, X_test.head()))