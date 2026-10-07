"""Definición del preprocesamiento de las variables."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def get_feature_types(X):
    """Devuelve las listas de columnas numéricas y categóricas."""
    cat_cols = [c for c in config.CAT_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols and c != config.TARGET]
    return num_cols, cat_cols


def build_preprocessor(X):
    """Crea el ColumnTransformer: escala las numéricas y codifica las categóricas.

    No se ajusta aquí: el ajuste (fit) lo hace el Pipeline al entrenar,
    para que solo aprenda de los datos de entrenamiento.
    """
    num_cols, cat_cols = get_feature_types(X)

    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


if __name__ == "__main__":
    from src.data.load_data import load_splits

    X_train, X_test, y_train, y_test = load_splits()
    num_cols, cat_cols = get_feature_types(X_train)
    print("Numéricas:", num_cols)
    print("Categóricas:", cat_cols)

    preprocessor = build_preprocessor(X_train)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    print("Forma original:", X_train.shape, "→ transformada:", X_train_prep.shape)
    print("Columnas generadas:", list(preprocessor.get_feature_names_out()))