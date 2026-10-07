"""Carga y división de los datos del proyecto."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_clean():
    """Lee el dataset limpio generado en el notebook de EDA."""
    return pd.read_csv(config.CLEAN_PATH)


def split_and_save(df):
    """Divide en train/test de forma estratificada y guarda ambos en data/processed."""
    train, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET],
    )

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)

    return train, test


def split_xy(df):
    """Separa un DataFrame en variables (X) y objetivo (y)."""
    X = df.drop(columns=config.TARGET)
    y = df[config.TARGET]
    return X, y


def load_splits():
    """Carga train y test ya divididos, separados en X e y."""
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)

    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_clean()
    train, test = split_and_save(df)
    print(f"Train: {train.shape} | Test: {test.shape}")
    print(f"Proporción de positivos -> train: {train[config.TARGET].mean():.2%} | "
          f"test: {test[config.TARGET].mean():.2%}")