# HEART — Predicción de riesgo cardíaco

Proyecto de Machine Learning que predice la presencia de un evento cardíaco a partir de datos clínicos básicos (constantes vitales, glucosa y marcadores cardíacos). Incluye análisis exploratorio, preprocesamiento, modelado, optimización de hiperparámetros, un pipeline reproducible en `src/` y una app web en Streamlit.


## Dataset

Fichero: `data/raw/Medicaldataset.csv` — 1.319 registros (509 negativos, 810 positivos).

| Variable | Descripción |
|---|---|
| Age | Edad |
| Gender | Sexo (codificado 0/1) |
| Heart rate | Frecuencia cardíaca |
| Systolic blood pressure | Presión arterial sistólica |
| Diastolic blood pressure | Presión arterial diastólica |
| Blood sugar | Glucosa en sangre |
| CK-MB | Enzima CK-MB |
| Troponin | Troponina |
| Result | Etiqueta original (`positive` / `negative`), convertida a `target` (0/1) en el notebook de EDA |

## Estructura del proyecto

```
HEART/
├── app/
│   └── app_streamlit.py        # App web (Streamlit)
├── data/
│   ├── raw/Medicaldataset.csv  # Datos originales
│   └── processed/              # clean_heart.csv, train.csv, test.csv
├── models/
│   └── xgboost_heart.joblib    # Pipeline entrenado (preprocesador + XGBoost)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_PREPROCESAMIENTO.ipynb
│   ├── 03_MODELADO.ipynb
│   └── 04_OPTIMIZACION.ipynb
├── reports/
│   └── metrics.json            # Métricas en test del modelo final
├── src/
│   ├── config.py               # Rutas y parámetros centrales
│   ├── data/load_data.py       # Carga y división train/test
│   ├── features/preprocessing.py  # ColumnTransformer (escalado + one-hot)
│   └── models/
│       ├── train.py            # Entrenamiento y evaluación
│       └── predict.py          # Predicción individual y por lotes
└── requirements.txt
```

## Metodología

1. **EDA** (`01_EDA.ipynb`): exploración y limpieza; genera `data/processed/clean_heart.csv`.
2. **Preprocesamiento** (`02_PREPROCESAMIENTO.ipynb`): división estratificada 80/20 y definición del preprocesador.
3. **Modelado** (`03_MODELADO.ipynb`): comparación de modelos candidatos.
4. **Optimización** (`04_OPTIMIZACION.ipynb`): `GridSearchCV` (5 folds) sobre XGBoost.

**Modelo final:** `Pipeline` de scikit-learn con `StandardScaler` (variables numéricas) + `OneHotEncoder` (`Gender`) + `XGBClassifier` (`n_estimators=100`, `max_depth=3`, `learning_rate=0.05`, `random_state=42`).

## Resultados (conjunto de test)

| Métrica | Valor |
|---|---|
| Accuracy | 0,989 |
| Precision | 0,982 |
| Recall | 1,000 |
| F1 | 0,991 |
| ROC AUC | 0,993 |

Fuente: [reports/metrics.json](reports/metrics.json).

## Instalación

Requiere Python 3.14 (el entorno del proyecto se creó con esta versión).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

Todos los comandos se ejecutan desde la raíz del proyecto.

```powershell
# 1. Dividir los datos en train/test (requiere data/processed/clean_heart.csv)
python -m src.data.load_data

# 2. Entrenar y evaluar el modelo (guarda models/ y reports/metrics.json)
python -m src.models.train

# 3. Probar predicciones de ejemplo
python -m src.models.predict

# 4. Lanzar la app web
streamlit run app/app_streamlit.py
```

### App Streamlit

- **Un paciente:** formulario con los datos clínicos; devuelve la probabilidad y el nivel de riesgo.
- **Varios pacientes (CSV):** sube un CSV con las mismas columnas de entrenamiento y descarga las predicciones.
- **Sobre el modelo:** métricas e hiperparámetros del modelo entrenado.

### Niveles de riesgo

| Probabilidad | Nivel |
|---|---|
| < 0,30 | Bajo |
| 0,30 – 0,60 | Moderado |
| ≥ 0,60 | Alto |

La clase predicha es positiva cuando la probabilidad es ≥ 0,5.

## Configuración

Rutas, semilla (`RANDOM_STATE=42`), tamaño de test (`0.2`) e hiperparámetros del modelo se definen en [src/config.py](src/config.py).

## Tecnologías

Python · pandas · scikit-learn · XGBoost · Streamlit · Jupyter · matplotlib · seaborn
