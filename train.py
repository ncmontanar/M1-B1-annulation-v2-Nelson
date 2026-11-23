"""Entraînement du modèle d'annulation v2 — Escale Atlantique (mission FastIA).

Tâche 9 : reporte ici la démarche finale mise au point dans tes notebooks.

"""

import json
import os
import time
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_MONTHS = 4
TARGET = "is_canceled"

load_dotenv(ROOT / ".env")


def load_data(engine) -> pd.DataFrame:
    """
    TODO 
    """
    raise NotImplementedError




# 1. Fonction pour séparer les derniers mois d'arrivée du reste du reste du jeu de données
def split_by_last_months(df: pd.DataFrame, months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les months derniers mois d'arrivée du reste."""
    cutoff = df["arrival_date"].max() - pd.DateOffset(months=months)
    return df[df["arrival_date"] <= cutoff], df[df["arrival_date"] > cutoff]

# 2. Fonction pour séparer le jeu de données en train et test selon les derniers mois d'arrivée
def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Met de côté les TEST_MONTHS derniers mois d'arrivée comme jeu de test."""
    return split_by_last_months(df, TEST_MONTHS)


def build_pipeline() -> Pipeline:
    """
    TODO
    """
    raise NotImplementedError


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    TODO 
    """
    raise NotImplementedError


def main() -> None:
    engine = create_engine(os.environ["DATABASE_URL"])
    df = load_data(engine)
    train, test = split_train_test(df)
    print(f"Entraînement : {len(train)} réservations | test : {len(test)} réservations")

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    model = build_pipeline()
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = round(time.perf_counter() - start, 1)

    metrics = evaluate(model, X_test, y_test)
    print("Métriques sur le jeu de test :", metrics)

    model_path = MODEL_DIR / "cancellation_model_v2.joblib"
    joblib.dump(model, model_path)
    info = {
        "model_name": "cancellation_model",
        "version": "2.0.0",
        "trained_on": date.today().isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "features": list(X_train.columns),
        "target": TARGET,
        "train_period": [str(train["arrival_date"].min()), str(train["arrival_date"].max())],
        "test_period": [str(test["arrival_date"].min()), str(test["arrival_date"].max())],
        "hyperparameters": {
            k: v for k, v in model.get_params().items()
            if k.startswith("classifier__") and isinstance(v, (int, float, str, bool, type(None)))
        },
        "test_metrics": metrics,
        "training_seconds": training_seconds,
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
    }
    (MODEL_DIR / "model_info_v2.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("Modèle et fiche sauvegardés dans", MODEL_DIR)


if __name__ == "__main__":
    main()
