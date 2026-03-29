from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DATA_PATH = Path("P653_pronostico_dataset.csv")
MODEL_PATH = Path("retinopathy_model.pkl")
FEATURES = ["age", "systolic_bp", "diastolic_bp", "cholesterol"]
TARGET = "prognosis"


def load_dataset(path: Path) -> pd.DataFrame:
    # Dataset uses semicolon delimiters.
    df = pd.read_csv(path, sep=";")

    # Some exports include an ID column we do not need for training.
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df


def train() -> None:
    df = load_dataset(DATA_PATH)

    df[TARGET] = df[TARGET].map({"no_retinopathy": 0, "retinopathy": 1})

    if df[TARGET].isna().any():
        raise ValueError("Unexpected prognosis labels found in dataset.")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    acc = model.score(X_test, y_test)
    print(f"Saved model to {MODEL_PATH} | test_accuracy={acc:.4f}")


if __name__ == "__main__":
    train()
