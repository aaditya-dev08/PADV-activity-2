# coding: utf-8
"""
src/train_model.py
------------------
Spotify Hit Prediction -- Model Training Script
Trains 3 classifiers, evaluates them, saves the best by F1-score.
Outputs: models/best_model.pkl, models/scaler.pkl,
         outputs/model_comparison.csv, outputs/feature_importance.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

warnings.filterwarnings("ignore")

# Paths
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT, "data",    "spotify_cleaned.csv")
MODEL_DIR   = os.path.join(ROOT, "models")
OUTPUT_DIR  = os.path.join(ROOT, "outputs")
MODEL_PATH  = os.path.join(MODEL_DIR,  "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR,  "scaler.pkl")
COMP_PATH   = os.path.join(OUTPUT_DIR, "model_comparison.csv")
FEAT_PATH   = os.path.join(OUTPUT_DIR, "feature_importance.csv")

# Features & Target
FEATURES = [
    "danceability", "energy", "loudness", "valence",
    "tempo", "acousticness", "instrumentalness",
    "speechiness", "liveness",
]
TARGET = "is_hit"


def load_data() -> pd.DataFrame:
    print("[*] Loading cleaned data from {} ...".format(DATA_PATH))
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Cleaned data not found at {}.\n"
            "Run `python src/prepare_data.py` first.".format(DATA_PATH)
        )
    df = pd.read_csv(DATA_PATH)
    print("   [OK] Shape: {}".format(df.shape))
    return df


def prepare_splits(df: pd.DataFrame):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: {}".format(missing))

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("   Train: {:,} | Test: {:,}".format(X_train_sc.shape[0], X_test_sc.shape[0]))
    print("   Hit rate (train): {:.1f}%  |  (test): {:.1f}%".format(
        y_train.mean() * 100, y_test.mean() * 100))
    return X_train_sc, X_test_sc, y_train, y_test, scaler


def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),    4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        "AUC_ROC":   round(roc_auc_score(y_test, y_proba),    4),
    }


def get_feature_importance(model, model_name: str) -> pd.DataFrame:
    """Extract feature importance for tree-based models or coefficients for LR."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(FEATURES))

    df = pd.DataFrame({
        "Feature":    FEATURES,
        "Importance": importances,
        "Model":      model_name,
    }).sort_values("Importance", ascending=False)
    return df


def train_all(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5,
            learning_rate=0.1, random_state=42
        ),
    }

    trained = {}
    for name, clf in models.items():
        print("\n   [Train] {} ...".format(name))
        clf.fit(X_train, y_train)
        trained[name] = clf
        print("          Done.")
    return trained


def main():
    print("\n" + "=" * 55)
    print("  SPOTIFY HIT PREDICTION -- MODEL TRAINING")
    print("=" * 55 + "\n")

    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load & split
    df = load_data()
    print("\n[*] Preparing train/test splits ...")
    X_train, X_test, y_train, y_test, scaler = prepare_splits(df)

    # 2. Train
    print("\n[*] Training classifiers ...")
    models = train_all(X_train, y_train)

    # 3. Evaluate
    print("\n[*] Evaluating models ...")
    results = []
    for name, model in models.items():
        metrics = evaluate(name, model, X_test, y_test)
        results.append(metrics)
        print("   {:<22s} | Acc {:.3f} | F1 {:.3f} | AUC {:.3f}".format(
            name, metrics["Accuracy"], metrics["F1"], metrics["AUC_ROC"]))

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(COMP_PATH, index=False)
    print("\n[SAVED] Model comparison -> {}".format(COMP_PATH))

    # 4. Pick best by F1
    best_row   = comparison_df.loc[comparison_df["F1"].idxmax()]
    best_name  = best_row["Model"]
    best_model = models[best_name]
    print("\n[BEST] {} (F1={:.4f})".format(best_name, best_row["F1"]))

    # 5. Classification report
    y_pred_best = best_model.predict(X_test)
    print("\n" + classification_report(y_test, y_pred_best, target_names=["Not Hit", "Hit"]))

    # 6. Feature importance
    feat_df = get_feature_importance(best_model, best_name)
    feat_df.to_csv(FEAT_PATH, index=False)
    print("[SAVED] Feature importance -> {}".format(FEAT_PATH))
    print("\n   Top features:")
    for _, row in feat_df.head(5).iterrows():
        bar = "#" * int(row["Importance"] * 40)
        print("   {:<20s} {}  {:.4f}".format(row["Feature"], bar, row["Importance"]))

    # 7. Save artifacts
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    print("\n[SAVED] best_model.pkl  -> {}".format(MODEL_PATH))
    print("[SAVED] scaler.pkl      -> {}".format(SCALER_PATH))

    print("\n" + "=" * 55)
    print("  TRAINING COMPLETE")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
