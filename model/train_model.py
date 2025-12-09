import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "model" / "best_model.pkl"

def load_data():
    print(f"Loading features from: {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)

    X = df[["Recency", "Frequency", "Monetary"]]
    y = df["Churn"]

    return X, y

def train_models(X, y):
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss"
        )
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs),
            "model": model,
            "scaler": scaler
        }

        print(f"{name} Results:")
        print(results[name])

    return results

def save_best_model(results):
    best_model = max(results, key=lambda m: results[m]["roc_auc"])
    print(f"\nBest Model: {best_model} with ROC-AUC {results[best_model]['roc_auc']:.4f}")

    joblib.dump({
        "model": results[best_model]["model"],
        "scaler": results[best_model]["scaler"]
    }, MODEL_OUTPUT_PATH)

    print(f"Best model saved to: {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    X, y = load_data()
    results = train_models(X, y)
    save_best_model(results)
