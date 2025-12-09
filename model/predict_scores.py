import pandas as pd
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_scored.csv"

def load_data_and_model():
    print("Loading features and model...")
    df = pd.read_csv(FEATURES_PATH)
    X = df[["Recency", "Frequency", "Monetary"]]

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    scaler = saved["scaler"]

    X_scaled = scaler.transform(X)

    return df, X_scaled, model

def score_customers():
    df, X_scaled, model = load_data_and_model()

    print("Predicting churn probabilities...")
    churn_probs = model.predict_proba(X_scaled)[:, 1]

    df["Churn_Prob"] = churn_probs

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Scored data saved to: {OUTPUT_PATH}")
    print(df.head())

if __name__ == "__main__":
    score_customers()
