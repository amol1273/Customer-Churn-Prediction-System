import pandas as pd
from pathlib import Path
import shap
import joblib
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
OUTPUT_DIR = BASE_DIR / "explainability" / "outputs"


def load_model_and_data():
    print("Loading model and data...")

    # Load features
    df = pd.read_csv(FEATURES_PATH)
    X = df[["Recency", "Frequency", "Monetary"]]

    # Load model + scaler
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    scaler = saved["scaler"]

    X_scaled = scaler.transform(X)

    return df, X, X_scaled, model


def generate_shap_values(model, X_scaled):
    print("Generating SHAP values with unified Explainer...")
    # This works for tree models, linear models, etc.
    explainer = shap.Explainer(model, X_scaled)
    shap_result = explainer(X_scaled)

    # shap_result is an Explanation object
    shap_values = shap_result.values
    return explainer, shap_values


def save_plots(explainer, shap_values, X):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Global summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
    plt.savefig(OUTPUT_DIR / "shap_summary_plot.png", bbox_inches="tight")

    # Global feature importance (bar)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=X.columns,
                      plot_type="bar", show=False)
    plt.savefig(OUTPUT_DIR / "shap_feature_importance.png", bbox_inches="tight")

    print("SHAP plots saved in:", OUTPUT_DIR)


def explain_single_customer(shap_values, X, df, customer_index=0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customer_id = df.iloc[customer_index]["CustomerID"]
    print(f"Explaining CustomerID: {customer_id}")

    vals = shap_values[customer_index]
    features = X.columns

    plt.figure()
    plt.barh(features, vals)
    plt.xlabel("SHAP value (impact on churn prediction)")
    plt.title(f"Local explanation for Customer {customer_id}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"customer_{customer_id}_bar_explanation.png")

    print(f"Customer-level bar explanation saved for {customer_id}")


if __name__ == "__main__":
    df, X, X_scaled, model = load_model_and_data()
    explainer, shap_values = generate_shap_values(model, X_scaled)
    save_plots(explainer, shap_values, X)
    explain_single_customer(shap_values, X, df, customer_index=0)
