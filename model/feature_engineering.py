import pandas as pd
from pathlib import Path


# ---------- CONFIG ----------
CHURN_THRESHOLD_DAYS = 90   # if Recency > 90 days → churned
# -----------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_transactions.csv"
FEATURES_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"


def load_clean_data():
    print(f"Loading cleaned data from: {PROCESSED_DATA_PATH}")
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["InvoiceDate"])
    
    # Ensure CustomerID is treated nicely
    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    else:
        raise ValueError("CustomerID column not found in cleaned data!")
    
    print("Data loaded for feature engineering.")
    print(df.head())
    return df


def create_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating RFM features...")

    # Snapshot date = one day after the latest invoice
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"Snapshot date: {snapshot_date}")

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
        "InvoiceNo": "nunique",                                   # Frequency (no. of invoices)
        "TotalAmount": "sum"                                      # Monetary (total spend)
    })

    rfm.rename(columns={
        "InvoiceDate": "Recency",
        "InvoiceNo": "Frequency",
        "TotalAmount": "Monetary"
    }, inplace=True)

    print("RFM features created.")
    print(rfm.head())
    return rfm


def add_churn_label(rfm: pd.DataFrame) -> pd.DataFrame:
    print(f"Adding churn label using threshold: {CHURN_THRESHOLD_DAYS} days...")

    # Churn rule: if Recency > threshold → churned
    rfm["Churn"] = (rfm["Recency"] > CHURN_THRESHOLD_DAYS).astype(int)

    churn_rate = rfm["Churn"].mean()
    print(f"Overall churn rate: {churn_rate:.2%}")

    return rfm


def save_features(df_features: pd.DataFrame):
    FEATURES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(FEATURES_OUTPUT_PATH, index=True)  # CustomerID stays as index
    print(f"Customer-level features saved to: {FEATURES_OUTPUT_PATH}")


if __name__ == "__main__":
    df_clean = load_clean_data()
    rfm = create_rfm_features(df_clean)
    rfm_with_churn = add_churn_label(rfm)
    save_features(rfm_with_churn)
