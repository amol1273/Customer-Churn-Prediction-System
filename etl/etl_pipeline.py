import pandas as pd
from pathlib import Path

RAW_FILE_NAME = "online_retail.csv"

#Set base directory as project root
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / RAW_FILE_NAME
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_transactions.csv"

def load_data():
    print(f"Loading data from:{RAW_DATA_PATH}")

    if RAW_DATA_PATH.suffix == ".csv":
        df = pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1")
    elif RAW_DATA_PATH.suffix  in [".xlsx", ".xls"]:
        df = pd.read_excel(RAW_DATA_PATH)
    else:
        raise ValueError("Unsupported file format.Use CSV or Excel.")

    print("Data loaded!")
    print(df.head())
    print(df.info())
    print("Missing values per column:")
    print(df.isnull().sum())    

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting Cleaning...")

    #Drop rows without CustomerID

    if "CustomerID" in df.columns:
        df  = df.dropna(subset=["CustomerID"])
    else:
        print("Warning: 'CustomerID' column not found. Check your dataset columns.")

    #Remove negative or zero quantity if present
    if "Quantity" in df.columns:
        df=df[df["Quantity"]>0]

    #Remove negative or zero unitprice
    if "UnitPrice" in df.columns:
        df = df[df["UnitPrice"]>0]

    #Convert InvoiceDate to datetime
    date_col_candidates =  ["InvoiceDate", "Invoice Date", "invoice_date"]
    for col in date_col_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.rename(columns={col: "InvoiceDate"}, inplace=True)
            break
    
    # Create TotalAmount
    if all(col in df.columns for col in ["Quantity", "UnitPrice"]):
        df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

    # Remove duplicates
    df = df.drop_duplicates()

    print("Cleaning complete!")
    return df

def save_data(df: pd.DataFrame):
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    save_data(df_clean)