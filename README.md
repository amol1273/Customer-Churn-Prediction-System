#  Customer Transaction Analytics & Churn Prediction System

An end-to-end **customer churn analytics system** built using Python, Machine Learning, SHAP explainability, and Power BI.

This project takes **raw retail transaction data**, processes it through an **ETL pipeline**, engineers **RFM features (Recency, Frequency, Monetary)**, trains a **churn prediction model**, explains it using **SHAP**, and visualizes the insights in a **Power BI dashboard**.

---

##  Features

-  **ETL Pipeline**
  - Cleans raw transaction data (missing customer IDs, negative quantities/prices, duplicates)
  - Creates `TotalAmount` and formats datetime columns

-  **Feature Engineering**
  - Aggregates transaction data at **customer level**
  - Computes **Recency, Frequency, Monetary (RFM)** features
  - Defines **churn** based on 90-day inactivity rule

-  **Machine Learning**
  - Models: Logistic Regression, Random Forest, XGBoost
  - Evaluated using Accuracy, Precision, Recall, F1, ROC-AUC
  - Saves the best-performing model as `best_model.pkl`

-  **Explainable AI (SHAP)**
  - Global feature importance plots
  - Customer-level explanations to understand why a customer is high-risk

-  **Power BI Dashboard**
  - Overview KPIs: total customers, churn rate, average churn probability
  - RFM segmentation visuals
  - High-risk churn customer list
  - Embedded SHAP feature importance chart

---

## Project Structure

```bash
Churn_Prediction_System/
│
├── data/
│   ├── raw/                      # Raw Kaggle transaction data
│   └── processed/                # Cleaned & feature datasets
│
├── etl/
│   └── etl_pipeline.py           # Data cleaning and transformation
│
├── model/
│   ├── feature_engineering.py    # RFM + churn label creation
│   ├── train_model.py            # Model training & evaluation
│   └── predict_scores.py         # Churn probability scoring
│
├── explainability/
│   ├── shap_explainer.py         # SHAP explainability scripts
│   └── outputs/                  # SHAP plots (PNG)
│
├── dashboard/
│   └── Customer_Churn.pbix       # Power BI dashboard (optional)
│
└── README.md
