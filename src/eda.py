import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/raw/customer_churn.csv")

# 1. Basic dataset information
print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)
df.info()

# 2. Check duplicate records
print("\n--- DUPLICATE CHECK ---")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate customer IDs:", df["customerID"].duplicated().sum())

# 3. Investigate TotalCharges
print("\n--- TOTAL CHARGES ANALYSIS ---")

print("Blank values:",
      df["TotalCharges"].str.strip().eq("").sum())

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("Missing values after conversion:",
      df["TotalCharges"].isnull().sum())

# 4. Numerical feature analysis
print("\n--- NUMERICAL FEATURES ---")

print(
    df[["tenure", "MonthlyCharges", "TotalCharges"]].describe()
)

# 5. Churn percentage
print("\n--- CHURN RATE ---")

print(
    df["Churn"].value_counts(normalize=True) * 100
)

# 6. Churn based on contract type
print("\n--- CHURN BY CONTRACT ---")

print(
    df.groupby("Contract", observed=True)["Churn"]
    .value_counts(normalize=True)
    .unstack(fill_value=0) * 100
)

# 7. Churn based on internet service
print("\n--- CHURN BY INTERNET SERVICE ---")

print(
    df.groupby("InternetService", observed=True)["Churn"]
    .value_counts(normalize=True)
    .unstack(fill_value=0) * 100
)