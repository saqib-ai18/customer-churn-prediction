import pandas as pd
from pathlib import Path

# Load original dataset
df = pd.read_csv("data/raw/customer_churn.csv")

print("Original dataset shape:", df.shape)

# 1. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 2. Investigate missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 3. Handle missing TotalCharges
# These customers have zero tenure
df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isnull()),
    "TotalCharges"
] = 0

# Verify remaining missing values
if df.isnull().sum().any():
    raise ValueError("Unexpected missing values remain!")

# 4. Convert target variable into binary format
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# 5. Check target variable
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# 6. Check data types
print("\nUpdated data types:")
print(df.dtypes)

# 7. Save cleaned dataset
output_path = Path("data/processed")
output_path.mkdir(parents=True, exist_ok=True)

df.to_csv(
    output_path / "customer_churn_clean.csv",
    index=False
)

print("\nCleaning completed successfully!")
print("Final dataset shape:", df.shape)