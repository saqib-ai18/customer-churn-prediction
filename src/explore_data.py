import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/customer_churn.csv")

# Display first five records
print("\nFirst 5 Records:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())