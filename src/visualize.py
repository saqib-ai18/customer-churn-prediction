import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load dataset
df = pd.read_csv("data/raw/customer_churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Create output directory
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Set visualization style
sns.set_theme(style="whitegrid")


# 1. Overall Churn Distribution
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.savefig(output_dir / "churn_distribution.png",
            bbox_inches="tight")
plt.show()


# 2. Churn by Contract Type
plt.figure(figsize=(9, 5))

sns.countplot(data=df, x="Contract", hue="Churn")

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.savefig(output_dir / "churn_by_contract.png",
            bbox_inches="tight")
plt.show()


# 3. Churn by Internet Service
plt.figure(figsize=(9, 5))

sns.countplot(data=df, x="InternetService", hue="Churn")

plt.title("Customer Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.savefig(output_dir / "churn_by_internet.png",
            bbox_inches="tight")
plt.show()


# 4. Customer Tenure Distribution
plt.figure(figsize=(9, 5))

sns.histplot(data=df, x="tenure", hue="Churn",
             bins=30, kde=True)

plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.savefig(output_dir / "tenure_distribution.png",
            bbox_inches="tight")
plt.show()


# 5. Monthly Charges vs Churn
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")

plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")

plt.savefig(output_dir / "monthly_charges.png",
            bbox_inches="tight")
plt.show()


print("All visualizations generated successfully!")