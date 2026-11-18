import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("data/processed/customer_churn_clean.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

# 2. Recreate the same test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load trained Logistic Regression model
model = joblib.load("models/logistic_regression.pkl")

# 4. Get churn probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Compare thresholds
thresholds = [0.3, 0.4, 0.5, 0.6]

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))