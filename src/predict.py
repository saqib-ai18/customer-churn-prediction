import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/logistic_regression.pkl")

# Selected classification threshold
THRESHOLD = 0.3


def predict_churn(customer_data):
    """
    Predict customer churn using customer information.

    Returns churn probability and risk classification.
    """

    customer_df = pd.DataFrame([customer_data])

    probability = model.predict_proba(customer_df)[0][1]

    prediction = int(probability >= THRESHOLD)

    if probability >= 0.7:
        risk = "High"
    elif probability >= THRESHOLD:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability) * 100, 2),
        "risk_level": risk
    }


# Example customer
if __name__ == "__main__":

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 256.5
    }

    result = predict_churn(customer)

    print("\nCustomer Churn Prediction")
    print("------------------------")

    for key, value in result.items():
        print(f"{key}: {value}")