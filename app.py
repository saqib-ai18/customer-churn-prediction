import streamlit as st
import pandas as pd

from src.predict import predict_churn


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/customer_churn_clean.csv")


df = load_data()


# Dashboard title
st.title("📊 Customer Churn Prediction Dashboard")
st.caption("Telecommunications Customer Analytics")


# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard Overview",
    "Churn Analysis",
    "Customer Prediction",
    "Model Performance"
])


# TAB 1: Dashboard Overview
with tab1:

    total_customers = len(df)
    churned_customers = int(df["Churn"].sum())
    retained_customers = total_customers - churned_customers
    churn_rate = churned_customers / total_customers * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Retained Customers", f"{retained_customers:,}")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    st.subheader("Customer Dataset")

    st.dataframe(df.head(10), use_container_width=True)


# TAB 2: Churn Analysis
with tab2:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")

        churn_data = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Customers": [retained_customers, churned_customers]
        })

        st.bar_chart(
            churn_data,
            x="Status",
            y="Customers"
        )

    with col2:
        st.subheader("Churn by Contract")

        contract_data = (
            df.groupby("Contract")["Churn"]
            .mean()
            .mul(100)
            .reset_index()
        )

        contract_data.columns = ["Contract", "Churn Rate (%)"]

        st.bar_chart(
            contract_data,
            x="Contract",
            y="Churn Rate (%)"
        )

    st.subheader("Churn by Internet Service")

    internet_data = (
        df.groupby("InternetService")["Churn"]
        .mean()
        .mul(100)
        .reset_index()
    )

    internet_data.columns = ["Internet Service", "Churn Rate (%)"]

    st.bar_chart(
        internet_data,
        x="Internet Service",
        y="Churn Rate (%)"
    )


# TAB 3: Customer Prediction
with tab3:

    st.subheader("Predict Customer Churn")

    st.write("Enter customer information below.")

    with st.form("customer_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox("Gender", ["Male", "Female"])

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12
            )

            phone = st.selectbox("Phone Service", ["Yes", "No"])

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True
        )
        if submitted:

                customer = {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges
                }

                try:
                    result = predict_churn(customer)

                    st.divider()
                    st.subheader("Prediction Results")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Churn Prediction",
                        result["churn_prediction"]
                    )

                    col2.metric(
                        "Churn Probability",
                        f'{result["churn_probability"]}%'
                    )

                    col3.metric(
                        "Risk Level",
                        result["risk_level"]
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# TAB 4: Model Performance
with tab4:

    st.subheader("Machine Learning Model Comparison")

    performance = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Logistic Regression": [
            80.55, 65.72, 55.88, 60.40, 84.21
        ],
        "XGBoost": [
            79.56, 64.24, 51.87, 57.40, 84.08
        ]
    })

    st.dataframe(
        performance,
        hide_index=True,
        width="stretch"
    )

    st.divider()

    st.subheader("Performance Visualization")

    chart_data = performance.set_index("Metric")

    st.bar_chart(
        chart_data,
        width="stretch"
    )

    st.divider()

    st.subheader("Best Performing Model")

    st.success(
        "Logistic Regression achieved the best overall "
        "results on our current test dataset."
    )

    st.info(
        "Logistic Regression ROC-AUC: 84.21% | "
        "XGBoost ROC-AUC: 84.08%"
    )

    st.caption(
        "Metrics shown use the original 0.5 classification "
        "threshold and the same stratified test dataset."
    )
