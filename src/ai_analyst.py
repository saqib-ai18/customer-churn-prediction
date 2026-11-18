import os
import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

BASE_URL = "https://integrate.api.nvidia.com/v1"


def get_client():

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not configured.")

    return OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        timeout=90.0
    )


def generate_churn_insights(df):

    """
    Analyse customer churn statistics using NVIDIA Nemotron.
    """

    required_columns = [
        "Churn",
        "Contract",
        "InternetService",
        "tenure",
        "MonthlyCharges"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    total_customers = len(df)

    churned = df[df["Churn"] == 1]
    retained = df[df["Churn"] == 0]

    churn_rate = len(churned) / total_customers * 100

    # Churn by contract
    contract_churn = (
        df.groupby("Contract")["Churn"]
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )

    # Churn by internet service
    internet_churn = (
        df.groupby("InternetService")["Churn"]
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )

    # Customer tenure
    average_tenure_churned = round(
        churned["tenure"].mean(), 2
    )

    average_tenure_retained = round(
        retained["tenure"].mean(), 2
    )

    # Monthly charges
    average_monthly_charges_churned = round(
        churned["MonthlyCharges"].mean(), 2
    )

    average_monthly_charges_retained = round(
        retained["MonthlyCharges"].mean(), 2
    )

    statistics = {
        "total_customers": total_customers,
        "churned_customers": len(churned),
        "retained_customers": len(retained),
        "churn_rate_percentage": round(churn_rate, 2),
        "contract_churn_percentage": contract_churn,
        "internet_service_churn_percentage": internet_churn,
        "average_tenure_churned": average_tenure_churned,
        "average_tenure_retained": average_tenure_retained,
        "average_monthly_charges_churned":
            average_monthly_charges_churned,
        "average_monthly_charges_retained":
            average_monthly_charges_retained
    }

    prompt = f"""
You are an experienced AI Business Analyst analysing a telecom
customer churn dataset.

Use only the statistics provided below.

DATA:
{json.dumps(statistics, indent=2)}

Create a professional business report with these sections:
1. Executive Summary
2. Important Churn Patterns
3. Customer Segments at Risk
4. Possible Reasons for Customer Churn
5. Customer Retention Strategies
6. Business Recommendations

STRICT RULES:
- Never invent statistics, dates, customer segments, or metrics.
- Do not estimate combined segment churn rates unless provided.
- Do not assume that two individually high-risk categories
  necessarily form a high-risk combined segment.
- Clearly distinguish observed facts from hypotheses.
- Do not invent monetary values or currency symbols.
- Do not claim that a strategy will produce a specific ROI
  without supporting calculations.
- Give practical recommendations based on the available data.
- If additional analysis is needed, explicitly mention it.
"""

    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional customer analytics "
                    "and business intelligence analyst."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=3000,
        stream=False,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    return {
        "statistics": statistics,
        "analysis": response.choices[0].message.content
    }