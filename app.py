import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.predict import predict_churn
from src.ai_analyst import generate_churn_insights


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme
BG = "#0B1020"
PANEL = "#111A2E"
PANEL_ALT = "#151F35"
BORDER = "rgba(148, 163, 184, 0.15)"
TEXT = "#E8EEF9"
MUTED = "#91A0B8"
BLUE = "#65A8FF"
CYAN = "#48D7D0"
RED = "#FF7185"
GREEN = "#48D6A2"
AMBER = "#F4BC62"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {{
        --bg: {BG}; --panel: {PANEL}; --panel-alt: {PANEL_ALT};
        --border: {BORDER}; --text: {TEXT}; --muted: {MUTED};
        --blue: {BLUE}; --cyan: {CYAN}; --red: {RED};
        --green: {GREEN}; --amber: {AMBER};
    }}
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }}
    .stApp {{ background: radial-gradient(ellipse at 15% 0%, #152441 0%, {BG} 42%); }}
    [data-testid="stHeader"] {{ background: rgba(11,16,32,0.88); }}
    [data-testid="stSidebar"] {{
        background: #0D1527;
        border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.4rem; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}
    h1, h2, h3 {{ font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }}
    h1 {{ font-size: 2rem !important; font-weight: 800 !important; }}
    h2 {{ font-size: 1.35rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 1.05rem !important; font-weight: 700 !important; }}
    p, label, span {{ color: var(--text); }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, {PANEL_ALT}, {PANEL});
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 112px;
    }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .82rem !important; }}
    [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-family: 'Manrope', sans-serif; }}
    [data-testid="stMetricDelta"] {{ font-size: .78rem; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(17,26,46,.78);
        border: 1px solid var(--border);
        border-radius: 16px;
    }}
    .hero {{
        padding: 25px 28px;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: linear-gradient(115deg, rgba(38,75,125,.38), rgba(17,26,46,.9) 58%, rgba(24,49,68,.7));
        margin-bottom: 22px;
    }}
    .eyebrow {{
        color: {CYAN}; text-transform: uppercase; letter-spacing: .16em;
        font-size: .68rem; font-weight: 700; margin-bottom: 8px;
    }}
    .hero-title {{
        color: {TEXT}; font-family: 'Manrope', sans-serif;
        font-size: 1.85rem; font-weight: 800; letter-spacing: -.045em;
        margin: 0 0 6px 0;
    }}
    .hero-copy {{ color: {MUTED}; font-size: .92rem; margin: 0; }}
    .brand {{
        font-family: 'Manrope', sans-serif; font-size: 1.35rem;
        font-weight: 800; letter-spacing: -.04em; color: {TEXT};
    }}
    .brand-mark {{
        display:inline-flex; width:34px; height:34px; align-items:center;
        justify-content:center; border-radius:11px; margin-right:9px;
        color:#07111F; font-weight:900;
        background:linear-gradient(135deg, {CYAN}, {BLUE});
    }}
    .side-caption {{ color:{MUTED}; font-size:.72rem; margin:4px 0 22px 44px; }}
    .section-kicker {{
        color:{MUTED}; font-size:.72rem; text-transform:uppercase;
        letter-spacing:.13em; font-weight:700; margin: 4px 0 12px 0;
    }}
    .insight {{
        padding: 15px 17px; border-radius: 14px;
        background: {PANEL}; border:1px solid {BORDER};
        margin-bottom:10px;
    }}
    .insight-title {{ font-weight:700; color:{TEXT}; margin-bottom:5px; }}
    .insight-copy {{ color:{MUTED}; font-size:.86rem; line-height:1.5; }}
    .pill {{
        display:inline-block; padding:5px 10px; border-radius:999px;
        font-size:.72rem; font-weight:700; background:rgba(72,215,208,.12);
        color:{CYAN}; border:1px solid rgba(72,215,208,.2);
    }}
    div.stButton > button, div.stFormSubmitButton > button {{
        border-radius: 10px; border: 1px solid rgba(101,168,255,.35);
        background: linear-gradient(110deg, #2877D5, #2362B2);
        color:white; font-weight:700; min-height:44px;
    }}
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
        border-color:{CYAN}; color:white; filter:brightness(1.08);
    }}
    [data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:12px; }}
    [data-baseweb="select"] > div, [data-baseweb="input"] > div {{
        background:{PANEL_ALT}; border-color:{BORDER};
    }}
    hr {{ border-color: {BORDER}; }}
    footer {{ visibility:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Data and chart helpers
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("data/processed/customer_churn_clean.csv")
    if "Churn" not in data.columns:
        raise ValueError("The cleaned dataset must contain a 'Churn' column.")
    data["Churn"] = pd.to_numeric(data["Churn"], errors="coerce").fillna(0).astype(int)
    return data


def plot_layout(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=TEXT, size=12),
        title_font=dict(family="Manrope", size=15, color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,.10)", zeroline=False, tickfont=dict(color=MUTED)),
    )
    return fig


def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


try:
    df = load_data()
except Exception as exc:
    st.error(f"Unable to load the cleaned dataset: {exc}")
    st.stop()

required_prediction_columns = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
missing_prediction_columns = [c for c in required_prediction_columns if c not in df.columns]
if missing_prediction_columns:
    st.error("Dataset is missing columns required by the prediction model: " + ", ".join(missing_prediction_columns))
    st.stop()

total_customers = len(df)
churned_customers = int(df["Churn"].sum())
retained_customers = total_customers - churned_customers
churn_rate = (churned_customers / total_customers * 100) if total_customers else 0.0


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div><span class="brand-mark">C</span><span class="brand">CHURNIQ</span></div>'
        '<div class="side-caption">CUSTOMER INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-kicker">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Churn Analytics",
            "Customer Explorer",
            "Churn Prediction",
            "Model Performance",
            "AI Business Analyst",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="section-kicker">DATASET STATUS</div>', unsafe_allow_html=True)
    st.markdown('<span class="pill">● Connected</span>', unsafe_allow_html=True)
    st.caption(f"{total_customers:,} customer records")
    st.caption(f"{len(df.columns)} dataset fields")
    st.caption("Source: Telco Customer Churn")
    st.divider()
    st.caption("CHURNIQ  ·  Customer intelligence workspace")


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">CUSTOMER INTELLIGENCE / OVERVIEW</div>
            <div class="hero-title">Know your customers. Retain your growth.</div>
            <p class="hero-copy">A clear view of customer retention, churn exposure and business performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{total_customers:,}", help="Total records in the cleaned dataset.")
    c2.metric("Customers churned", f"{churned_customers:,}", help="Customers whose Churn value is 1.")
    c3.metric("Customers retained", f"{retained_customers:,}")
    c4.metric("Overall churn rate", f"{churn_rate:.2f}%", help="Churned customers divided by total customers.")

    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        with st.container(border=True):
            section_title("Customer retention mix", "Share of the customer base by current churn status.")
            mix = pd.DataFrame({"Status": ["Retained", "Churned"], "Customers": [retained_customers, churned_customers]})
            fig = px.pie(
                mix, names="Status", values="Customers", hole=.72,
                color="Status", color_discrete_map={"Retained": GREEN, "Churned": RED},
            )
            fig.update_traces(textinfo="percent", textfont_size=13, marker=dict(line=dict(color=PANEL, width=3)))
            fig.add_annotation(text=f"<b>{churn_rate:.1f}%</b><br><span style='font-size:11px'>CHURN</span>",
                               x=.5, y=.5, showarrow=False, font=dict(size=22, color=TEXT))
            plot_layout(fig, 315)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            section_title("Churn by contract", "Observed churn rate within each contract group.")
            contract = df.groupby("Contract", dropna=False)["Churn"].mean().mul(100).reset_index(name="Churn rate")
            contract = contract.sort_values("Churn rate", ascending=True)
            fig = px.bar(contract, x="Churn rate", y="Contract", orientation="h", text=contract["Churn rate"].map(lambda x: f"{x:.1f}%"))
            fig.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
            fig.update_xaxes(range=[0, max(50, contract["Churn rate"].max() * 1.2)], ticksuffix="%")
            plot_layout(fig, 315)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    lower_left, lower_right = st.columns([1.1, 1])
    with lower_left:
        with st.container(border=True):
            section_title("Churn by internet service", "Compare churn rates across service categories.")
            internet = df.groupby("InternetService", dropna=False)["Churn"].mean().mul(100).reset_index(name="Churn rate")
            fig = px.bar(internet, x="InternetService", y="Churn rate", text=internet["Churn rate"].map(lambda x: f"{x:.1f}%"))
            fig.update_traces(marker_color=CYAN, textposition="outside", cliponaxis=False)
            fig.update_yaxes(range=[0, max(50, internet["Churn rate"].max() * 1.2)], ticksuffix="%")
            plot_layout(fig, 300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with lower_right:
        with st.container(border=True):
            section_title("What stands out", "Signals visible in the current dataset.")
            if "Contract" in df.columns:
                highest_contract = contract.sort_values("Churn rate", ascending=False).iloc[0]
                st.markdown(
                    f'<div class="insight"><div class="insight-title">Highest contract churn</div>'
                    f'<div class="insight-copy">{highest_contract["Contract"]}: '
                    f'<b style="color:{RED}">{highest_contract["Churn rate"]:.2f}%</b> observed churn.</div></div>',
                    unsafe_allow_html=True,
                )
            if "InternetService" in df.columns:
                highest_internet = internet.sort_values("Churn rate", ascending=False).iloc[0]
                st.markdown(
                    f'<div class="insight"><div class="insight-title">Highest internet-service churn</div>'
                    f'<div class="insight-copy">{highest_internet["InternetService"]}: '
                    f'<b style="color:{AMBER}">{highest_internet["Churn rate"]:.2f}%</b> observed churn.</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="insight"><div class="insight-title">Interpretation note</div>'
                '<div class="insight-copy">These are group-level associations, not proof of why individual customers leave.</div></div>',
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        section_title("Customer records", "A small preview of the cleaned dataset.")
        st.dataframe(df.head(8), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Churn analytics
# ---------------------------------------------------------------------
elif page == "Churn Analytics":
    st.markdown('<div class="eyebrow">EXPLORE THE DRIVERS</div>', unsafe_allow_html=True)
    section_title("Churn analytics", "Compare customer groups and explore where churn is concentrated.")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        contract_filter = st.multiselect("Contract type", sorted(df["Contract"].dropna().unique()), default=sorted(df["Contract"].dropna().unique()))
    with filter_col2:
        internet_filter = st.multiselect("Internet service", sorted(df["InternetService"].dropna().unique()), default=sorted(df["InternetService"].dropna().unique()))
    with filter_col3:
        tenure_range = st.slider("Tenure range (months)", 0, int(df["tenure"].max()), (0, int(df["tenure"].max())))

    filtered = df[
        df["Contract"].isin(contract_filter)
        & df["InternetService"].isin(internet_filter)
        & df["tenure"].between(tenure_range[0], tenure_range[1])
    ]
    f1, f2, f3 = st.columns(3)
    f1.metric("Filtered customers", f"{len(filtered):,}")
    f2.metric("Filtered churn rate", f"{filtered['Churn'].mean() * 100:.2f}%" if len(filtered) else "—")
    f3.metric("Churned in selection", f"{int(filtered['Churn'].sum()):,}")

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            section_title("Churn rate by contract")
            data = filtered.groupby("Contract")["Churn"].mean().mul(100).reset_index(name="Churn rate")
            fig = px.bar(data, x="Contract", y="Churn rate", color="Contract", text=data["Churn rate"].map(lambda x: f"{x:.1f}%"),
                         color_discrete_sequence=[BLUE, CYAN, AMBER])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_yaxes(ticksuffix="%")
            plot_layout(fig, 330)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with b:
        with st.container(border=True):
            section_title("Churn rate by internet service")
            data = filtered.groupby("InternetService")["Churn"].mean().mul(100).reset_index(name="Churn rate")
            fig = px.bar(data, x="InternetService", y="Churn rate", color="InternetService", text=data["Churn rate"].map(lambda x: f"{x:.1f}%"),
                         color_discrete_sequence=[CYAN, BLUE, AMBER])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_yaxes(ticksuffix="%")
            plot_layout(fig, 330)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            section_title("Tenure profile", "Customer count by tenure month and churn status.")
            tenure_data = filtered.copy()
            tenure_data["Churn status"] = tenure_data["Churn"].map({0: "Retained", 1: "Churned"})
            fig = px.histogram(tenure_data, x="tenure", color="Churn status", nbins=24,
                               barmode="overlay", opacity=.78,
                               color_discrete_map={"Retained": GREEN, "Churned": RED})
            plot_layout(fig, 330)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with b:
        with st.container(border=True):
            section_title("Monthly charges", "Distribution by churn status.")
            charge_data = filtered.copy()
            charge_data["Churn status"] = charge_data["Churn"].map({0: "Retained", 1: "Churned"})
            fig = px.box(charge_data, x="Churn status", y="MonthlyCharges", color="Churn status",
                         color_discrete_map={"Retained": GREEN, "Churned": RED}, points=False)
            plot_layout(fig, 330)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------
# Customer explorer
# ---------------------------------------------------------------------
elif page == "Customer Explorer":
    st.markdown('<div class="eyebrow">CUSTOMER 360</div>', unsafe_allow_html=True)
    section_title("Customer explorer", "Search, filter and inspect customer records.")

    search = st.text_input("Search customer ID", placeholder="Type a customer ID...")
    x1, x2, x3 = st.columns(3)
    with x1:
        status_filter = st.selectbox("Churn status", ["All", "Churned", "Retained"])
    with x2:
        contract_filter = st.selectbox("Contract", ["All"] + sorted(df["Contract"].dropna().unique().tolist()))
    with x3:
        service_filter = st.selectbox("Internet service", ["All"] + sorted(df["InternetService"].dropna().unique().tolist()))

    view = df.copy()
    if search and "customerID" in view.columns:
        view = view[view["customerID"].astype(str).str.contains(search, case=False, na=False)]
    if status_filter == "Churned":
        view = view[view["Churn"] == 1]
    elif status_filter == "Retained":
        view = view[view["Churn"] == 0]
    if contract_filter != "All":
        view = view[view["Contract"] == contract_filter]
    if service_filter != "All":
        view = view[view["InternetService"] == service_filter]

    st.caption(f"Showing {len(view):,} matching records")
    st.dataframe(view, use_container_width=True, hide_index=True, height=480)

    if not view.empty and "customerID" in view.columns:
        selected_id = st.selectbox("Inspect customer", view["customerID"].astype(str).tolist())
        selected = view[view["customerID"].astype(str) == selected_id].iloc[0]
        with st.container(border=True):
            st.markdown(f"**Customer profile · {selected_id}**")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Tenure", f"{int(selected['tenure'])} months")
            p2.metric("Monthly charges", f"{float(selected['MonthlyCharges']):.2f}")
            p3.metric("Contract", str(selected["Contract"]))
            p4.metric("Status", "Churned" if int(selected["Churn"]) == 1 else "Retained")


# ---------------------------------------------------------------------
# Churn prediction
# ---------------------------------------------------------------------
elif page == "Churn Prediction":
    st.markdown('<div class="eyebrow">PREDICTIVE WORKSPACE</div>', unsafe_allow_html=True)
    section_title("Customer churn prediction", "Enter a customer's profile to estimate churn probability.")
    st.info("The prediction uses the existing Logistic Regression pipeline and its configured 0.30 classification threshold.")

    with st.container(border=True):
        with st.form("customer_form"):
            st.markdown("### Customer profile")
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
                partner = st.selectbox("Has partner", ["Yes", "No"])
                dependents = st.selectbox("Has dependents", ["Yes", "No"])
                tenure = st.number_input("Tenure (months)", 0, 72, 12)
                phone = st.selectbox("Phone service", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            with c2:
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
                online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
                device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            with c3:
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless billing", ["Yes", "No"])
                payment = st.selectbox("Payment method", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=1.0)
                total_charges = st.number_input("Total charges", min_value=0.0, value=840.0, step=10.0)

            submitted = st.form_submit_button("Run churn prediction", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        try:
            result = predict_churn(customer)
            probability = float(result["churn_probability"])
            risk = result["risk_level"]
            st.write("")
            st.markdown("### Prediction result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Prediction", result["churn_prediction"])
            r2.metric("Churn probability", f"{probability:.2f}%")
            r3.metric("Risk classification", risk)
            color = RED if risk == "High" else AMBER if risk == "Medium" else GREEN
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=probability,
                number={"suffix": "%", "font": {"color": TEXT, "size": 30}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": MUTED},
                    "bar": {"color": color},
                    "bgcolor": PANEL_ALT,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(72,214,162,.12)"},
                        {"range": [30, 70], "color": "rgba(244,188,98,.12)"},
                        {"range": [70, 100], "color": "rgba(255,113,133,.12)"},
                    ],
                },
                title={"text": "Estimated churn probability", "font": {"color": MUTED, "size": 13}},
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=45, b=5), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("This is a model estimate, not a guarantee that the customer will churn.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown('<div class="eyebrow">MODEL EVALUATION</div>', unsafe_allow_html=True)
    section_title("Model performance", "Compare both models on the same stratified test split.")

    performance = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Logistic Regression": [80.55, 65.72, 55.88, 60.40, 84.21],
        "XGBoost": [79.56, 64.24, 51.87, 57.40, 84.08],
    })
    best_model = "Logistic Regression"
    k1, k2, k3 = st.columns(3)
    k1.metric("Best ROC-AUC", "84.21%", "Logistic Regression")
    k2.metric("Logistic F1 score", "60.40%")
    k3.metric("XGBoost F1 score", "57.40%")

    left, right = st.columns([1.35, 1])
    with left:
        with st.container(border=True):
            section_title("Metric comparison")
            long = performance.melt(id_vars="Metric", var_name="Model", value_name="Score")
            fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                         color_discrete_map={"Logistic Regression": BLUE, "XGBoost": CYAN}, text_auto=".1f")
            fig.update_yaxes(range=[0, 100], ticksuffix="%")
            plot_layout(fig, 360)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            section_title("Evaluation snapshot")
            st.markdown(
                '<div class="insight"><div class="insight-title">Current leading model</div>'
                '<div class="insight-copy"><b style="color:#65A8FF">Logistic Regression</b> performs slightly better across the displayed metrics.</div></div>'
                '<div class="insight"><div class="insight-title">ROC-AUC</div>'
                '<div class="insight-copy">84.21% for Logistic Regression and 84.08% for XGBoost.</div></div>'
                '<div class="insight"><div class="insight-title">Threshold note</div>'
                '<div class="insight-copy">The table uses the original 0.50 classification threshold. The prediction page uses the configured 0.30 threshold.</div></div>',
                unsafe_allow_html=True,
            )
    with st.container(border=True):
        section_title("Detailed metrics")
        st.dataframe(performance, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# AI business analyst
# ---------------------------------------------------------------------
elif page == "AI Business Analyst":
    st.markdown('<div class="eyebrow">GENERATIVE BUSINESS INTELLIGENCE</div>', unsafe_allow_html=True)
    section_title("AI business analyst", "Turn customer churn statistics into an executive-ready analysis.")
    st.markdown(
        '<div class="insight"><div class="insight-title">NVIDIA Nemotron</div>'
        '<div class="insight-copy">The AI analyst receives aggregated dataset statistics and generates explanations and retention recommendations. '
        'Its narrative should be treated as decision support, not as proof of causation.</div></div>',
        unsafe_allow_html=True,
    )

    if st.button("Generate business analysis", type="primary", use_container_width=False):
        try:
            with st.spinner("Nemotron is analysing the customer statistics..."):
                result = generate_churn_insights(df)
                st.session_state["ai_analysis"] = result
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    if "ai_analysis" in st.session_state:
        result = st.session_state["ai_analysis"]
        stats = result.get("statistics", {})
        st.write("")
        a, b, c = st.columns(3)
        a.metric("Customers analysed", f"{stats.get('total_customers', 0):,}")
        b.metric("Customers churned", f"{stats.get('churned_customers', 0):,}")
        c.metric("Overall churn rate", f"{stats.get('churn_rate_percentage', 0)}%")
        with st.container(border=True):
            st.markdown("### Executive analysis")
            st.markdown(result.get("analysis", "No analysis was returned."))
        st.download_button(
            "Download analysis as Markdown",
            data=result.get("analysis", ""),
            file_name="churn_business_analysis.md",
            mime="text/markdown",
        )
        with st.expander("View source statistics"):
            st.json(stats)
    else:
        st.caption("No report generated in this session yet.")
