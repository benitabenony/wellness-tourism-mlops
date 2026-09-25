"""
Streamlit App - Wellness Tourism Package Purchase Predictor
Loads the model committed to the repository and predicts whether a prospective
customer is likely to purchase the Wellness Tourism Package.
"""
import json
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_PATH = "tourism_project/deployment/model.joblib"
METRICS_PATH = "tourism_project/deployment/metrics.json"

st.set_page_config(
    page_title="Wellness Tourism Purchase Predictor",
    page_icon="🧳",
    layout="wide",
)

# ── Custom styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(46, 158, 91, 0.08);
        border: 1px solid rgba(46, 158, 91, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .app-header {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #2E9E5B 0%, #1B6B3E 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .app-header h1 { margin: 0; font-size: 1.7rem; }
    .app-header p { margin: 4px 0 0 0; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


model = load_model()
metrics_info = load_metrics()

# ── Sidebar: model card ──────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        "Predicts whether a prospective customer is likely to purchase "
        "**Visit with Us's** newly launched **Wellness Tourism Package**, "
        "so the sales team can prioritise outreach before making contact."
    )

    if metrics_info:
        st.subheader("Champion Model")
        st.write(f"**Algorithm:** {metrics_info.get('best_model', 'N/A')}")
        m = metrics_info.get("metrics", {})
        col_a, col_b = st.columns(2)
        col_a.metric("F1 Score", f"{m.get('f1', 0):.3f}")
        col_b.metric("ROC-AUC", f"{m.get('roc_auc', 0):.3f}")
        col_a.metric("Precision", f"{m.get('precision', 0):.3f}")
        col_b.metric("Recall", f"{m.get('recall', 0):.3f}")
        st.caption(
            "Selected from 6 tuned candidate algorithms "
            "(Decision Tree, Bagging, Random Forest, AdaBoost, "
            "Gradient Boosting, XGBoost) by held-out test F1 score."
        )

    st.divider()
    st.caption(
        "🔄 Built as part of an end-to-end MLOps pipeline: GitHub Actions handles "
        "data validation, cleaning, model tuning with MLflow tracking, and "
        "redeploys this app automatically whenever the model is refreshed."
    )

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🧳 Wellness Tourism Package — Purchase Predictor</h1>
    <p>Predict whether a prospective customer is likely to buy, before your sales team makes contact.</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_insights = st.tabs(["🔮 Predict", "📊 Model Insights"])

# ── Tab 1: Predict ────────────────────────────────────────────────────
with tab_predict:
    with st.form("customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Customer Profile")
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            st.caption("Typical range in training data: 18–61 years")
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            occupation = st.selectbox(
                "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
            )
            designation = st.selectbox(
                "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
            )
            monthly_income = st.number_input("Monthly Income", 1000, 200000, 22000, step=1000)
            st.caption("Typical range in training data: ₹1,000 – ₹98,000")
            passport = st.selectbox("Holds Passport?", ["Yes", "No"])
            own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

        with col2:
            st.subheader("🏖️ Trip & Pitch Details")
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
            product_pitched = st.selectbox(
                "Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
            )
            preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
            duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 180, 15)
            num_followups = st.number_input("Number of Followups", 0, 20, 3)
            pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
            num_persons_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
            num_children = st.number_input("Number of Children Visiting (<5 yrs)", 0, 5, 0)
            num_trips = st.number_input("Number of Trips (avg/year)", 0, 30, 3)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Age": age,
            "TypeofContact": typeof_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": num_persons_visiting,
            "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": num_trips,
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": pitch_satisfaction,
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": num_children,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        st.subheader("Prediction Result")

        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.success("✅ **Likely to PURCHASE** the Wellness Tourism Package")
                st.balloons()
            else:
                st.warning("❌ **Unlikely to purchase** the Wellness Tourism Package")

            st.metric("Purchase Probability", f"{probability:.1%}")
            st.caption(
                "This prediction is generated by a model trained on historical customer "
                "data and is intended to support, not replace, the sales team's judgement."
            )

        with gauge_col:
            gauge_color = "#2E9E5B" if probability >= 0.5 else "#C4442E"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(196, 68, 46, 0.15)"},
                        {"range": [50, 100], "color": "rgba(46, 158, 91, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 2},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Model Insights ─────────────────────────────────────────────
with tab_insights:
    if metrics_info:
        st.subheader("How the champion model was chosen")
        st.write(
            "6 candidate algorithms were tuned with `GridSearchCV` and tracked in MLflow. "
            f"**{metrics_info['best_model']}** was selected by highest F1 score on the "
            "held-out test set."
        )

        results_df = pd.DataFrame(metrics_info["all_results"]).sort_values("f1", ascending=False)
        display_df = results_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
        display_df = display_df.rename(columns={"model": "Model"})
        st.dataframe(
            display_df.style.highlight_max(
                subset=["accuracy", "precision", "recall", "f1", "roc_auc"], color="#d4f4dd"
            ).format({c: "{:.3f}" for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]}),
            use_container_width=True,
            hide_index=True,
        )

        fig = go.Figure()
        metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df["model"],
                y=results_df[metric],
            ))
        fig.update_layout(
            barmode="group",
            title="Model comparison across metrics (held-out test set)",
            yaxis=dict(range=[0, 1]),
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Winning hyperparameters: `{metrics_info.get('best_params', {})}`"
        )
    else:
        st.info("Model metrics file not found — comparison chart unavailable.")
