"""
Streamlit App - Wellness Tourism Package Purchase Predictor
Loads the model committed to the repository and predicts whether a prospective
customer is likely to purchase the Wellness Tourism Package.
"""
import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "tourism_project/deployment/model.joblib"
METRICS_PATH = "tourism_project/deployment/metrics.json"

st.set_page_config(
    page_title="Wellness Tourism Purchase Predictor",
    page_icon="🧳",
    layout="centered",
)


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
        st.subheader("Model")
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
        "Built as part of an end-to-end MLOps pipeline: GitHub Actions handles "
        "data validation, cleaning, model tuning with MLflow tracking, and "
        "redeploys this app automatically whenever the model is refreshed."
    )

# ── Main page ─────────────────────────────────────────────────────────
st.title("🧳 Wellness Tourism Package — Purchase Predictor")
st.write(
    "Enter a customer's details below to predict whether they are likely to "
    "purchase the newly launched **Wellness Tourism Package**."
)

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Profile")
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
        st.subheader("Trip & Pitch Details")
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

    result_col, gauge_col = st.columns([2, 1])
    with result_col:
        if prediction == 1:
            st.success("✅ **Likely to PURCHASE** the Wellness Tourism Package")
        else:
            st.warning("❌ **Unlikely to purchase** the Wellness Tourism Package")
    with gauge_col:
        st.metric("Purchase Probability", f"{probability:.1%}")

    st.progress(float(probability))

    st.caption(
        "This prediction is generated by a model trained on historical customer data "
        "and is intended to support, not replace, the sales team's judgement."
    )
