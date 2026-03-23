import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
model  = joblib.load(os.path.join(BASE_DIR, "logistic_regression_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
le_age  = joblib.load(os.path.join(BASE_DIR, "le_age.joblib"))
le_type = joblib.load(os.path.join(BASE_DIR, "le_type.joblib"))

THRESHOLD = 0.4

BUSINESS_AGE_CATEGORIES  = list(le_age.classes_)
BUSINESS_TYPE_CATEGORIES = list(le_type.classes_)

st.set_page_config(page_title="Credit Risk Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Credit Risk Predictor")
st.markdown("Fill in the loan and business details below to assess default risk.")
st.divider()

if "result" not in st.session_state:
    st.session_state.result = None

with st.form("prediction_form", clear_on_submit=False):
 
    st.subheader("Loan Details")
    col1, col2 = st.columns(2)
 
    with col1:
        grossapproval = st.number_input(
            "Gross Approval Amount ($)",
            min_value=1.0, value=25000.0, step=1000.0
        )
        sbaguaranteedapproval = st.number_input(
            "SBA Guaranteed Amount ($)",
            min_value=0.0, value=20000.0, step=1000.0
        )
        initialinterestrate = st.number_input(
            "Initial Interest Rate (%)",
            min_value=0.0, max_value=100.0, value=6.0, step=0.1
        )
        terminmonths = st.number_input(
            "Loan Term (months)",
            min_value=1, max_value=600, value=120, step=1
        )
        jobssupported = st.number_input(
            "Jobs Supported",
            min_value=0, value=3, step=1
        )
 
    with col2:
        fixedorvariableinterestind = st.selectbox(
            "Interest Rate Type",
            options=[0, 1],
            format_func=lambda x: "Fixed" if x == 0 else "Variable"
        )
        businessage_label = st.selectbox(
            "Business Age",
            options=BUSINESS_AGE_CATEGORIES
        )
        businesstype_label = st.selectbox(
            "Business Type",
            options=BUSINESS_TYPE_CATEGORIES
        )
 
    st.divider()
    st.subheader("Bank-Level Statistics")
    col3, col4 = st.columns(2)
 
    with col3:
        totalloancounts = st.number_input(
            "Total Loan Count (bank)", min_value=1, value=182, step=1
        )
        totaldefaultcounts = st.number_input(
            "Total Default Count (bank)", min_value=0, value=3, step=1
        )
        totalpctdefault = st.number_input(
            "Total Default Rate (bank)",
            min_value=0.0, max_value=1.0, value=0.0165, step=0.001, format="%.4f"
        )
        yearlyloancounts = st.number_input(
            "Yearly Loan Count (bank)", min_value=0, value=41, step=1
        )
 
    with col4:
        yearlydefaultcounts = st.number_input(
            "Yearly Default Count (bank)", min_value=0, value=3, step=1
        )
        cumulativedefault = st.number_input(
            "Cumulative Defaults (bank)", min_value=0, value=3, step=1
        )
        cumulativeloansissued = st.number_input(
            "Cumulative Loans Issued (bank)", min_value=1, value=82, step=1
        )
 
    submitted = st.form_submit_button(
        "Predict Risk", use_container_width=True, type="primary"
    )

if submitted:
    if sbaguaranteedapproval > grossapproval:
        st.error("SBA Guaranteed Amount cannot exceed Gross Approval Amount.")
        st.stop()
 
    businessage_encoded  = int(le_age.transform([businessage_label])[0])
    businesstype_encoded = int(le_type.transform([businesstype_label])[0])

    grossapproval_t   = np.log1p(grossapproval)
    jobssupported_t   = np.log1p(jobssupported)
    sbaguranteedscore = np.log1p(sbaguaranteedapproval) / np.log1p(grossapproval)
 
    features = pd.DataFrame([{
        "grossapproval":              grossapproval_t,
        "initialinterestrate":        initialinterestrate,
        "fixedorvariableinterestind": fixedorvariableinterestind,
        "businessage":                businessage_encoded,
        "businesstype":               businesstype_encoded,
        "terminmonths":               terminmonths,
        "jobssupported":              jobssupported_t,
        "sbaguranteedscore":          sbaguranteedscore,
        "totalloancounts":            totalloancounts,
        "totaldefaultcounts":         totaldefaultcounts,
        "totalpctdefault":            totalpctdefault,
        "yearlyloancounts":           yearlyloancounts,
        "yearlydefaultcounts":        yearlydefaultcounts,
        "cumulativedefault":          cumulativedefault,
        "cumulativeloansissued":      cumulativeloansissued,
    }])
 
    features_scaled = scaler.transform(features)
    prob = float(model.predict_proba(features_scaled)[0][1])
    prediction = int(prob >= THRESHOLD)
 
    st.session_state.result = {
        "prob":       prob,
        "prediction": prediction,
        "business":   businesstype_label,
        "age":        businessage_label,
    }

if st.session_state.result:
    r = st.session_state.result
    st.divider()
    st.subheader("Prediction Result")
 
    col_res, col_prob = st.columns(2)
    with col_res:
        if r["prediction"] == 1:
            st.error("⚠️ **High Risk** — Likely to Default")
        else:
            st.success("✅ **Low Risk** — Unlikely to Default")
    with col_prob:
        st.metric("Default Probability", f"{r['prob']:.1%}")
 
    st.markdown("**Risk Level**")
    st.progress(r["prob"])
    st.caption(
        f"Business type: {r['business']} | "
        f"Business age: {r['age']} | "
        f"Threshold: {THRESHOLD} | "
        f"Raw probability: {r['prob']:.4f}"
    )