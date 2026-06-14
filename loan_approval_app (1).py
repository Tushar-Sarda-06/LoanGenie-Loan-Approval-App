import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
 
st.set_page_config(page_title="LoanGenie", layout="centered")
st.header("🏦 LoanGenie - Loan Approval Predictor")
st.write("Fill in the details below to check if your loan is likely to be approved.")
 
try:
    scaler = pk.load(open("scaler.pkl", "rb"))
    model  = pk.load(open("model.pkl",  "rb"))
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()
 
# ── Inputs ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    no_of_dep  = st.slider("Number of Dependents", 0, 5, 2)
    grad       = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employment = st.selectbox("Self Employed?", ["No", "Yes"])
    income     = st.slider("Annual Income (INR)", 200000, 9900000, 5000000, step=100000)
 
with col2:
    loan_amt   = st.slider("Loan Amount Required (INR)", 300000, 39500000, 15000000, step=100000)
    loan_dur   = st.slider("Loan Duration (Years)", 2, 20, 10)
    cibil      = st.slider("CIBIL Score", 300, 900, 650)
    assets     = st.slider("Total Assets Value (INR)", 0, 58000000, 10000000, step=100000)
 
# ── Encode inputs (must match training encoding exactly) ───────────────────
grad_enc = 1 if grad == "Graduate" else 0      # Graduate=1, Not Graduate=0
emp_enc  = 1 if employment == "Yes" else 0     # Yes=1, No=0
 
# ── Feature Engineering (must mirror Cell 8 exactly) ──────────────────────
def build_feature_vector(no_of_dep, grad_enc, emp_enc,
                          income, loan_amt, loan_dur, cibil, assets):
    X = {
        # Base features
        "no_of_dependents": no_of_dep,
        "education"        : grad_enc,
        "self_employed"    : emp_enc,
        "income_annum"     : income,
        "loan_amount"      : loan_amt,
        "loan_term"        : loan_dur,
        "cibil_score"      : cibil,
        "assets"           : assets,
        # Engineered features
        "cibil_bin"        : float(np.searchsorted([499,549,599,649,699,724,749,800], cibil)),
        "cibil_gte_700"    : int(cibil >= 700),
        "cibil_gte_750"    : int(cibil >= 750),
        "cibil_lt_500"     : int(cibil <  500),
        "cibil_sq"         : cibil ** 2,
        "loan_to_income"   : loan_amt  / (income + 1),
        "loan_to_assets"   : loan_amt  / (assets + 1),
        "income_per_dep"   : income    / (no_of_dep + 1),
        "cibil_x_income"   : cibil     * income,
        "log_income_annum" : np.log1p(income),
        "log_loan_amount"  : np.log1p(loan_amt),
        "log_assets"       : np.log1p(assets),
    }
    return pd.DataFrame([X])
 

if st.button("🔍 Predict Loan Approval"):
    try:
        input_df     = build_feature_vector(no_of_dep, grad_enc, emp_enc,
                                             income, loan_amt, loan_dur, cibil, assets)
        input_scaled = scaler.transform(input_df)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1]
 
        st.markdown("---")
        if prediction == 1:
            st.success(f"✅ Loan Approved  (Confidence: {probability*100:.1f}%)")
        else:
            st.error(f"❌ Loan Rejected  (Approval probability: {probability*100:.1f}%)")
 
        st.markdown(f"**CIBIL Score:** {cibil} | "
                    f"**Loan-to-Income Ratio:** {loan_amt/income:.2f}x | "
                    f"**Loan-to-Assets Ratio:** {loan_amt/max(assets,1):.2f}x")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
