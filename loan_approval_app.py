import streamlit as st 
import pandas as pd 
import pickle as pk 

st.set_page_config(page_title="LoanGenie", layout="centered")

st.header('🏦 LoanGenie - Loan Approval Predictor')
st.write("Fill in the details below to check if your loan is likely to be approved.")
try:
    scaler = pk.load(open('scaler.pkl', 'rb'))  # This is actually the scaler
    model = pk.load(open('model.pkl', 'rb'))  # This is actually the model
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()
no_of_dep = st.slider('Number of Dependents', 0, 5)
grad = st.selectbox('Education Level', ['Graduated', 'Not Graduated'])
employment_status = st.selectbox('Self Employed?', ['Yes', 'No'])
Annual_Income = st.slider('Annual Income (INR)', 0, 10000000)
Loan_Amount_Required = st.slider('Loan Amount Required (INR)', 0, 10000000)
Loan_Duration = st.slider('Loan Duration (Years)', 0, 20)
Cibil = st.slider('CIBIL Score', 0, 1000)
Assets = st.slider('Total Assets Value (INR)', 0, 10000000)


grad_s = 0 if grad == 'Graduated' else 1
emp_s = 1 if employment_status == 'Yes' else 0

if st.button("Predict Loan Approval"):
    try:
        input_df = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount_Required,
                          Loan_Duration, Cibil, Assets]],
        columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
                 'loan_amount', 'loan_term', 'cibil_score', 'assets'])  

        
        input_scaled = scaler.transform(input_df)         
        prediction = model.predict(input_scaled)          

        if prediction[0] == 1:
            st.success('✅ Loan Is Approved')
        else:
            st.error('❌ Loan Is Rejected')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
