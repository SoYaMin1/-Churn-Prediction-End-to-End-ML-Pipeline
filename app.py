# app.py
import streamlit as st
import pandas as pd
from src.inference import predict_churn

st.title("Прогнозирование оттока клиентов (Churn Prediction)")

st.markdown("Введите данные клиента, чтобы оценить вероятность оттока.")

# Форма ввода (упрощенная)
with st.form("customer_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (месяцы)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        
    submitted = st.form_submit_button("Предсказать")

if submitted:
    # Собираем словарь (остальные поля заполняем заглушками для простоты примера)
    customer_data = {
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': internet_service, 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': tech_support,
        'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': contract,
        'PaperlessBilling': 'Yes', 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    result = predict_churn(customer_data)
    
    if result['churn_prediction'] == 1:
        st.error(f"Клиент склонен к оттоку! Вероятность: {result['churn_probability']:.2%}")
    else:
        st.success(f"Клиент, скорее всего, останется. Вероятность оттока: {result['churn_probability']:.2%}")