import joblib
import pandas as pd

def predict_churn(customer_data: dict):
    """
    Загружает сохраненный пайплайн и делает предсказание.
    """
    model = joblib.load('models/churn_pipeline.joblib')
    
    # Преобразуем словарь в DataFrame (одна строка)
    df = pd.DataFrame([customer_data])
    
    # Предсказание
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(probability, 4),
        "risk_level": "High" if probability > 0.5 else "Low"
    }

if __name__ == "__main__":
    # Пример: новый клиент
    new_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }
    
    result = predict_churn(new_customer)
    print(result)