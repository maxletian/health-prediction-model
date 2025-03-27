from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and preprocessing objects
model = joblib.load("health_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Define disease prevention mapping
disease_prevention = {
    "Obesity": "Maintain a balanced diet, exercise regularly, and stay hydrated.",
    "Lung Disease": "Reduce or quit smoking, practice breathing exercises, and stay active.",
    "Liver Disease": "Limit alcohol intake and maintain a liver-friendly diet.",
    "Heart Disease": "Engage in daily physical activity like walking or yoga.",
    "Hypertension": "Practice mindfulness, meditation, and get enough sleep.",
    "Diabetes": "Monitor blood sugar, eat fiber-rich foods, and maintain a healthy weight."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    # Convert input data to DataFrame
    input_data = pd.DataFrame([data])

    print("Input Data:", input_data)

    # Encode categorical data
    for col in label_encoders:
        try:
            input_data[col] = label_encoders[col].transform([input_data[col][0]])[0]
        except ValueError:
            input_data[col] = -1  # Assign a default label (e.g., -1) for unseen values

    # Convert to numerical values
    input_data = input_data.astype(float)

    # Scale numerical features
    input_scaled = scaler.transform(input_data)

    # Predict base risk probability
    base_risk = model.predict_proba(input_scaled)[0][1]

    # Adjust risk percentage based on factors
    risk_percentage = base_risk * 100  # Convert to percentage

    # Extract individual health factors
    age = int(data.get("Age", 30))
    bmi = float(data.get("BMI", 25))
    smoking = data.get("Smoking", "No")
    alcohol = data.get("Alcohol Consumption", "None")
    exercise = data.get("Exercise Frequency", "Regular")
    stress = data.get("Stress Level", "Moderate")
    pre_conditions = data.get("Pre-existing Conditions", "None")

    # Adjust risk based on Age
    if age > 60:
        risk_percentage += 15  # Senior citizens have higher risk
    elif age >= 40:
        risk_percentage += 8  # Middle-aged adults have moderate risk increase
    elif age >= 30:
        risk_percentage += 5  # Younger adults have lower risk increase

    # Adjust based on other factors
    if bmi > 30:
        risk_percentage += 10  # Higher BMI increases risk
    if smoking == "Yes":
        risk_percentage += 15  # Smoking increases risk
    if alcohol == "High":
        risk_percentage += 10  # Heavy alcohol consumption increases risk
    if exercise == "None":
        risk_percentage += 8  # Lack of exercise increases risk
    if stress == "High":
        risk_percentage += 5  # High stress contributes to risk
    if pre_conditions in ["Diabetes", "Hypertension"]:
        risk_percentage += 12  # Existing conditions raise risk

    # Ensure risk percentage stays within realistic bounds (10% - 95%)
    risk_percentage = max(10, min(risk_percentage, 95))

    # Identify potential diseases
    diseases = []
    if bmi > 30:
        diseases.append("Obesity")
    if smoking == "Yes":
        diseases.append("Lung Disease")
    if alcohol == "High":
        diseases.append("Liver Disease")
    if exercise == "None":
        diseases.append("Heart Disease")
    if stress == "High":
        diseases.append("Hypertension")
    if pre_conditions in ["Diabetes", "Hypertension"]:
        diseases.append("Diabetes")

    # Generate prevention tips
    prevention_methods = [disease_prevention[d] for d in diseases if d in disease_prevention]

    return jsonify({
        "risk_percentage": round(risk_percentage, 2),
        "potential_diseases": diseases,
        "prevention_methods": prevention_methods
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
