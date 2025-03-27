import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("health_data.csv")  # Ensure your dataset is named correctly

# Encode categorical variables
label_encoders = {}
categorical_columns = ["Gender", "Smoking", "Alcohol Consumption", "Exercise Frequency", 
                        "Diet Quality", "Stress Level", "Pre-existing Conditions"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# Map health risk to numerical values
risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["Predicted Health Risk"] = df["Predicted Health Risk"].map(risk_mapping)

# Define features and target
X = df.drop(columns=["Predicted Health Risk"])
y = df["Predicted Health Risk"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test) * 100  # Convert to percentage

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
import joblib
joblib.dump(model, "health_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model and preprocessing objects saved successfully.")
