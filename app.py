import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your Random Forest model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Credit Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON input like:
    {
        "V1": value,
        "V2": value,
        ...
        "Amount": value,
        "Time": value
    }
    """
    data = request.json
    df = pd.DataFrame([data])

    # Optional: scale 'Amount' and 'Time' if your model expects it
    scaler = StandardScaler()
    if 'Amount' in df.columns:
        df['Amount'] = scaler.fit_transform(df[['Amount']])
    if 'Time' in df.columns:
        df['Time'] = scaler.fit_transform(df[['Time']])

    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    # Use Render's port environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
