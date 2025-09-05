import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Credit Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # expecting JSON input
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
