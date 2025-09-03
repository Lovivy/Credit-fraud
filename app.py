from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    # Scale Time (0th) and Amount (last)
    features[:, [0, -1]] = scaler.transform(features[:, [0, -1]])
    proba = model.predict_proba(features)[0, 1]
    prediction = int(proba > 0.5)
    return jsonify({"prediction": prediction, "probability": float(proba)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
