from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load only the model
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]   # expecting list of numbers
    X = np.array(data).reshape(1, -1)
    
    prediction = model.predict(X)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
