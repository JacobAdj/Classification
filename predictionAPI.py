from flask import Flask, request, jsonify
import joblib
import pandas as pd


app = Flask(__name__)
model = joblib.load("./mlruns/0/0d66714ecebe4138b95f2f715569749f/artifacts/iris_model/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = pd.DataFrame([data])
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})


if __name__ == "__main__":

    app.run()