from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    rooms = int(request.form["rooms"])
    bathrooms = int(request.form["bathrooms"])
    kitchens = int(request.form["kitchens"])
    washrooms = int(request.form["washrooms"])

    features = np.array([[rooms, bathrooms, kitchens, washrooms]])

    prediction = model.predict(features)

    price = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted House Price: ₹ {price}"
    )

if __name__ == "__main__":
    app.run(debug=True)