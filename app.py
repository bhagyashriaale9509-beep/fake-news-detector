from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        news = request.form["news"]

        data = vectorizer.transform([news])

        prob = model.predict_proba(data)[0]
        result = model.predict(data)[0]

        confidence = max(prob) * 100

        if result == 1:
            prediction = f"✅ Real News ({confidence:.2f}% confidence)"
        else:
            prediction = f"❌ Fake News ({confidence:.2f}% confidence)"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)