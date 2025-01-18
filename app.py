from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load pre-trained model and scaler
model = joblib.load('cardio_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            inputs = [
                float(request.form['age']),
                float(request.form['height']),
                float(request.form['weight']),
                int(request.form['gender']),
                float(request.form['ap_hi']),
                float(request.form['ap_lo']),
                int(request.form['cholesterol']),
                int(request.form['gluc']),
                int(request.form['smoke']),
                int(request.form['alco']),
                int(request.form['active'])
            ]

            inputs = np.array(inputs).reshape(1, -1)
            inputs_scaled = scaler.transform(inputs)
            prediction_prob = model.predict_proba(inputs_scaled)[0][1]

            if prediction_prob >= 0.5:
                prediction = "High Risk"
            else:
                prediction = "Low Risk"

            return render_template(
                'result.html',
                prediction=prediction,
                inputs=request.form,
                advice="Consider a healthy diet, regular exercise, and consultation with a doctor."
                if prediction == "High Risk" else "Maintain your healthy lifestyle."
            )
        except Exception as e:
            return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
