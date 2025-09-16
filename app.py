# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the saved models and scaler
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

nn_model = load_model('stress_level_nn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = [
            float(request.form['snoring_rate']),
            float(request.form['respiration_rate']),
            float(request.form['body_temperature']),
            float(request.form['limb_movement']),
            float(request.form['blood_oxygen']),
            float(request.form['eye_movement']),
            float(request.form['sleeping_hours']),
            float(request.form['heart_rate'])
        ]

        # Convert input data to numpy array and scale it
        input_data_scaled = scaler.transform(np.array([input_data]))

        # Predict using Random Forest
        rf_prediction = rf_model.predict(input_data_scaled)[0]

        # Predict using Neural Network
        nn_prediction = np.argmax(nn_model.predict(input_data_scaled), axis=1)[0]

        # Map predictions to stress levels
        stress_levels = {0: "Low", 1: "Medium", 2: "High"}
        rf_prediction_label = stress_levels[rf_prediction]
        nn_prediction_label = stress_levels[nn_prediction]

        # Return predictions as JSON
        return jsonify({
            "Random Forest Prediction": rf_prediction_label,
            "Neural Network Prediction": nn_prediction_label
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)