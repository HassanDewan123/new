from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model and encoders using absolute paths
base_path = os.path.dirname(__file__)
rf_model_path = os.path.join(base_path, "models", "rf_model.pkl")
label_encoders_path = os.path.join(base_path, "models", "label_encoders.pkl")

rf_model = joblib.load(rf_model_path)
label_encoders = joblib.load(label_encoders_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from the form
        user_data = {
            'gender': request.form['gender'].capitalize(),
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'].capitalize(),
            'work_type': request.form['work_type'].capitalize(),
            'Residence_type': request.form['Residence_type'].capitalize(),
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status'].capitalize()
        }

        # Encode categorical inputs
        for col in label_encoders:
            user_data[col] = label_encoders[col].transform([user_data[col]])[0]

        # Create a DataFrame for the model
        user_df = pd.DataFrame([user_data])

        # Make prediction
        prediction = rf_model.predict(user_df)[0]
        prediction_prob = rf_model.predict_proba(user_df)[0][1]

        # Return prediction result
        result = {
            "prediction": "Stroke risk detected" if prediction == 1 else "No stroke risk",
            "probability": round(prediction_prob, 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
