# api/hospital_api.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
#frontEnd connection code
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Optional if called only from Node.js backend

#The ML model was giving error dur to hospital_api being in api directory so I changed its location
# Load trained model
model_path = 'hospital_recommender.pkl'
if not os.path.exists(model_path):
    raise Exception("❌ Model file not found. Train the model first.")
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return "HospiML API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        features = np.array([[
            data['available_beds'],
            data['occupied_beds'],
            data['recent_admissions'],
            data['rejection_rate'],
            data['avg_discharge_time'],
            data['avg_length_of_stay'],
            data['success_rate']
        ]])

        recommendation = model.predict(features)[0]
        probability = float(model.predict_proba(features)[0][1])  # confidence for class 1

        return jsonify({
            "recommendation": int(recommendation),
            "probability": round(probability, 2)
        })

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # 5000 is fallback if not running on Render
app.run(host='0.0.0.0', port=port)

