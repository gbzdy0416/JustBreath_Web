from flask import Flask, jsonify, request, render_template
import tensorflow as tf
import numpy as np
import secrets
import joblib
from losses import custom_loss

app = Flask(__name__)

model = tf.keras.models.load_model("BreathingModel_2.keras", custom_objects={"custom_loss": custom_loss})
scaler = joblib.load('scaler.save')
output_scaler = joblib.load('y_scaler.save')

def preprocess_input(raw_data):
    intensity_function = int(raw_data['intensity_function'])
    intensity_time = int(raw_data['intensity_time'])
    age_group = int(raw_data['age_group'])
    gender = int(raw_data['gender'])
    return scaler.transform(np.array([[intensity_function, intensity_time, age_group, gender]]))


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    features = preprocess_input(data) #normalization
    prediction = model.predict(features)
    inhale,  exhale, repetition, r, s = output_scaler.inverse_transform(prediction.reshape(1, -1))[0]
    mode = secrets.randbelow(100) % 2
    if mode == 1:
        inhale = 5
        exhale = 5
        repetition = 6
    return jsonify({
        "inhale": int(inhale * 1000),
        "exhale": int(exhale * 1000),
        "repetition": int(repetition),
        "mode": int(mode)
    })

@app.route('/ai', methods=['GET'])
def ai_breath():
    return render_template("index_copy.html")

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)