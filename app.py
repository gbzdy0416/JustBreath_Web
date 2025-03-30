from flask import Flask, jsonify, request, render_template
import tensorflow as tf
import numpy as np
app = Flask(__name__)

model = tf.keras.models.load_model("BreathingModel.h5")

def preprocess_input(raw_data):
    goal = raw_data['goal']
    intensity_function = raw_data['intensity_function'] / 100.0
    intensity_time = raw_data['intensity_time'] / 100.0
    return np.array([goal, intensity_function, intensity_time])


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    features = preprocess_input(data) #normalization
    prediction = model.predict(features.reshape(1, -1))[0]
    output_min = np.array([746, 3, 5, 4])  # tick_length, inhale, exhale, repetition
    output_max = np.array([1250, 9, 10, 9])
    tick_length, inhale,  exhale, repetition = prediction * (output_max - output_min) + output_min
    return jsonify({
        "inhale_ms": int(inhale * tick_length),
        "exhale_ms": int(exhale * tick_length),
        "repetition": int(repetition)
    })

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)