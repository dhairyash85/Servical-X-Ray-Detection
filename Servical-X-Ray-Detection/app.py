from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
import tensorflow as tf
import cv2

app = Flask(__name__)

model = load_model('trained_model.h5')

def preprocess_image(image_file):
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    image = cv2.resize(image, (512, 512))

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_gray = image_gray / 255.0

    image_input = np.expand_dims(image_gray, axis=0)
    image_input = np.expand_dims(image_input, axis=-1)
    print(image_input.shape)
    return image_input

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']

    image = preprocess_image(image_file)

    predictions = model.predict(np.array([image]))

    prediction_json = {
        'fracture_probability': predictions[0][0],
        'c1c7_probabilities': predictions[1][0].tolist()
    }
    print(prediction_json)
    return jsonify(prediction_json)

if __name__ == '__main__':
    app.run(debug=True)
