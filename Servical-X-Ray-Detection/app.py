from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
import tensorflow as tf
import cv2

app = Flask(__name__)

# Load the saved model
model = load_model('trained_model.h5')

# Preprocess incoming images
def preprocess_image(image_file):
    # Read the image file
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to match the input size of your model
    image = cv2.resize(image, (512, 512))

    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to range [0, 1]
    image_gray = image_gray / 255.0

    # Reshape the image to match the input shape of the model
    image_input = np.expand_dims(image_gray, axis=0)
    image_input = np.expand_dims(image_input, axis=-1)
    print(image_input.shape)
    return image_input

# Define a route for predicting on images
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    # Preprocess the image
    image = preprocess_image(image_file)

    # Make predictions
    predictions = model.predict(np.array([image]))

    # Convert predictions to JSON format
    # Modify this part according to your model's output format
    prediction_json = {
        'fracture_probability': predictions[0][0],
        'c1c7_probabilities': predictions[1][0].tolist()
    }
    print(prediction_json)
    # Return the predictions as JSON
    return jsonify(prediction_json)

if __name__ == '__main__':
    app.run(debug=True)
