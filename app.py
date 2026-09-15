import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import json

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback if only full tensorflow is installed locally
    import tensorflow.lite as tflite


app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model/flower_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print("Classes:", class_names)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "flower" not in request.files:
        return "No image uploaded"

    file = request.files["flower"]

    if file.filename == "":
        return "No image selected"

    # Open image
    image = Image.open(file).convert("RGB")

    # Resize
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction via TFLite interpreter
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = (
        float(predictions[0][predicted_index]) * 100
    )

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )