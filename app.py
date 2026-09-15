import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"


from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import json
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(
    "model/flower_model.keras"
)

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

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

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