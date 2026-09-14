
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# Create Flask application
app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/flower_model.keras")

# Load class names
with open("model/class_names.json", "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print("Classes:", class_names)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Check if image was uploaded
    if "flower" not in request.files:
        return "No image uploaded"

    file = request.files["flower"]

    # Check if file was selected
    if file.filename == "":
        return "No image selected"

    # Open image
    image = Image.open(file).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image)

   

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array)

    # Get highest probability
    predicted_index = np.argmax(predictions[0])

    # Get predicted flower name
    predicted_class = class_names[predicted_index]

    # Calculate confidence
    confidence = float(predictions[0][predicted_index]) * 100

    # Show result
    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2)
    )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)
