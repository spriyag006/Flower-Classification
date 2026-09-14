
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os


app = Flask(__name__)


model = tf.keras.models.load_model("model/flower_model.keras")


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

  
    image = Image.open(file).convert("RGB")

  
    image = image.resize((224, 224))


    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array)

  
    predicted_index = np.argmax(predictions[0])

  
    predicted_class = class_names[predicted_index]

  
    confidence = float(predictions[0][predicted_index]) * 100

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2)
    )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)
