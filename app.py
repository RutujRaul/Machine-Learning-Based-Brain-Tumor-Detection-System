import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load trained model
model_path = "models/brain_tumor_classifier.h5"
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Define class labels
class_labels = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_tumor(img_path):
    """Predict tumor type from MRI image."""
    img = load_img(img_path, target_size=(224, 224))  # Load image
    img = img_to_array(img)  # Convert to NumPy array

    # Convert grayscale images to RGB (model expects 3 channels)
    if img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize

    # Predict
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_labels[class_index], confidence

@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handle file upload and prediction."""
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected!")

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Get prediction
            prediction, confidence = predict_tumor(file_path)

            return render_template(
                "result.html",
                prediction=prediction,
                confidence=confidence,
                image_url=file_path,
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
