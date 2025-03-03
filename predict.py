import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model_path = "models/brain_tumor_classifier.h5"
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Define class labels
class_labels = {0: "glioma", 1: "meningioma", 2: "no tumor", 3: "pituitary"}

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load image
    img = cv2.resize(img, (224, 224))  # Resize to match MobileNetV2 input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make a prediction
def predict_tumor(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the class with highest probability
    confidence = np.max(prediction)  # Get confidence score

    tumor_type = class_labels[predicted_class]
    return tumor_type, confidence

if __name__ == "__main__":
    image_path = input("Enter the path of the brain MRI image: ")
    
    if not os.path.exists(image_path):
        print("❌ Error: File not found!")
    else:
        tumor_type, confidence = predict_tumor(image_path)
        print(f"🧠 Prediction: {tumor_type.capitalize()} (Confidence: {confidence:.2f})")
