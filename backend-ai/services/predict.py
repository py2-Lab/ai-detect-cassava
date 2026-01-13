import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# model = tf.keras.models.load_model(
#     "model/mobilenetv2_singkong2.keras",
#     compile=False
# )

try:
    model = tf.keras.models.load_model(
        "model/mobilenetv2_singkong2.keras",
        compile=False
    )
except Exception as e:
    print("ERROR loading model:", e)
    model = None

# SESUAI DATASET COLAB (2 KELAS)
class_names = ["matang", "mentah"]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict_image(image_bytes):
    image = preprocess_image(image_bytes)
    predictions = model.predict(image)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    return class_names[class_index], confidence
