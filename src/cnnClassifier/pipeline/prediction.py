import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model_path = os.path.join("artifacts", "training", "model.h5")
        model = load_model(model_path)

        # Preprocess image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0  # Ensure the image is scaled properly

        # Predict
        predictions = model.predict(test_image)
        print("Raw model predictions:", predictions)  # Debugging line
        result = np.argmax(predictions, axis=1)
        print("Predicted class index:", result)  # Debugging line

        # Interpretation of results
        class_indices = {0: 'Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}
        if result[0] in class_indices:
            prediction = class_indices[result[0]]
            return [{"image": prediction}]
        else:
            return [{"image": "Unknown"}]
