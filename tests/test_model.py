import pytest
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = "models/fashion-model.h5"

# Load the model
@pytest.fixture(scope="module")
def model():
    model = load_model(MODEL_PATH)
    return model

def preprocess_image(image_path, target_size):
    # Load the image in grayscale mode
    image = load_img(image_path, target_size=target_size, color_mode="grayscale")
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    return image_array

# Test function for each image
@pytest.mark.parametrize("image_path, expected_class", [
    ("tests/test_images/sandal_test.jpeg", 5),
])
def test_model_prediction(model, image_path, expected_class):
    image_array = preprocess_image(image_path, target_size=(28, 28)) # Input to model is 28x28

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=-1)[0] # Get the class with the highest probability

    # Assert prediction
    assert predicted_class == expected_class, f"Expected {expected_class}, but got {predicted_class}"
