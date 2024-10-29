import pytest
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to your .h5 model file
MODEL_PATH = "models/fashion-model.h5"

# Load the model before tests
@pytest.fixture(scope="module")
def model():
    model = load_model(MODEL_PATH)
    return model

def preprocess_image(image_path, target_size):
    # Load the image in grayscale mode
    image = load_img(image_path, target_size=target_size, color_mode="grayscale")
    image_array = img_to_array(image)  # Convert to numpy array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize if needed

    # Expand dimensions to (batch, height, width, channels) as (1, height, width, 1)
    return image_array

# Test function for each image
@pytest.mark.parametrize("image_path, expected_class", [
    ("tests/test_images/sandal_test.jpeg", 5),  # Replace 0 with the expected class for image1
    # Add more images as needed
])
def test_model_prediction(model, image_path, expected_class):
    # Preprocess the image
    image_array = preprocess_image(image_path, target_size=(28, 28))  # Adjust target_size as per your model's input

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]  # Get the class with the highest probability

    # Assert that the prediction matches the expected class
    assert predicted_class == expected_class, f"Expected {expected_class}, but got {predicted_class}"
