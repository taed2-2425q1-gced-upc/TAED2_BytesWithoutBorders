import numpy as np
import pytest
from fastapi.testclient import TestClient
from byteswithoutborders.api import app
import io
from PIL import Image

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def initialize_startup_events():
    # This context will trigger the startup event
    with client:
        yield

def test_health_check():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info():
    """Test the model information endpoint."""
    response = client.get("/models/info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "Fashion Classifier",
        "version": "1.2",
        "architecture": "CNN",
        "input_shape": [28, 28, 1],
        "trained_on": "Fashion MNIST",
        "num_classes": 10,
    }


def test_training_data_insights():
    """Test the training data insights endpoint."""
    response = client.get("/data/insights")
    assert response.status_code == 200
    data = response.json()
    assert "class_distribution" in data
    assert "total_images" in data


def test_feedback():
    """Test the feedback submission endpoint."""
    feedback_data = {
        "predicted_class": 0,
        "actual_class": 1,
        "image_data": [0] * 784,
        "comments": "Test feedback"
    }
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    assert response.json()["message"] == "Feedback received"

test_picture_raw =  [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 13, 73, 0, 0, 1, 4, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 36, 136, 127, 62, 54, 0, 0, 0, 1, 3, 4, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 102, 204, 176, 134, 144, 123, 23, 0, 0, 0, 0, 12, 10, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 155, 236, 207, 178, 107, 156, 161, 109, 64, 23, 77, 130, 72, 15],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 69, 207, 223, 218, 216, 216, 163, 127, 121, 122, 146, 141, 88, 172,
             66],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 200, 232, 232, 233, 229, 223, 223, 215, 213, 164, 127, 123, 196,
             229, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 183, 225, 216, 223, 228, 235, 227, 224, 222, 224, 221, 223, 245,
             173, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 193, 228, 218, 213, 198, 180, 212, 210, 211, 213, 223, 220, 243,
             202, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 12, 219, 220, 212, 218, 192, 169, 227, 208, 218, 224, 212, 226, 197,
             209, 52],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 99, 244, 222, 220, 218, 203, 198, 221, 215, 213, 222, 220, 245, 119,
             167, 56],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 55, 236, 228, 230, 228, 240, 232, 213, 218, 223, 234, 217, 217, 209,
             92, 0],
            [0, 0, 1, 4, 6, 7, 2, 0, 0, 0, 0, 0, 237, 226, 217, 223, 222, 219, 222, 221, 216, 223, 229, 215, 218, 255,
             77, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 62, 145, 204, 228, 207, 213, 221, 218, 208, 211, 218, 224, 223, 219, 215, 224,
             244, 159, 0],
            [0, 0, 0, 0, 18, 44, 82, 107, 189, 228, 220, 222, 217, 226, 200, 205, 211, 230, 224, 234, 176, 188, 250,
             248, 233, 238, 215, 0],
            [0, 57, 187, 208, 224, 221, 224, 208, 204, 214, 208, 209, 200, 159, 245, 193, 206, 223, 255, 255, 221, 234,
             221, 211, 220, 232, 246, 0],
            [3, 202, 228, 224, 221, 211, 211, 214, 205, 205, 205, 220, 240, 80, 150, 255, 229, 221, 188, 154, 191, 210,
             204, 209, 222, 228, 225, 0],
            [98, 233, 198, 210, 222, 229, 229, 234, 249, 220, 194, 215, 217, 241, 65, 73, 106, 117, 168, 219, 221, 215,
             217, 223, 223, 224, 229, 29],
            [75, 204, 212, 204, 193, 205, 211, 225, 216, 185, 197, 206, 198, 213, 240, 195, 227, 245, 239, 223, 218,
             212, 209, 222, 220, 221, 230, 67],
            [48, 203, 183, 194, 213, 197, 185, 190, 194, 192, 202, 214, 219, 221, 220, 236, 225, 216, 199, 206, 186,
             181, 177, 172, 181, 205, 206, 115],
            [0, 122, 219, 193, 179, 171, 183, 196, 204, 210, 213, 207, 211, 210, 200, 196, 194, 191, 195, 191, 198, 192,
             176, 156, 167, 177, 210, 92],
            [0, 0, 74, 189, 212, 191, 175, 172, 175, 181, 185, 188, 189, 188, 193, 198, 204, 209, 210, 210, 211, 188,
             185, 174, 183, 203, 142, 15],
            [0, 0, 0, 0, 43, 83, 136, 146, 172, 174, 179, 183, 183, 181, 176, 166, 167, 171, 186, 193, 203, 192, 190,
             183, 169, 82, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]


def test_predict_single_image_raw():
    """Test the single-image prediction endpoint."""
    test_data = {
        "image_data": test_picture_raw
    }
    response = client.post("/predict/single-image/raw", json=test_data)
    assert response.status_code == 200
    assert "predicted_class" in response.json()


def test_predict_double_image_raw():
    """Test the single-image prediction endpoint."""
    test_data = {
        "image_data_1": test_picture_raw,
        "image_data_2": test_picture_raw
    }
    response = client.post("/predict/double-image/raw", json=test_data)
    assert response.status_code == 200
    assert "predicted_class_image_1" in response.json()
    assert "predicted_class_image_2" in response.json()


def create_dummy_image():
    # Create a 28x28 grayscale image (similar to Fashion MNIST)
    dummy_data = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    image = Image.fromarray(dummy_data, 'L')  # 'L' mode for grayscale
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes.read()


def test_predict_single_image_file():
    """Test the single-image prediction endpoint with correct input format."""
    # Generate a dummy image
    image_data = create_dummy_image()

    # Send a POST request to the endpoint with the correct file format
    response = client.post(
        "/predict/single-image/uploadFile",
        files={"file": ("dummy_image.png", image_data, "image/png")}
    )

    # Assert the response is successful and has the expected output format
    assert response.status_code == 200
    response_data = response.json()
    assert "status" in response_data
    assert response_data["status"] == "success"
    assert "data" in response_data
    assert "image_1" in response_data["data"]
    assert "predicted_class" in response_data["data"]["image_1"]



def test_predict_double_image_upload_file():
    """Test the double-image prediction endpoint."""
    image_data_1 = create_dummy_image()
    image_data_2 = create_dummy_image()
    response = client.post(
        "/predict/double-image/uploadFile",
        files={
            "file_1": ("dummy_image_1.png", image_data_1, "image/png"),
            "file_2": ("dummy_image_2.png", image_data_2, "image/png"),
        }
    )
    assert response.status_code == 200
    assert "predicted_class_image_1" in response.json()
    assert "predicted_class_image_2" in response.json()
