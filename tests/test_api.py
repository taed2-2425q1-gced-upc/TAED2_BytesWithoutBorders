# test_api.py
import pytest
from fastapi.testclient import TestClient
from byteswithoutborders.api import app  # Adjust the path as needed

# Create a TestClient instance to simulate HTTP requests
client = TestClient(app)

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
