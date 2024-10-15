import pytest
from typer.testing import CliRunner
from pathlib import Path
from train import app as train_app
from predict import app as predict_app
from byteswithoutborders.config import MODELS_DIR, PROCESSED_DATA_DIR

runner = CliRunner()

# Test if the training runs without errors
def test_train():
    # Define the file paths
    features_path = PROCESSED_DATA_DIR / "features.csv"
    labels_path = PROCESSED_DATA_DIR / "labels.csv"
    model_path = MODELS_DIR / "model.pkl"
    
    # Create dummy files for testing
    features_path.touch()  # Create an empty file to simulate test data
    labels_path.touch()    # Create an empty file to simulate test labels
    
    # Run the training command
    result = runner.invoke(train_app, [str(features_path), str(labels_path), str(model_path)])
    
    # Assert that the training completed without errors
    assert result.exit_code == 0
    assert "Modeling training complete" in result.output

    # Check if the model file was created (optional)
    assert model_path.exists()

# Test if the prediction runs without errors
def test_predict():
    # Define the file paths
    features_path = PROCESSED_DATA_DIR / "test_features.csv"
    model_path = MODELS_DIR / "model.pkl"
    predictions_path = PROCESSED_DATA_DIR / "test_predictions.csv"
    
    # Create dummy files for testing
    features_path.touch()  # Create an empty file to simulate test features
    model_path.touch()     # Create an empty model file for testing
    
    # Run the prediction command
    result = runner.invoke(predict_app, [str(features_path), str(model_path), str(predictions_path)])
    
    # Assert that the prediction completed without errors
    assert result.exit_code == 0
    assert "Inference complete" in result.output
    
    # Check if the predictions file was created (optional)
    assert predictions_path.exists()
