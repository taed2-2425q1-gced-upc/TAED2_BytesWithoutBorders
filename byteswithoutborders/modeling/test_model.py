from byteswithoutborders.modeling.train import main as train_main
from byteswithoutborders.modeling.predict import main as predict_main

def test_train_model():
    # Call the train function
    result = train_main()
    
    # Assert that the training was successful
    assert result is True, "Training did not complete successfully."

def test_predict_model():
    # Call the predict function
    result = predict_main()
    
    # Assert that the prediction was successful
    assert result is True, "Prediction did not complete successfully."