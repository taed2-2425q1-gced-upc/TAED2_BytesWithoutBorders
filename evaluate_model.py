# evaluate_model.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load test data and model
test_data = pd.read_csv("data/processed/fashion_mnist_test.csv")
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

model = joblib.load("models/model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
