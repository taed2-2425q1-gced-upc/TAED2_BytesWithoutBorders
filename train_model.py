# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load training data
train_data = pd.read_csv("data/processed/fashion_mnist_train.csv")
X = train_data.drop('label', axis=1)  # Assuming 'label' is the target column
y = train_data['label']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "models/model.pkl")
