import os
import pandas as pd
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("zalando-datasets/fashion_mnist")

# Ensure the data directory exists
os.makedirs("../data/processed", exist_ok=True)

# Convert to pandas DataFrames
train_data = dataset['train'].to_pandas()
test_data = dataset['test'].to_pandas()

# Save to CSV
train_data.to_csv("data/processed/fashion_mnist_train.csv", index=False)
test_data.to_csv("data/processed/fashion_mnist_test.csv", index=False)

print("Data has been saved to CSV files.")
