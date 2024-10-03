# download_data.py

from datasets import load_dataset

# Load the Fashion MNIST dataset
dataset = load_dataset("zalando-datasets/fashion_mnist")

# Save the dataset to local files
dataset['train'].save_to_disk('data/raw/fashion_mnist/train')
dataset['test'].save_to_disk('data/raw/fashion_mnist/test')

print("Fashion MNIST dataset downloaded and saved.")