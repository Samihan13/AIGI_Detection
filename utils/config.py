# config.py
import os


# Base directory for dataset
BASE_DIR = '/data'
train_dir = os.path.join(BASE_DIR, 'train')
test_dir = os.path.join(BASE_DIR, 'test')

# Check if the directories exist
print("Train directory exists:", os.path.exists(train_dir))
print("Test directory exists:", os.path.exists(test_dir))


# Paths to train and test data
DATA_PATHS = {
    'train': os.path.join(BASE_DIR, 'train'),
    'test': os.path.join(BASE_DIR, 'test')
}

# Training parameters
TRAINING_PARAMS = {
    'batch_size': 64,
    'num_epochs': 25,
    'learning_rate': 0.001,
    'momentum': 0.9
}

# Model parameters
MODEL_PARAMS = {
    'num_classes': 2,
    'pretrained': False
}
"""
# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
"""