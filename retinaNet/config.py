import torch
BATCH_SIZE = 4 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 100 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.
CONFIDENCE_THRESHOLD = 0.5
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')
# Training images and XML files directory.
TRAIN_DIR = 'data/BCCD/train'
# Validation images and XML files directory.
VALID_DIR = 'data/BCCD/valid'
# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'RBC', 'WBC', 'Platelets'
]
NUM_CLASSES = len(CLASSES)
# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False
# Location to save model and plots.
OUT_DIR = 'outputs'