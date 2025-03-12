import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import StandardScaler

def load_chemical_data(filepath):
    """Load and preprocess chemical process data."""
    data = pd.read_csv(filepath)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def load_image_data(image_dir, img_size=(128, 128)):
    """Load and preprocess image data."""
    images = []
    for filename in os.listdir(image_dir):
        img = cv2.imread(os.path.join(image_dir, filename))
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images)
