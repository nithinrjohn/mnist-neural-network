import os
import numpy as np
from PIL import Image

def load_img(img_path, resize_dim=(28, 28)):
    with Image.open(img_path).convert('L') as img:
        img_resized = img.resize(resize_dim)
        img_array = np.array(img_resized).flatten() / 255.0
    return img_array.reshape(-1, 1) # reshape to (features, 1) so it matches the (features, N) shape forward() expects

def load_dataset(folder_path):
    images = []
    labels = []
    for label_str in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, label_str)
        if not os.path.isdir(label_path):
            continue
        label_int = int(label_str)
        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)
            img_vector = load_img(file_path).flatten()
            images.append(img_vector)
            labels.append(label_int)
    X = np.array(images).T  # shape: (features, samples)
    Y = np.zeros((10, len(labels)))
    for i, label in enumerate(labels):
        Y[label, i] = 1
    return X, Y