import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import neural_network as nn
import loader as ldr

IMAGE_PATH = "dataset/test/img_1.jpg"

# Initialize network
net = nn.NeuralNetwork()
net.load("models/network.npz")

x = ldr.load_img(IMAGE_PATH)
prediction, confidence = net.predict(x)

# Show prediction
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(np.array(Image.open(IMAGE_PATH)), cmap="gray")
ax.axis("off")
ax.set_title(f"Prediction: {prediction}    Confidence: {confidence:.2%}", fontsize=13)
plt.show()