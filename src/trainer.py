import numpy as np
import neural_network as nn
import loader as ldr

# Initalize seed
np.random.seed(42)

# Load training data
training_data_path = 'dataset/train'
images, labels = ldr.load_dataset(training_data_path)

# Initialize network
layer_sizes = [784, 128, 64, 10]
net = nn.NeuralNetwork()
net.initalize(layer_sizes)

# Train network
net.train(images, labels, epochs=200, learning_rate=0.1, plot=True)

# Save network
net.save('models/network.npz')