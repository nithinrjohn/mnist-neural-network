import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    # ReLU activation function
    return np.maximum(0, x)

def relu_derivative(x):
    # Derivative of ReLU
    return np.where(x > 0, 1, 0)

def softmax(logits):
    # Softmax activation function
    logits_exp = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    return logits_exp / np.sum(logits_exp, axis=0, keepdims=True)

def cross_entropy_loss(output, labels):
    # Compute cross-entropy loss
    m = labels.shape[1]
    loss = -np.sum(labels * np.log(output + 1e-8)) / m
    return loss

class NeuralNetwork:
    def __init__(self):
        self.weights = []
        self.biases = []

    def initalize(self, layer_sizes):
        # Initialize weights and biases
        self.layer_sizes = layer_sizes

        self.weights = []
        self.biases = []

        for i in range(1, len(layer_sizes)):
            # He initialization: scale by sqrt(2 / fan_in) instead of a flat 0.01,
            # so the variance of activations stays reasonable as layer sizes change
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2 / layer_sizes[i-1])
            bias_vector = np.zeros((layer_sizes[i], 1))

            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward(self, x):
        # Forward pass through the network
        activations = [x]
        zs = []

        for i in range(len(self.weights)):
            z = self.weights[i] @ activations[-1] + self.biases[i]
            zs.append(z)

            if i == len(self.weights) - 1:
                a = softmax(z)
            else:
                a = relu(z)
            activations.append(a)

        return activations, zs

    def backward(self, x, y, activations, zs):
        # Backward pass to compute gradients
        m = y.shape[1]  # number of samples in the batch, used to average gradients
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Compute error at output layer
        delta = activations[-1] - y

        # Gradients for last layer (averaged over the batch)
        grads_w[-1] = np.dot(delta, activations[-2].T) / m
        grads_b[-1] = np.sum(delta, axis=1, keepdims=True) / m

        # Propagate error backwards
        for layer_idx in range(len(self.weights) - 2, -1, -1):
            delta = self.weights[layer_idx+1].T @ delta * relu_derivative(zs[layer_idx])
            grads_w[layer_idx] = (delta @ activations[layer_idx].T) / m
            grads_b[layer_idx] = np.sum(delta, axis=1, keepdims=True) / m

        return grads_w, grads_b

    def optimize(self, grads_w, grads_b, learning_rate):
        # Gradients are already averaged over the batch in backward(), so just apply them
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def train(self, X, Y, epochs, learning_rate, plot=False):
            # Training loop
            loss_history = []
            accuracy_history = []

            if plot:
                plt.ion()  # Turn on interactive mode for real-time updating
                fig, ax_loss = plt.subplots(figsize=(7, 5))
                ax_acc = ax_loss.twinx()  # Second y-axis sharing the same x-axis, for accuracy

                line_loss, = ax_loss.plot([], [], color='#2980b9', linewidth=2, label='Loss')
                line_acc, = ax_acc.plot([], [], color='#27ae60', linewidth=2, label='Accuracy')

                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss', color='#2980b9')
                ax_acc.set_ylabel('Accuracy', color='#27ae60')
                ax_loss.set_title('Loss & Accuracy vs Epoch')

                ax_loss.tick_params(axis='y', labelcolor='#2980b9')
                ax_acc.tick_params(axis='y', labelcolor='#27ae60')
                ax_acc.set_ylim(0, 1.0)

                # Combined legend for both lines, placed below the plot so it
                # never overlaps the lines or the stats box
                ax_loss.legend(
                    handles=[line_loss, line_acc],
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=2,
                    frameon=True,
                    fontsize=9
                )

                # Leave room at the top (stats box) and bottom (legend) of the figure
                fig.subplots_adjust(bottom=0.2, top=0.82)

                # Stats box lives above the plot, in figure coordinates, so it's
                # fully outside the axes and never overlaps the lines
                stats_text = fig.text(
                    0.5, 0.93, '',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#bdc3c7', alpha=0.9),
                    fontsize=10,
                    color='#2c3e50'
                )
                
                # Ensure margins are set up
                ax_loss.set_xlim(0, epochs)

            for epoch in range(epochs):
                
                # Forward pass
                activations, zs = self.forward(X)

                # Calculate loss
                loss = cross_entropy_loss(activations[-1], Y)
                loss_history.append(loss)

                # Calculate accuracy: compare predicted class (argmax of softmax output)
                # against the true class (argmax of one-hot labels) for the full batch
                predictions = np.argmax(activations[-1], axis=0)
                true_labels = np.argmax(Y, axis=0)
                accuracy = np.mean(predictions == true_labels)
                accuracy_history.append(accuracy)

                # Backward pass
                grads_w, grads_b = self.backward(X, Y, activations, zs)

                # Update parameters
                self.optimize(grads_w, grads_b, learning_rate)

                # Real-time plot updates
                if plot:
                    current_epoch = epoch + 1
                    
                    # Update line data
                    line_loss.set_data(range(current_epoch), loss_history)
                    line_acc.set_data(range(current_epoch), accuracy_history)
                    
                    # Update the stats text above the plot
                    stats_text.set_text(
                        f"Epoch: {current_epoch}/{epochs}    "
                        f"Loss: {loss:.4f}    "
                        f"Accuracy: {accuracy:.2%}"
                    )
                    
                    # Dynamically scale the loss Y axis to fit the moving loss values
                    ax_loss.set_ylim(min(loss_history) * 0.9, max(loss_history) * 1.1)
                    
                    # Redraw canvas and process GUI events swiftly
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
        
            if plot:
                plt.ioff()  # Turn off interactive mode when done
                plt.show()  # Keep the final window open until closed manually

    def save(self, filename):
        # Save weights and biases to a file
        np.savez(filename, weights=np.array(self.weights, dtype=object), biases=np.array(self.biases, dtype=object))

    def load(self, filename):
        # Load weights and biases from a file
        data = np.load(filename, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])

    def predict(self, X):
        # Predict labels for input data
        activations, _ = self.forward(X)
        probs = activations[-1]
        predictions = np.argmax(probs, axis=0)

        prediction = predictions[0]
        prediction_prob = probs[predictions[0], 0]
        return prediction, prediction_prob