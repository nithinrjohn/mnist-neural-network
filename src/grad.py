import numpy as np

np.random.seed(42)

data = [
    (-4, -8),
    (-3, -6),
    (-2, -4),
    (-1, -2),
    (0, 0),
    (1, 2),
    (2, 4),
    (3, 6),
    (4, 8),
]

x = np.array([x for x, y in data], dtype=float)
y = np.array([y for x, y in data], dtype=float)

# Initialize parameters
w = np.random.randn()
b = 0.0

learning_rate = 1e-2
epochs = 100

for epoch in range(epochs + 1):
    # Forward
    y_pred = w * x + b

    # Loss
    loss = np.mean((y_pred - y) ** 2)
    if epoch % 10 == 0:
        print(f"[{epoch:4d}/{epochs}] loss: {loss:.4f}")

    # Backward
    dLdw = np.mean((2 * x) * (y_pred - y))
    dLdb = np.mean(2 * (y_pred - y))

    # Update
    w -= learning_rate * dLdw
    b -= learning_rate * dLdb

print(f"y = {w:.4f}x + {b:.4f}")