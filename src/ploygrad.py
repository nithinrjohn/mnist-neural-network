import numpy as np

np.random.seed(42)

data = [
    (-4, -68),
    (-3, -30),
    (-2, -10),
    (-1, -2),
    (0, 0),
    (1, 2),
    (2, 10),
    (3, 30),
    (4, 68),
]

x = np.array([x for x, y in data], dtype=float)
y = np.array([y for x, y in data], dtype=float)

# Initialize parameters
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-4
epochs = 100000

for epoch in range(epochs + 1):
    # Forward
    y_pred = a * x**3 + b * x**2 + c * x + d

    # Loss
    loss = np.mean((y_pred - y) ** 2)

    if epoch % 500 == 0:
        print(f"[{epoch:4d}/{epochs}] loss: {loss:.6f}")

    # Backward
    error = y_pred - y

    dLda = np.mean(2 * error * x**3)
    dLdb = np.mean(2 * error * x**2)
    dLdc = np.mean(2 * error * x)
    dLdd = np.mean(2 * error)

    # Gradient descent
    a -= learning_rate * dLda
    b -= learning_rate * dLdb
    c -= learning_rate * dLdc
    d -= learning_rate * dLdd

print(f"y = {a:.4f}x^3 + {b:.4f}x^2 + {c:.4f}x + {d:.4f}")