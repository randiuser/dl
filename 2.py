import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=0)

functions = [sigmoid, tanh, relu, leaky_relu, softmax]
x = np.linspace(-10, 10, 1000)

plt.figure(figsize=(15, 10))
for i, func in enumerate(functions, 1):
    plt.subplot(2, 3, i)
    plt.plot(x, func(x))
    plt.title(func.__name__)
    plt.grid(True)

plt.tight_layout()
plt.show()
