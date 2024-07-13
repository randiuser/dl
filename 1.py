import numpy as np

def hebbian(x, y, w, lr):
    return w + lr * np.outer(y, x)

def perceptron(x, y, w, lr):
    y_pred = np.sign(np.dot(w, x))
    return w + lr * np.outer(y - y_pred, x)

def delta(x, y, w, lr):
    y_pred = np.dot(w, x)
    return w + lr * np.outer(y - y_pred, x)

def correlation(x, y, w, lr):
    return w + lr * np.outer(y - np.dot(w, x), x)

def outstar(x, y, w, lr):
    return w + lr * np.outer(y, x) - lr * w

# Example usage
x = np.array([1, 2, 3])
y = np.array([0, 1])
w = np.random.rand(2, 3)
lr = 0.1

rules = [hebbian, perceptron, delta, correlation, outstar]
for rule in rules:
    w_new = rule(x, y, w, lr)
    print(f"{rule.__name__} output:\n{w_new}\n")
