import numpy as np

class Perceptron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def predict(self, inputs):
        return 1 if np.dot(inputs, self.weights) + self.bias > 0 else 0

# Initialize perceptron
perceptron = Perceptron([0.2, 0.4, 0.2], -0.5)

# Test data (hero, heroine, climate)
test_data = [
    ([1, 1, 1], 1),  # All conditions true, should go
    ([0, 0, 0], 0),  # No conditions true, shouldn't go
    ([1, 1, 0], 1),  # Hero and heroine true, should go
    ([0, 1, 1], 1),  # Heroine and climate true, should go
    ([1, 0, 0], 0),  # Only hero true, shouldn't go
]

correct = sum(perceptron.predict(inputs) == output for inputs, output in test_data)
accuracy = correct / len(test_data) * 100

print(f"Accuracy: {accuracy}%")
