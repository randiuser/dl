from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
X, y = fetch_openml('CIFAR_10', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32') / 255.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate SVM
svm = LinearSVC(random_state=42).fit(X_train, y_train)
print(f"SVM Accuracy: {accuracy_score(y_test, svm.predict(X_test)):.4f}")

# Train and evaluate Softmax
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42).fit(X_train, y_train)
print(f"Softmax Accuracy: {accuracy_score(y_test, softmax.predict(X_test)):.4f}")
