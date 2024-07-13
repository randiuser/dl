import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load and preprocess data
data = pd.read_csv('sonar_dataset.csv', header=None)
X = data.iloc[:, :-1].values
y = (data.iloc[:, -1] == 'M').astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# Define model with dropout
model = Sequential([
    Dense(60, activation='relu', input_shape=(60,)),
    Dropout(0.2),
    Dense(30, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and evaluate
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy (with dropout): {test_acc:.4f}')

# Compare with model without dropout
model_no_dropout = Sequential([
    Dense(60, activation='relu', input_shape=(60,)),
    Dense(30, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_no_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_no_dropout.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
_, test_acc_no_dropout = model_no_dropout.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy (without dropout): {test_acc_no_dropout:.4f}')
