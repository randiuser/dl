import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model(use_batchnorm=False, use_dropout=False):
    model = Sequential([Flatten(input_shape=(28, 28))])
    
    model.add(Dense(128, activation='relu'))
    if use_batchnorm:
        model.add(BatchNormalization())
    if use_dropout:
        model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu'))
    if use_batchnorm:
        model.add(BatchNormalization())
    if use_dropout:
        model.add(Dropout(0.3))
    
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models
models = {
    'Base': create_model(),
    'BatchNorm': create_model(use_batchnorm=True),
    'Dropout': create_model(use_dropout=True),
    'BatchNorm+Dropout': create_model(use_batchnorm=True, use_dropout=True)
}

for name, model in models.items():
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'{name} - Test accuracy: {test_acc:.4f}, Val accuracy: {history.history["val_accuracy"][-1]:.4f}')
