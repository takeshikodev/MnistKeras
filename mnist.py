import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Data loading MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Data loaded.")

print("Data preprocessing...")
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=10)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)
print("Data preprocessing completed.")

print("Model building...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
print("Model built.")
model.summary()

print("Model compilation...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Model compiled.")

print("Model training...")
history = model.fit(x_train, y_train_one_hot, epochs=5, batch_size=32, validation_data=(x_test, y_test_one_hot))
print("Model training completed.")

print("Model evaluation on test data...")
loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
print(f"Loss on test data: {loss:.4f}")
print(f"Accuracy on test data: {accuracy:.4f}")

print("Predictions for first 5 test images...")
predictions = model.predict(x_test[:5])

print("Predictions:")
for i in range(5):
    predicted_class = np.argmax(predictions[i])
    true_class = y_test[i]
    print(f"Image {i}: Predicted {predicted_class}, True {true_class}")

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    axes[i].imshow(x_test[i], cmap=plt.cm.binary)
    predicted_class = np.argmax(predictions[i])
    true_class = y_test[i]
    axes[i].set_title(f"Pred: {predicted_class}\nTrue: {true_class}")
    axes[i].axis('off')
plt.show()