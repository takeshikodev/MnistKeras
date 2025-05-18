import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Загрузка данных Fashion MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print("Данные Fashion MNIST загружены.")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Предобработка данных...")
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=10)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)
print("Предобработка завершена.")

print("Построение модели CNN для Fashion MNIST...")
model_cnn_fashion = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
print("Модель CNN построена.")
model_cnn_fashion.summary()

print("Компиляция модели CNN...")
model_cnn_fashion.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
print("Модель CNN скомпилирована.")

print("Обучение модели CNN для Fashion MNIST...")
history_fashion = model_cnn_fashion.fit(x_train, y_train_one_hot, epochs=5, batch_size=32, validation_data=(x_test, y_test_one_hot))
print("Обучение завершено.")

print("Оценка модели CNN на тестовых данных Fashion MNIST...")
loss_fashion, accuracy_fashion = model_cnn_fashion.evaluate(x_test, y_test_one_hot, verbose=0)
print(f"Потери на тестовых данных Fashion MNIST: {loss_fashion:.4f}")
print(f"Точность на тестовых данных Fashion MNIST: {accuracy_fashion:.4f}")

print("Предсказания модели CNN для первых 5 тестовых изображений Fashion MNIST...")
predictions_fashion = model_cnn_fashion.predict(x_test[:5])

print("Предсказания CNN для Fashion MNIST:")
for i in range(5):
    predicted_index = np.argmax(predictions_fashion[i])
    true_index = y_test[i]
    print(f"Изображение {i}: Предсказано {predicted_index} ({class_names[predicted_index]}), Настоящая {true_index} ({class_names[true_index]})")

fig, axes = plt.subplots(1, 5, figsize=(12, 4))
for i in range(5):
    axes[i].imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    predicted_index = np.argmax(predictions_fashion[i])
    true_index = y_test[i]
    axes[i].set_title(f"Pred: {class_names[predicted_index]}\nTrue: {class_names[true_index]}", fontsize=8)
    axes[i].axis('off')
plt.show()