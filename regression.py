import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Генерация искусственных данных...")
np.random.seed(42) 
x_values = np.random.uniform(low=-10, high=10, size=1000)
y_values = x_values**2 + np.random.normal(loc=0, scale=5, size=1000)

indices = np.arange(len(x_values))
np.random.shuffle(indices)
x_values = x_values[indices]
y_values = y_values[indices]

split_index = int(0.8 * len(x_values))
x_train, x_test = x_values[:split_index], x_values[split_index:]
y_train, y_test = y_values[:split_index], y_values[split_index:]

print(f"Сгенерировано {len(x_values)} пар данных (x, y).")
print(f"Обучающая выборка: {len(x_train)} примеров.")
print(f"Тестовая выборка: {len(x_test)} примеров.")

print("Построение модели для регрессии...")
model_regression = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)), 
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
print("Модель регрессии построена.")
model_regression.summary()

print("Компиляция модели...")
model_regression.compile(optimizer='adam',
                         loss='mean_squared_error',
                         metrics=['mean_absolute_error'])
print("Модель скомпилирована.")

print("Обучение модели...")
history_regression = model_regression.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
print("Обучение завершено.")

print("Оценка модели на тестовых данных...")
loss_regression, mae_regression = model_regression.evaluate(x_test, y_test, verbose=0)
print(f"Среднеквадратичная ошибка (MSE) на тесте: {loss_regression:.4f}")
print(f"Средняя абсолютная ошибка (MAE) на тесте: {mae_regression:.4f}")

print("Предсказания модели и визуализация...")

x_for_prediction = np.linspace(min(x_values), max(x_values), 200).reshape(-1, 1)

y_predicted = model_regression.predict(x_for_prediction)

plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, label='Исходные данные (с шумом)', alpha=0.5)
plt.plot(x_for_prediction, y_predicted, color='red', linewidth=3, label='Предсказание модели')
plt.title("Регрессия: Исходные данные vs Предсказание нейросети")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()