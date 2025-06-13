import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Generating artificial data...")
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

print(f"Generated {len(x_values)} pairs of data (x, y).")
print(f"Training set: {len(x_train)} examples.")
print(f"Test set: {len(x_test)} examples.")

print("Building model for regression...")
model_regression = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)), 
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
print("Regression model built.")
model_regression.summary()

print("Model compilation...")
model_regression.compile(optimizer='adam',
                         loss='mean_squared_error',
                         metrics=['mean_absolute_error'])
print("Model compiled.")

print("Model training...")
history_regression = model_regression.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
print("Model training completed.")

print("Model evaluation on test data...")
loss_regression, mae_regression = model_regression.evaluate(x_test, y_test, verbose=0)
print(f"Mean squared error (MSE) on test: {loss_regression:.4f}")
print(f"Mean absolute error (MAE) on test: {mae_regression:.4f}")

print("Model predictions and visualization...")

x_for_prediction = np.linspace(min(x_values), max(x_values), 200).reshape(-1, 1)

y_predicted = model_regression.predict(x_for_prediction)

plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, label='Original data (with noise)', alpha=0.5)
plt.plot(x_for_prediction, y_predicted, color='red', linewidth=3, label='Model prediction')
plt.title("Regression: Original data vs Model prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()