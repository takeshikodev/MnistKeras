# MnistKeras

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [License](#license)

## Overview

**MnistKeras** is a Python project demonstrating various neural network implementations using TensorFlow and Keras. The project includes models for image classification (MNIST digits and Fashion MNIST) and regression tasks, showcasing fundamental deep learning concepts and techniques.

### Target Audience
- Machine Learning Practitioners
- Deep Learning Students
- Data Scientists
- AI Researchers

## Features

### Image Classification
- **MNIST Digits**: Classification of handwritten digits (0-9)
- **Fashion MNIST**: Classification of clothing items across 10 categories
- **Model Evaluation**: Performance metrics and visualization of predictions

### Regression
- **Synthetic Data Generation**: Creating artificial data with quadratic relationship
- **Regression Model**: Neural network for predicting continuous values
- **Visualization**: Plotting of original data vs. model predictions

## Technologies

- **Python**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing library
- **Matplotlib**: Data visualization library

## Project Structure

```
MnistKeras/
├── mnist.py           # MNIST digits classification model
├── fashion.py         # Fashion MNIST classification model
├── regression.py      # Regression model with synthetic data
├── requirements.txt   # Project dependencies
├── LICENSE            # MIT License
└── README.md          # Project documentation
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/takeshikodev/MnistKeras.git
   cd MnistKeras
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run any of the provided scripts to train and evaluate the respective models:

### MNIST Digits Classification
```bash
python mnist.py
```

### Fashion MNIST Classification
```bash
python fashion.py
```

### Regression Model
```bash
python regression.py
```

## Models

### MNIST Digits Model
- **Architecture**: Simple feedforward neural network
- **Layers**: Flatten input → Dense(128, ReLU) → Dense(10, Softmax)
- **Training**: 5 epochs with Adam optimizer
- **Performance**: Evaluated on test set with accuracy metrics

### Fashion MNIST Model
- **Architecture**: Convolutional Neural Network (CNN)
- **Layers**: Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Performance**: Higher accuracy than simple feedforward networks

### Regression Model
- **Architecture**: Feedforward neural network
- **Layers**: Dense(64, ReLU) → Dense(64, ReLU) → Dense(1, Linear)
- **Training**: 50 epochs with Adam optimizer
- **Metrics**: Mean Squared Error (MSE) and Mean Absolute Error (MAE)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

Copyright (c) 2025 Takeshiko