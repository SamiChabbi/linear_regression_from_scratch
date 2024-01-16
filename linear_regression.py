#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

def visualize_dataset(x, y):
    """
    Visualize the dataset.

    Parameters:
    - x: Input features.
    - y: Target values.
    """
    plt.scatter(x, y)
    plt.title("Dataset Visualization")
    plt.xlabel("Input Features")
    plt.ylabel("Target Values")
    plt.show()

def initialize_model_parameters():
    """
    Initialize model parameters randomly.

    Returns:
    - theta: Randomly initialized parameters.

    Explanation:
    The purpose of this exercise is to find the right parameters 'a' and 'b'
    that are on our vector 'theta'. We initialize it randomly, and the machine
    will improve it gradually.
    """
    theta = np.random.rand(2, 1)
    return theta

def prepare_input_matrix(x):
    """
    Add a column of ones to the input matrix.

    Parameters:
    - x: Input features.

    Returns:
    - X: Input matrix with an added column of ones.

    Explanation:
    This function adds a column of ones to the input matrix 'X', allowing us
    to use the formula F = X * theta for our linear regression model, where
    'theta' is a vector containing our parameters 'a' and 'b'.
    """
    return np.hstack((x, np.ones(x.shape)))

def linear_regression_model(X, theta):
    """
    Compute predictions using the linear regression model.

    Parameters:
    - X: Input matrix.
    - theta: Model parameters.

    Returns:
    - Predicted values.
    """
    return X.dot(theta)

def mean_squared_error(X, y, theta):
    """
    Compute the mean squared error cost function.

    Parameters:
    - X: Input matrix.
    - y: Target values.
    - theta: Model parameters.

    Returns:
    - Cost value.

    Explanation:
    The cost function measures how far our predictions are from the actual
    target values. It is the mean of the squared differences.
    """
    m = len(y)
    return 1 / (2 * m) * np.sum((linear_regression_model(X, theta) - y) ** 2)

def gradient(X, y, theta):
    """
    Compute the gradient of the cost function with respect to parameters.

    Parameters:
    - X: Input matrix.
    - y: Target values.
    - theta: Model parameters.

    Returns:
    - Gradient vector.

    Explanation:
    The gradient is the derivative of the cost function with respect to 'a' and 'b'.
    It indicates the slope, showing the direction to improve our parameters.
    """
    m = len(y)
    return 1 / m * X.T.dot(linear_regression_model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
    """
    Update parameters using gradient descent.

    Parameters:
    - X: Input matrix.
    - y: Target values.
    - theta: Initial model parameters.
    - learning_rate: Learning rate for gradient descent.
    - n_iterations: Number of iterations for gradient descent.

    Returns:
    - Updated model parameters.
    """
    for _ in range(n_iterations):
        theta = theta - learning_rate * gradient(X, y, theta)
    return theta

def visualize_model(x, y, model, label, color):
    """
    Visualize the dataset along with the model.

    Parameters:
    - x: Input features.
    - y: Target values.
    - model: Function representing the model.
    - label: Label for the plot legend.
    - color: Color for the plot.

    Explanation:
    This function visualizes the dataset along with the model predictions.
    """
    plt.scatter(x, y, label="Dataset")
    plt.plot(x, model, label=label, color=color)
    plt.title(f"Model Visualization - {label}")
    plt.xlabel("Input Features")
    plt.ylabel("Target Values")
    plt.legend()
    plt.show()

def main():
    # Generating synthetic dataset
    x, y = make_regression(n_samples=100, n_features=1, noise=10)
    y = y.reshape(y.shape[0], 1)

    # Visualizing dataset
    visualize_dataset(x, y)

    # Initializing model parameters
    theta = initialize_model_parameters()

    # Preparing input matrix
    X = prepare_input_matrix(x)

    # Visualizing initial model
    initial_model = linear_regression_model(X, theta)
    visualize_model(x, y, initial_model, "Initial Model", "red")

    # Computing initial cost
    initial_cost = mean_squared_error(X, y, theta)
    print("Initial Cost:", initial_cost)

    # Running gradient descent
    final_theta = gradient_descent(X, y, theta, learning_rate=0.01, n_iterations=1000)

    # Visualizing final model
    final_model = linear_regression_model(X, final_theta)
    visualize_model(x, y, final_model, "Final Model", "green")

    # Computing final cost
    final_cost = mean_squared_error(X, y, final_theta)
    print("Final Cost:", final_cost)

if __name__ == "__main__":
    main()

