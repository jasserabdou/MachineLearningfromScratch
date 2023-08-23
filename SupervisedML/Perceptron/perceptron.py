import numpy as np


class Perceptron:
    """
    A simple implementation of the Perceptron algorithm for binary classification.

    Parameters:
    - lr (float): Learning rate for weight updates. Default is 0.01.
    - n_iters (int): Number of iterations for training. Default is 1000.

    Attributes:
    - lr (float): Learning rate for weight updates.
    - n_iters (int): Number of iterations for training.
    - activation (function): Activation function (unit step function).
    - weights (ndarray): Weights for the perceptron's features.
    - bias (float): Bias term for the perceptron.

    Methods:
    - fit(X, y): Fit the perceptron model to the given training data.
    - predict(X): Predict binary class labels for input data.
    - unit_step(z): Unit step activation function.
    - accuracy(y_true, y_pred): Calculate accuracy of predictions.
    """

    def __init__(self, lr=0.01, n_iters=1000):
        """
        Initialize the Perceptron.

        Parameters:
        - lr (float): Learning rate for weight updates. Default is 0.01.
        - n_iters (int): Number of iterations for training. Default is 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.activation = self.unit_step
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the perceptron model to the given training data.

        Parameters:
        - X (ndarray): Training data with shape (n_samples, n_features).
        - y (ndarray): Target labels with shape (n_samples,).

        Returns:
        None
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.unit_step(linear_output)
                update = self.lr * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predict binary class labels for input data.

        Parameters:
        - X (ndarray): Input data with shape (n_samples, n_features).

        Returns:
        - y_pred (ndarray): Predicted class labels with shape (n_samples,).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.unit_step(linear_output)
        return y_pred

    def unit_step(self, z):
        """
        Unit step activation function.

        Parameters:
        - z (float or ndarray): Input value(s).

        Returns:
        - output (int or ndarray): Output value(s) after applying the unit step function.
        """
        return np.where(z >= 0, 1, 0)

    def accuracy(self, y_true, y_pred):
        """
        Calculate accuracy of predictions.

        Parameters:
        - y_true (ndarray): True class labels with shape (n_samples,).
        - y_pred (ndarray): Predicted class labels with shape (n_samples,).

        Returns:
        - accuracy (float): Accuracy of the predictions.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
