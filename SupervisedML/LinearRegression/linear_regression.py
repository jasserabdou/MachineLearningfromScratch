import numpy as np


class LinearRegression:
    """
    A simple linear regression implementation using gradient descent.

    Attributes:
        lr (float): Learning rate for gradient descent.
        n_iters (int): Number of iterations for gradient descent.
        weights (ndarray): Coefficients for linear regression.
        bias (float): Intercept for linear regression.

    Methods:
        - __init__(self, lr=0.001, n_iters=1000): Initialize the LinearRegression model.
        - fit(X, y): Fit the linear regression model to the given training data.
        - predict(X): Predict target values using the trained model.
        - mse(y_true, y_predicted): Calculate the Mean Squared Error (MSE) between true and predicted values.
        - r2_score(y_true, y_predicted): Calculate the coefficient of determination (R-squared) between true and predicted values.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize the LinearRegression model.

        Args:
            lr (float, optional): Learning rate for gradient descent. Defaults to 0.001.
            n_iters (int, optional): Number of iterations for gradient descent. Defaults to 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the linear regression model to the given training data.

        Args:
            X (ndarray): Training data features.
            y (ndarray): Target values.
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict target values using the trained model.

        Args:
            X (ndarray): Input data features.

        Returns:
            ndarray: Predicted target values.
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    def mse(self, y_true, y_predicted):
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.

        Args:
            y_true (ndarray): True target values.
            y_predicted (ndarray): Predicted target values.

        Returns:
            float: Mean Squared Error.
        """
        return np.mean((y_true - y_predicted) ** 2)

    def r2_score(self, y_true, y_predicted):
        """
        Calculate the coefficient of determination (R-squared) between true and predicted values.

        Args:
            y_true (ndarray): True target values.
            y_predicted (ndarray): Predicted target values.

        Returns:
            float: R-squared value.
        """
        corr_matrix = np.corrcoef(y_true, y_predicted)
        corr = corr_matrix[0, 1]
        return np.power(corr, 2)
