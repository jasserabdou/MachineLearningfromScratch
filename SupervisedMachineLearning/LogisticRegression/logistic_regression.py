import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent.

    Attributes:
        learning_rate (float): Learning rate for gradient descent.
        n_iters (int): Number of iterations for gradient descent.
        weights (ndarray): Coefficients for logistic regression.
        bias (float): Intercept for logistic regression.
    """

    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initialize the LogisticRegression model.

        Args:
            learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.001.
            n_iters (int, optional): Number of iterations for gradient descent. Defaults to 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the logistic regression model to the given training data.

        Args:
            X (ndarray): Training data features.
            y (ndarray): Target values.

        Raises:
            ValueError: If the shape of X and y do not match.

        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
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
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x (ndarray): Input values.

        Returns:
            ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def mse(self, y_true, y_predicted):
        """
        Calculate the Mean Squared Error (MSE) between true and predicted values.

        Args:
            y_true (ndarray): True target values.
            y_predicted (ndarray): Predicted target values.

        Returns:
            float: Mean Squared Error.
        """
        return np.mean(np.power((y_true - y_predicted), 2))

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
        return corr**2

    def accuracy(y_true, y_pred):
        """
        Calculate the accuracy between true and predicted values.

        Args:
            y_true (ndarray): True target values.
            y_pred (ndarray): Predicted target values.

        Returns:
            float: Accuracy value.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
