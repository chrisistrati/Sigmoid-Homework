import numpy as np

class LogisticRegressionSMLH:
    def __init__(self, learning_rate: float = 0.05, max_iter: int = 100000) -> None:
        """
        The constructor of the Logistic Regression model.
        :param learning_rate: float, default=0.05
        :param max_iter: int, default=100000
        """
        # Setting up the hyperparameters.
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter

    def sigmoid(self, y: np.array) -> np.array:
        """The sigmoid function."""
        return 1 / (1 + np.exp(-y))

    def fit(self, X: np.array, y: np.array):
        """Fit the model."""
        # Ensure X is 2-dimensional
        X = X if X.ndim == 2 else X.reshape(-1, 1)
        self.coef_ = np.zeros(X.shape[1] + 1)

        # Add the intercept column
        X = np.hstack((X, np.ones((len(X), 1))))

        # Weights updating process
        for i in range(self.__max_iter):
            # Prediction
            pred = self.sigmoid(np.dot(X, self.coef_))

            # Compute the gradient
            gradient = np.dot(X.T, (pred - y)) / y.size

            # Update the weights
            self.coef_ -= gradient * self.__learning_rate

        return self

    def predict_proba(self, X: np.array) -> np.array:
        """Return class probabilities."""
        # Ensure X is 2-dimensional
        X = X if X.ndim == 2 else X.reshape(-1, 1)
        X = np.hstack((X, np.ones((len(X), 1))))

        # Compute probabilities
        prob = self.sigmoid(np.dot(X, self.coef_))
        return np.hstack(((1 - prob).reshape(-1, 1), prob.reshape(-1, 1)))

    def predict(self, X: np.array) -> np.array:
        """Return predictions."""
        # Ensure X is 2-dimensional
        X = X if X.ndim == 2 else X.reshape(-1, 1)
        X = np.hstack((X, np.ones((len(X), 1))))

        return (self.sigmoid(np.dot(X, self.coef_)) > 0.5).astype(int)