import numpy as np

class LogisticRegressionSMLH:
    def __init__(self, learning_rate : float = 0.05, max_iter : int = 100000) -> None:
        
        """The constructor of the Logistic Regression
        model.
        :param learning_rate: float, default=0.05
        The learning rate of the model.
        :param max_iter: int, default = 100000
        The number of iterations to go through
        ."""
        
        # Setting up the hyperparameters.
        self.__learning_rate = learning_rate
        self.__max_iter = max_iter

    def sigmoid(self, y : np.array) -> np.array:
        """
        The sigmoid function.
        :param y: np.array
        The predictions of the linear function.
        """
        return 1 / (1 + np.exp(-y))
    
    def fit(self, X : np.array, y : np.array):
        
        """
        The fit function of the model.
        :param X: 2-d np.array
        The matrix with the features.
        :param y: 1-d np.array
        The target vector.
        """

        # Creatting the weights vector,
        self.coef_ = np.zeros(len(X[0])+1)

        # Adding the intercept column.
        X = np.hstack((X, np.ones((len(X), 1))))

        # The weights updating process.
        for i in range(self.__max_iter):
            # Prediction.
            pred = self.sigmoid(np.dot(X, self.coef_))

            # Computing the gradient.
            gradient = np.dot(X.T, (pred - y)) / y.size

            # Updating the weights.
            self.coef_ -= gradient * self.__learning_rate

        return self
    
    def predict_proba(self, X : np.array) -> np.array:
        """    
        This function returns the class probabilities.
        :param X: 2-d np.array
        The features matrix.
        :return: 2-d, np.array
        The array with the probabilities for
        every class
        for every sample.
        """

        # Adding the intercept column.
        X = np.hstack((X, np.ones((len(X), 1))))

        # Computing the probabilities.
        prob = self.sigmoid(np.dot(X, self.coef_))
        
        # Returning the probabilities.
        return np.hstack(((1 - prob).reshape(-1, 1), prob.reshape(-1, 1)))
    
    def predict(self, X : np.array) -> np.array:
        """
        This function returns the predictions of the
        model.
        :param X: 2-d np.array
        The features matrix.
        :return: 2-d, np.array
        The array with the probabilities for
        every class
        for every sample.
        """

        # Adding the intercept column.
        X = np.hstack((X, np.ones((len(X), 1))))
        return (self.sigmoid(np.dot(X, self.coef_)) > 0.5)* 1