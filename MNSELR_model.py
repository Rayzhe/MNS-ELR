import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from MNSELR_APGM import L21L1

class Class_L21L1(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda1=1.55, lambda21=2.55):
        # Initialize lambda parameters
        self.lambda1 = lambda1
        self.lambda21 = lambda21
        self.W = None
        self.b = None

    def fit(self, X, Y):
        """
        Fit the L21L1 model.
        X: shape = (n_trials, n_channels, n_channels)
        Y: shape = (n_trials,)
        """
        # Transpose to shape (n_channels, n_channels, n_trials) for consistency
        X = X.transpose((1, 2, 0))
        self.W, self.b = L21L1(X, Y, self.lambda1, self.lambda21)
        self.classes_ = np.unique(Y)
        return self

    def decision_function(self, Rstest):
        """
        Compute the raw decision scores: f(x) = trace(W @ R) + b
        Rstest: shape = (n_trials, n_channels, n_channels)
        Returns:
            scores: shape = (n_trials,)
        """
        n = Rstest.shape[0]
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = np.trace(self.W @ Rstest[i]) + self.b
        return scores

    def predict_proba(self, Rstest):
        """
        For binary classification, convert the decision_function output to class probabilities
        using the sigmoid function. Returns probabilities for both negative and positive classes.
        Returns:
            probs: shape = (n_trials, 2), columns correspond to [p(neg), p(pos)]
        """
        scores = self.decision_function(Rstest)
        p_pos = 1 / (1 + np.exp(-scores))    # Probability of positive class
        p_neg = 1 - p_pos                    # Probability of negative class
        probs = np.vstack([p_neg, p_pos]).T  # shape = (n_trials, 2)
        return probs

    def predict(self, Rstest):
        """
        Predict labels using learned W and b.
        Returns:
            y_pred: shape = (n_trials,)
        """
        n = Rstest.shape[0]
        y_pred = np.zeros(n)
        for i in range(n):
            y_pred[i] = np.sign(np.trace(self.W @ Rstest[i]) + self.b)
        return y_pred

    def score(self, Rstest, y_true):
        """
        Compute accuracy score.
        """
        y_pred = self.predict(Rstest)
        return accuracy_score(y_true, y_pred)
