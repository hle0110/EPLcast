import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class ScaledLogisticModel(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=1000)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
