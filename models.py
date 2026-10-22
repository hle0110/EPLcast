import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class ScaledLogisticModel(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=2000)

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


DRAW_LABEL = 'draw'
DRAW_FACTOR_GRID = np.round(np.arange(0.70, 1.3001, 0.01), 2)


def adjust_draws(proba, draw_index, factor):
    adjusted = np.array(proba, dtype=float, copy=True)
    adjusted[:, draw_index] *= factor
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def fit_draw_factor(proba, codes, draw_index, grid=DRAW_FACTOR_GRID):
    rows = np.arange(len(codes))
    losses = [-np.log(np.clip(adjust_draws(proba, draw_index, f)[rows, codes], 1e-15, 1.0)).mean() for f in grid]
    return float(grid[int(np.argmin(losses))])


class DrawCalibratedModel:
    def __init__(self, model, draw_factor=1.0):
        self.model = model
        self.draw_factor = draw_factor
        self.classes_ = model.classes_

    def predict_proba(self, X):
        draw_index = list(self.classes_).index(DRAW_LABEL)
        return adjust_draws(self.model.predict_proba(X), draw_index, self.draw_factor)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
