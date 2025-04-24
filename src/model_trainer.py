from sklearn.base import BaseEstimator
import numpy as np

def train_model(model: BaseEstimator, X: np.ndarray, y:np.ndarray) -> BaseEstimator:
    model.fit(X, y)
    return model