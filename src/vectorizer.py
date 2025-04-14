import numpy as np
from sklearn.base import BaseEstimator

# Use for training data
def vectorize_data_fit(vectorizer: BaseEstimator, X_train: np.ndarray) -> np.ndarray:
    X_vect = vectorizer.fit_transform(X_train)
    return X_vect

# Use for testing data
def vectorize_data(vectorizer: BaseEstimator, X_test: np.ndarray) -> np.ndarray:
    X_vect = vectorizer.transform(X_test)
    return X_vect
