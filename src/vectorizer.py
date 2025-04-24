import numpy as np
from sklearn.base import BaseEstimator

# Use for training data
def vectorize_data_fit(vectorizer: BaseEstimator,X_train: np.ndarray) -> np.ndarray:
    return vectorizer.fit_transform(X_train)

# Use for testing data
def vectorize_data(vectorizer: BaseEstimator, X_test: np.ndarray) -> np.ndarray:
    return vectorizer.transform(X_test)