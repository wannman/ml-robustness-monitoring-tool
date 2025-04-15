from scipy.sparse import hstack
import numpy as np
from sklearn.base import BaseEstimator

# Use for training data
def vectorize_data_fit(
        title_vectorizer: BaseEstimator, 
        desc_vectorizer: BaseEstimator, 
        X_title_train: np.ndarray,
        X_desc_train: np.ndarray) -> np.ndarray:

    X_title_vect = title_vectorizer.fit_transform(X_title_train)
    X_desc_vect = desc_vectorizer.fit_transform(X_desc_train)
    return hstack([X_title_vect, X_desc_vect])

# Use for testing data
def vectorize_data(
        title_vectorizer: BaseEstimator, 
        desc_vectorizer: BaseEstimator, 
        X_title_test: np.ndarray,
        X_desc_test: np.ndarray) -> np.ndarray:
    
    X_title_vect = title_vectorizer.transform(X_title_test)
    X_desc_vect = desc_vectorizer.transform(X_desc_test)
    return hstack([X_title_vect, X_desc_vect])