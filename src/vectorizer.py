from scipy.sparse import hstack
import numpy as np
from sklearn.base import BaseEstimator

# Use for training data
def vectorize_data_fit(
        #title_vectorizer: BaseEstimator, 
        #desc_vectorizer: BaseEstimator,
        vectorizer: BaseEstimator, 
        #X_title_train: np.ndarray,
        X_train: np.ndarray) -> np.ndarray:

    #X_title_vect = title_vectorizer.fit_transform(X_title_train)
    #X_desc_vect = desc_vectorizer.fit_transform(X_desc_train)
    return vectorizer.fit_transform(X_train)

# Use for testing data
def vectorize_data(
        #title_vectorizer: BaseEstimator, 
        #desc_vectorizer: BaseEstimator,
        vectorizer: BaseEstimator, 
        #X_title_test: np.ndarray,
        X_test: np.ndarray) -> np.ndarray:
    
    #X_title_vect = title_vectorizer.transform(X_title_test)
    #X_desc_vect = desc_vectorizer.transform(X_desc_test)
    return vectorizer.transform(X_test)