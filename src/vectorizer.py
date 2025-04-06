from sklearn.base import BaseEstimator

def vectorize_data(vectorizer: BaseEstimator,  X_train: list, X_test:list, fit: bool=True) -> tuple:
    
    if fit:
        X_train_vect = vectorizer.fit_transform(X_train)
        X_test_vect = vectorizer.transform(X_test)

    else:
        X_train_vect = vectorizer.transform(X_train)
        X_test_vect = vectorizer.transform(X_test)
    
    return X_train_vect, X_test_vect

