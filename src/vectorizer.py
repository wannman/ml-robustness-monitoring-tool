
def vectorize_data(vectorizer, X_train, X_test, fit=True):
    
    if fit:
        X_train_vect = vectorizer.fit_transform(X_train)
        X_test_vect = vectorizer.transform(X_test)

    else:
        X_train_vect = vectorizer.transform(X_train)
        X_test_vect = vectorizer.transform(X_test)
    
    return X_train_vect, X_test_vect

