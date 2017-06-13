from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_regression
import numpy as np
import sklearn

def generate_dataset(n_train,n_test,n_features,noise=0.1):
    X,y = make_regression(n_samples=int(n_train+n_test),
                          n_features=int(n_features),
                          noise=noise,random_state=101)
    X_train=X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    X_scaler = sklearn.preprocessing.StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return X_train,X_test,y_train,y_test
