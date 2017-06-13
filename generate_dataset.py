import sklearn.datasets
import sklearn.preprocessing

def generate_dataset(n_train=1000,n_test=100,n_features=30,n_informative=5,noise=0):
    X,y,coef=sklearn.datasets.make_regression(n_samples=n_train+n_test,n_features=n_features,n_informative=n_informative,noise=noise,random_state=101,coef=True)
    y=y.reshape(-1,1)
    X_train=X[:n_train,:]
    X_test=X[n_train:,:]
    y_train=y[:n_train]
    y_test=y[n_train:]
    X_scaler=sklearn.preprocessing.StandardScaler()
    X_train=X_scaler.fit_transform(X_train)
    X_test=X_scaler.transform(X_test)
    y_scaler=sklearn.preprocessing.StandardScaler()
    y_train=y_scaler.fit_transform(y_train)
    y_test=y_scaler.transform(y_test)
    return X_train,X_test,y_train.ravel(),y_test.ravel(),coef

def get_minibatch(X,y,batch_size):
    from sklearn.utils import resample
    X,y=resample(X,y,random_state=101)
    n_cols=y.shape[0]
    for i in range(int(n_cols/batch_size)):
        yield (X[i*batch_size:(i+1)*batch_size,:],y[i*batch_size:(i+1)*batch_size])
    if n_cols%batch_size!=0:
        res=n_cols%batch_size
        yield(X[-res:,:],y[-res:])

def main():
    import sklearn.metrics
    import sklearn.linear_model
    import time
    regr=sklearn.linear_model.SGDRegressor()
    training_time_v=[]
    r2_score_v=[]
    X_train,X_test,y_train,y_test,coef=generate_dataset(noise=0.5)
    for X,y in get_minibatch(X_train,y_train,batch_size=50):
        tick1=time.time()
        regr.partial_fit(X,y)
        y_test_pred=regr.predict(X_test)
        tick2=time.time()
        r2_score_v.append(sklearn.metrics.r2_score(y_test,y_test_pred))
        training_time_v.append(tick2-tick1)

    return r2_score_v,training_time_v,regr





