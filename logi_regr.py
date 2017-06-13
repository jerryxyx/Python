from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
X,y=make_classification(n_sample=100,n_features=2,n_informatic=2,n_redundant=0,
        n_repeat=0,n_clusters_per_class=1,class_sep=2,random_state=101)
X_train,X_test,y_train,y_test=train_test_split(X,y.astype(float),
        test_size=0.33,random_state=101)
regr=LinearRegression()
regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)
def sigmoid(x):
    return 1/(np.exp(-x)+1)
y_possibility=sigmoid(y_pred)
y_pred2=[y_pred>0.5]
y_pred3=[y_possibility>0.5]
clf=LogisticRegression()
clf.fit(X_train,y_train)
y_pred4=clf.predict(y_test)
x_min,x_max=X[:,0].min(),X[:,0].max()
y_min,y_max=X[:,1].min(),X[:,1].max()
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
X_mesh=np.c_[xx.ravel(),yy.ravel()]
Z=clf.predict(X_mesh)
ZZ=Z.reshape((xx.shape[0],xx.shape[1]))
plt.pcolormesh(xx,yy,ZZ)
plt.show()

