from sklearn import decomposition, datasets, preprocessing
import numpy as np
from random import sample, seed
import pandas as pd
boston = datasets.load_boston()
dataset = pd.DataFrame(boston.data,columns=boston.feature_names)
X = dataset
y = boston.target

#Decomposition
standardization = preprocessing.StandardScaler()
Xs = standardization.fit_transform(X)
pca = decomposition.PCA()
C = pca.fit_transform(Xs)
explained_variance = pca.explained_variance_ratio_

# Missing Data
Xm = X.copy()
seed(19)
missing = sample(range(len(y)),len(y)//4)
Xm.ix[missing,5] = np.nan
