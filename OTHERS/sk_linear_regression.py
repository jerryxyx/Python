from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_boston
import pandas as pd
boston=load_boston()
dataset=pd.DataFrame(boston.data,columns=boston.feature_names)
dataset['target']=boston.target
X=dataset.ix[:,:-1]
y=dataset['target'].values
linear_regression=linear_model.LinearRegression(normalize=False,fit_intercept=
        True)
Stand_coef_linear_reg=make_pipeline(StandardScaler(),linear_regression)
Stand_coef_linear_reg.fit(X,y)
for coef,var in sorted(zip(map(abs,Stand_coef_linear_reg.steps[1][1].coef_),dataset.columns[:-1]),reverse=True):
    print('%6.3f %s' % (coef,var))
