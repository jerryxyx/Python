import pandas as pd
from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
boston=load_boston()
dataset=pd.DataFrame(boston.data,columns=boston.feature_names)
dataset['target']=boston.target
X=dataset.ix[:,:-1]
y=dataset['target'].values
varnames=dataset.columns[:-1]
def r2_est(X,y):
    linear_regression = linear_model.LinearRegression(fit_intercept=True)
    return r2_score(y_true=y,y_pred=linear_regression.fit(X,y).predict(X))
def delete_var(X,y,varnames):
    imp = []
    for i in range(X.shape[1]):
        selection=[j for j in range(X.shape[1]) if j!=i]
        imp.append((r2_est(X,y)-r2_est(X.ix[:,selection],y),varnames[i]))
    return imp
def st_reg(X,y,varnames):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    linear_regression=linear_model.LinearRegression(fit_intercept=True)
    st_regression = make_pipeline(StandardScaler(),linear_regression)
    coef = st_regression.fit(X,y).steps[1][1].coef_
    intercept = st_regression.fit(X,y).steps[1][1].intercept_
    coef_list =[]
    for i,varname in enumerate(varnames):
        coef_list.append((coef[i],varname))
    coef_list.append((intercept,'intercept'))
    return coef_list

def interaction_reg(X,y,varnames):
    from sklearn.preprocessing import PolynomialFeatures
    create_interactions = PolynomialFeatures(degree=2,interaction_only=True,
            include_bias=Falte)
    Xi=create_interactions.fit_transform(X)
    main_effects=create_interactions.n_input_features_
    baseline=r2_est(X,y)
    increments=[]
    for k,effect in enumerate(create_interactions.powers_[main_effects:]):
        termA,termB = varnames[effect==1]
        Xii=Xi[:,list(range(main_effects))+[main_effects+k]]
        increment=r2_est(Xii,y)-baseline
        if(increment>0.01):
            increments.append((increment,termA,termB))
    return increments
