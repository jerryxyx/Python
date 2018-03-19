import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import load_boston

boston =load_boston()
dataset=pd.DataFrame(data=boston.data,columns=boston.feature_names)
dataset['target']=boston.target

def visualize_correlation_matrix(data,hurdle=0.0):
    X = data.ix[:,:-1]
    R=np.corrcoef(X,rowvar=0)
    R[np.where(np.abs(R)<hurdle)]=0
    heatmap=plt.pcolor(R,cmap=mpl.cm.coolwarm,alpha=0.8)
    heatmap.axes.set_xticks(np.arange(R.shape[1])+0.5,minor=False)
    heatmap.axes.set_xticklabels(dataset.columns[:-1],rotation=30)
    heatmap.axes.set_yticks(np.arange(R.shape[0])+0.5,minor=False)
    heatmap.axes.set_yticklabels(data.columns[:-1],minor=False)
    heatmap.axes.set_frame_on(False)
    plt.colorbar()
    plt.tick_params(axis='both',which='both',bottom='off',top='on')
    plt.show()

visualize_correlation_matrix(dataset,0.5)
