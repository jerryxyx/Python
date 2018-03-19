import numpy as np
import sklearn.linear_model
import madelon as md
Xt=md.Xt
yt=md.yt
Xv=md.Xv
yv=md.yv

scaling_list=np.linspace(0.1,0.95,5)
selection_threshold_list=np.linspace(0,0.95,5)

scaling1,selection_threshold1=np.meshgrid(scaling_list,selection_threshold_list)
output=[]

for i in np.arange(len(scaling_list)):
    for j in np.arange(len(selection_threshold_list)):
        scaling = scaling1[i,j]
        selection_threshold = selection_threshold1[i,j]
        selector = sklearn.linear_model.RandomizedLogisticRegression(scaling=scaling,selection_threshold=selection_threshold)
        selector.fit(Xt,yt)
        n_nonzeros = np.sum(selector.scores_!=0)
        n_threshold = np.sum(selector.scores_>=selection_threshold)
        print('scaling:',scaling,'selection_threshold:',selection_threshold,'n_nonzeros:',n_nonzeros,'n_threshold:',n_threshold)
        output+='scaling:'+str(scaling)+'selection_threshold:'+str(selection_threshold)+'n_nonzeros:'+str(n_nonzeros)+'n_threshold:'+str(n_threshold)+'\n'

