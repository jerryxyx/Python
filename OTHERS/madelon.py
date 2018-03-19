import numpy as np
import urllib, urllib.request

train_data='https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
validation_data='https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
train_response='https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
validation_response='https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
Xt=np.loadtxt(urllib.request.urlopen(train_data))
yt=np.loadtxt(urllib.request.urlopen(train_response))
Xv=np.loadtxt(urllib.request.urlopen(validation_data))
yv=np.loadtxt(urllib.request.urlopen(validation_response))
