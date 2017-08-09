import numpy as np
import pandas as pd

data = [['TI', 1.57e11, 15, .9], ['SE', 1.63e11, 22, 1], ['PS', 0.52e11, 40, .8], ['HI', 1.29e11, 30, 1.1]]
dataset = pd.DataFrame(data=data, columns=['Ticker', 'Cap', 'P/E', 'B/P'])
ep = 1 / dataset.ix[:, 'P/E']
dataset.ix[:, 'E/P'] = ep
print(dataset)


# 10.1
def exposure(v, c):
    cmean = 0
    for i in range(len(v)):
        cmean += v[i] * c[i]
    cmean /= np.sum(c)
    return (v - cmean) / np.std(v,ddof=1)


ep_exposure = exposure(dataset.ix[:, 'E/P'], dataset.ix[:, 'Cap'])
bp_exposure = exposure(dataset.ix[:, 'B/P'], dataset.ix[:, 'Cap'])
X = pd.concat([ep_exposure, bp_exposure], axis=1)
print("explosure matrix:")
print(X)

# 10.2
r = np.array([0.06, -.0067, -.0414, -.0428])
# using pseudo inverse to compute factor returns
# which is equalvalent to least square regression
f = np.matmul(np.linalg.pinv(X), r)
print("factor return vector:")
print(f)

epsilon = r - np.matmul(X, f)
print("idiosyncratic return vector:")
print(epsilon)
