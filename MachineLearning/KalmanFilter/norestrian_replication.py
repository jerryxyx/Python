import matplotlib.pyplot as plt
import numpy as np
from pykalman import *
import scipy as sp
import scipy.optimize

def f1(x):
    return np.sin(x)+np.log(x+0.5)

def f2(x):
    return np.cos(x)+1/(x+0.5)

def f3(x):
    return np.exp(0.05*x)

def index_fun(w):
    return [f1(i/10)*w[i]+f2(i/10)*(1-w[i]) for i in range(100)]

w1 = np.random.rand(100)
w2 = np.random.rand(100)
w3 = np.random.rand(100)

x = [i/10 for i in range(100)]
weights = [0.5+0.5*np.sin(i/10) for i in range(50)]
weights = weights +[0.5 for  i in range(20)]
weights = weights +[0.25 for  i in range(15)]
weights = weights +[0.4 for i in range(15)]
weights1 = weights
weights2 = [0.2 for i in range(20)]+[0.3 for i in range(20)] + [np.log(1+i/10) for i in range(30)] + \
           [0.5+ 0.5*np.cos(i/10) for i in range(30)]
#actually the weight refers to the funding not the ratio of shares
weights3 = [1-weights1[i]-weights2[i] for i in range(100)]

#weights123 redefinition
weights1 = [0.2 for i in range(100)]
weights2 = [0.3 for i in range(100)]
weights3 = [0.5 for i in range(100)]

index = index_fun(weights)
plt.subplot(331)
y1 = [f1(xi) for xi in x]
y1_ob = y1 + w1
y1_ob = np.asarray(y1_ob)
plt.ylabel("toy return1")
plt.plot(x,y1_ob)

plt.subplot(332)
y2 = [f2(xi) for xi in x]
y2_ob = y2 + w2
y2_ob = np.asarray(y2_ob)
plt.ylabel("toy return2")
plt.plot(x,y2_ob)

plt.subplot(333)
y3 = [f3(xi) for xi in x]
y3_ob = y3 + w3
y3_ob = np.asarray(y3_ob)
plt.ylabel("toy return3")
plt.plot(x,y3_ob)

plt.subplot(334)
plt.plot(x,weights1)
plt.ylabel("weight1")
plt.subplot(335)
plt.plot(x,weights2)
plt.ylabel("weight2")
plt.subplot(336)
plt.plot(x,weights3)
plt.ylabel("weight3")

plt.subplot(337)
index_ = [y1_ob[i]*weights1[i]+y2_ob[i]*weights2[i]+y3_ob[i]*weights3[i] for i in range(100)]
plt.scatter(x,index_,marker="+",label="true index")





posteriori_estimate_x=[]
priori_estimate_x=[]
priori_covariance =[]
posteriori_covariance=[]

y_ob = np.stack((y1_ob,y2_ob,y3_ob))
return_covariance = np.cov(y_ob)#return_covariance(no use here as the process is actually the weights vector)
process_covariance = np.eye(3)*1e-5#We guess the process_covariance here
print(return_covariance)

#initial guess
posteriori_estimate_x=[[0,0,0]]
posteriori_covariance.append([[1,1,1],[1,1,1],[1,1,1]])

kf = KalmanFilter(transition_matrices=np.eye(3),observation_matrices=y_ob.T[:,np.newaxis],
                  transition_covariance=np.eye(3)*0.1,observation_covariance=0,
                  initial_state_mean=np.zeros(3),initial_state_covariance=np.eye(3),
                  n_dim_state=3,n_dim_obs=1)
state_means, state_covs = kf.filter(index_)
print(state_means)

# for i in range(100):
#     #predict
#     priori_estimate_x.append(posteriori_estimate_x[i])                                  #n*1
#     priori_covariance.append(posteriori_covariance[i]+process_covariance)               #n*n
#     #update
#     H = y_ob.T[i]                                                                       #1*n
#     measurement_residual = index_[i]-np.matmul(H,np.asarray(priori_estimate_x[i]).T)    #scalar
#     innovation_covariance = np.matmul(np.matmul(H,np.asarray(priori_covariance[i])),H.T)#scalar
#     # print("h")
#     # print(H.T)
#     # print("innovation_covariance")
#     # print(innovation_covariance)
#     # print("priori_covariance")
#     # print(priori_covariance[i])
#     # print("end")
#     kalman_gain = np.matmul(priori_covariance[i],H.T)/innovation_covariance             #n*1
#     # print("kalman gain")
#     # print(kalman_gain)
#     posteriori_estimate_x.append(priori_estimate_x[i]+kalman_gain*measurement_residual) #n*1
#     posteriori_covariance.append(np.matmul(np.eye(3)-np.matmul(kalman_gain,H),priori_covariance[i]))#n*n

estimate_index =[]
for i in range(100):
    estimate_index.append(np.matmul(np.asarray(state_means[i]),y_ob.T[i].T))

plt.plot(x,estimate_index,label="estimate index")
plt.ylabel("index comparision")

plt.subplot(338)
plt.plot(x,state_means)
plt.ylabel("weights")
plt.show()