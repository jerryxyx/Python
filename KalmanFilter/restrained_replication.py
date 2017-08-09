import matplotlib.pyplot as plt
import numpy as np
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
sufficient_posteriori_estimate_x=[]
sufficient_priori_estimate_x =[]
priori_covariance =[]
sufficient_priori_covariance=[]
posteriori_covariance=[]
sufficient_posteriori_covariance=[]

#in fact it is the priori estimation, but here is the posteriori one
y_ob = np.stack((y1_ob,y2_ob,y3_ob))
return_covariance = np.cov(y_ob)#weitht(beta) process_covariance
weights_covariance = np.eye(3)*0.01#how to measure the process covariance? or is it should be an input parameter of our model?
sufficient_weights_covariance = np.diag([0.01,0.01,0])
# print(return_covariance)
# print(sufficient_weights_covariance)

#initial guess
posteriori_estimate_x=[[0.33,0.33,0.33]]
sufficient_posteriori_estimate_x = [[0.33,0.33,1]]
posteriori_covariance.append([[1,1,1],[1,1,1],[1,1,1]])
sufficient_posteriori_covariance.append(np.diag([1,1,0]))


for i in range(100):
    #predict
    priori_estimate_x.append(posteriori_estimate_x[i])                                  #n*1
    sufficient_priori_estimate_x.append(sufficient_posteriori_estimate_x[i])            #[x[1],x[2],1]
    priori_covariance.append(posteriori_covariance[i]+weights_covariance)               #n*n
    sufficient_priori_covariance.append(sufficient_posteriori_covariance[i]+sufficient_weights_covariance)
    #update
    H = y_ob.T[i]#1*n
    sufficient_H = [h-H[-1] for h in H[:-1]]
    sufficient_H.append(H[-1])# sufficient_H=[H[1]-H[3],H[2]-H[3],H[3]]
    sufficient_H= np.asarray(sufficient_H)

    measurement_residual = index_[i]-np.matmul(H,np.asarray(priori_estimate_x[i]).T)    #scalar
    sufficient_measurement_residual = index_[i]-np.matmul(sufficient_H,np.asarray(sufficient_priori_estimate_x[i]).T)
    innovation_covariance = np.matmul(np.matmul(H,np.asarray(priori_covariance[i])),H.T)#scalar
    sufficient_innovation_covariance = np.matmul(np.matmul(sufficient_H,np.asarray(sufficient_priori_covariance[i])),sufficient_H.T)#scalar
    # print("h")
    # print(H.T)
    # print("innovation_covariance")
    # print(innovation_covariance)
    # print("priori_covariance")
    # print(priori_covariance[i])
    # print("end")
    kalman_gain = np.matmul(priori_covariance[i],H.T)/innovation_covariance             #n*1
    sufficient_kalman_gain = np.matmul(sufficient_priori_covariance[i],sufficient_H.T)/sufficient_innovation_covariance
    # print("kalman gain")
    # print(kalman_gain)
    posteriori_estimate_x.append(priori_estimate_x[i]+kalman_gain*measurement_residual) #n*1
    sufficient_posteriori_estimate_x.append(sufficient_priori_estimate_x[i]+sufficient_kalman_gain*sufficient_measurement_residual)
    posteriori_covariance.append(np.matmul(np.eye(3)-np.matmul(kalman_gain,H),priori_covariance[i]))#n*n
    sufficient_posteriori_covariance.append(np.matmul(np.eye(3)-np.matmul(sufficient_kalman_gain,H),sufficient_priori_covariance[i]))
    print("estimated accuracy of the state estimate")
    print(posteriori_covariance[i])
    print(sufficient_posteriori_covariance[i])

estimate_index =[]
sufficient_estimate_index = []
for i in range(100):

    H = y_ob.T[i]  # 1*n
    sufficient_H = [h - H[-1] for h in H[:-1]]
    sufficient_H.append(H[-1])
    sufficient_H=np.asarray(sufficient_H)

    estimate_index.append(np.matmul(np.asarray(posteriori_estimate_x[i]),H.T))
    sufficient_estimate_index.append(np.matmul(np.asarray(sufficient_posteriori_estimate_x[i]),sufficient_H.T))

plt.plot(x,estimate_index,label="estimate index")
plt.plot(x,sufficient_estimate_index)
plt.ylabel("index comparision")

plt.subplot(338)
plt.plot(x,posteriori_estimate_x[:-1])
plt.ylabel("weights")

plt.subplot(339)
H3 = [sufficient_posteriori_estimate_x[i][-1] for i in range(100)]
sufficient_underlying_weights=[np.asarray(sufficient_posteriori_estimate_x[i])-np.asarray([H3[i],H3[i],0]) for i in range(100)]
plt.plot(x,sufficient_posteriori_estimate_x[:-1])
plt.show()