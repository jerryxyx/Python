import numpy as np
import matplotlib.pyplot as plt

def payoff_call(s,x):
    return (s-x+abs(s-x))/2
s=np.arange(30,70,5)
x=45
call=2.5
profit=payoff_call(s,x)-call
y2=np.zeros(len(s))
x3=[x,x]
y3=[-30,50]
plt.plot(s,y2,'--')
plt.plot(s,profit)
plt.plot(s,-profit)
plt.plot(x3,y3)
plt.annotate('Call option for buyers',xy=(55,15),xytext=(35,20),arrowprops=
        dict(facecolor='blue',shrink=0.01))
plt.annotate('Call option for sellers',xy=(55,-10),xytext=(40,-20),arrowprops=
        dict(facecolor='red',shrink=0.01))
plt.annotate('Exercise price',xy=(45,-30),xytext=(50,-20),arrowprops=
        dict(facecolor='black',shrink=0.01))
plt.ylim(-30,50)
plt.title('Profit/Loss function')
plt.xlabel('Stock price')
plt.ylabel('Profit/Loss')

