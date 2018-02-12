def fibonacci_sequence(first,second,n):
    sequence = []
    sequence.append(first)
    sequence.append(second)
    if n<2:
        print("error")
        return

    for i in range(n-2):
        sequence.append(sequence[-1]+sequence[-2])
    return sequence

print(fibonacci_sequence(1,1,10))

def newton_method(f,df,x0,err):
    delta = abs(f(x0))
    while delta>err:
        x0 = x0 - f(x0)/df(x0)
        delta = abs(f(x0))
    return x0

def myfun(x):
    return x**5+x**3+1

def dmyfun(x):
    return 5*x**4+3*x**2

root = newton_method(myfun,dmyfun,1,1e-6)
print(root,myfun(root))

import numpy as np

class Matrix:
    """matrix class with Gaussian elimination"""
    def __init__(self,n):
        self.data = np.zeros([n,n+1])
        self.nRow = n

    def mycopy(self,arry):
        self.data = arry

    def GaussianElimination(self):
        for i in range(self.nRow-1):
            # find the max row
            maxEl = self.data[i,i]
            maxRow = i
            for j in range(i+1,self.nRow):
                if self.data[j,i]>maxEl:
                    maxEl = self.data[j,i]
                    maxRow = j
            # swap
            temp = np.copy(self.data[maxRow,:])
            self.data[maxRow,:] = self.data[i,:]
            self.data[i,:] = temp
            # self.showData()
            # elimination
            for j in range(i+1,self.nRow):
                c = self.data[j,i]/self.data[i,i]
                self.data[j,:] = self.data[j,:] - c*self.data[i,:]

                # if self.data[j,i]<1e-6:
                #     self.data[j,i]=0
                # else:
                #     print("error")
                #     break
    def showData(self):
        print(self.data)

    def sovleUpperTriangleMatrix(self):
        self.root = [0 for i in range(self.nRow)]
        for i in range(self.nRow-1,-1,-1):
            xi = self.data[i, self.nRow] / self.data[i, i]
            for j in range(self.nRow-1,-1,-1):
                self.data[j,self.nRow] -= xi * self.data[j,i]
                self.data[j, i] = 0
            self.root[i] = xi
        print(self.root)



myMatrix = Matrix(3)
myMatrix.mycopy(np.random.random([3,4]))
myMatrix.showData()
print("************")
myMatrix.GaussianElimination()
myMatrix.showData()
print("************")
myMatrix.sovleUpperTriangleMatrix()

import sys


def poly_value(poly, x):
    nOrder = len(poly) - 1
    value = 0
    for i in range(nOrder+1):
        value += poly[i] * x ** (nOrder - i)
    return value


def der_poly(poly):
    nOrder = len(poly) - 1
    dpoly = []
    for i in range(nOrder):
        dpoly.append(poly[i] * (nOrder - i))
    return dpoly


# def newton_method(poly, x0):
#     fx = poly_value(poly, x0)
#     dpoly = der_poly(poly)
#     df = poly_value(dpoly, x0)
#     # for i in range(10):
#     #     x0 = x0 - fx / df
#     #     if abs(poly_value(poly, x0)) < 1e-5:
#     #         break
#     #     newton_method(poly, x0)
#     if abs(poly_value(poly, x0)) < 1e-5:
#         return x0
#     else:
#         newton_method(poly,x0-fx/df)
def newton_method(poly,x0,err):
    fx = poly_value(poly, x0)
    dpoly = der_poly(poly)
    df = poly_value(dpoly,x0)
    delta = abs(fx)
    while delta>err:

        x0 = x0 - fx/df
        fx = poly_value(poly,x0)
        df = poly_value(der_poly(poly),x0)
        delta = abs(poly_value(poly,x0))
    return x0

print(newton_method([-30, 0, -4, 13, -7, -14],1,1e-5))
