import numpy as numpy
import scipy as scipy
from sklearn import cluster
import matplotlib.pyplot as plt
 
 
 
def set2List(NumpyArray):
    list_ = []
    for item in NumpyArray:
        list_.append(item.tolist())
    return list_
 
def GenerateData():
    x1=numpy.random.randn(50,2)
    x2x=numpy.random.randn(80,1)+12
    x2y=numpy.random.randn(80,1)
    x2=numpy.column_stack((x2x,x2y))
    x3=numpy.random.randn(100,2)+8
    x4=numpy.random.randn(120,2)+15
    z=numpy.concatenate((x1,x2,x3,x4))
    return z


class DBSCAN:
    def __init__(self,Epsilon,MinumumPoints,DistanceMethod = 'euclidean'):
        self.Epsilon = Epsilon
        self.MinumumPoints = MinumumPoints
        self.DistanceMethod = DistanceMethod
    
    def fit_predict(self,Dataset,query_index):
        Epsilon = self.Epsilon
        MinumumPoints = self.MinumumPoints
        DistanceMethod = self.DistanceMethod
    #    Dataset is a mxn matrix, m is number of item and n is the dimension of data
        m,n=Dataset.shape
        Visited=numpy.zeros(m,'int')
        Type=numpy.zeros(m)
    #   -1 noise, outlier
    #    0 border
    #    1 core
        ClustersList=[]
        Cluster=[]
        PointClusterNumber=numpy.zeros(m)
        PointClusterNumberIndex=1
        PointNeighbors=[]
        DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset, DistanceMethod))
        
        
        if Visited[query_index]==0:
            Visited[query_index]=1
            PointNeighbors=numpy.where(DistanceMatrix[query_index]<Epsilon)[0]
            print("point neighbors",PointNeighbors)
            if len(PointNeighbors)<MinumumPoints:
                Type[query_index]=-1
            else:
                for k in range(len(Cluster)):
                    Cluster.pop()
                Cluster.append(query_index)
                PointClusterNumber[query_index]=PointClusterNumberIndex


                PointNeighbors=set2List(PointNeighbors)    
                self.ExpandCluster(Dataset[query_index], PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  )
                Cluster.append(PointNeighbors[:])
                ClustersList.append(Cluster[:])
                PointClusterNumberIndex=PointClusterNumberIndex+1

#         print("Type",Type)
        return PointClusterNumber

    def cluster(self,Dataset):
        Epsilon = self.Epsilon
        MinumumPoints = self.MinumumPoints
        DistanceMethod = self.DistanceMethod
    #    Dataset is a mxn matrix, m is number of item and n is the dimension of data
        m,n=Dataset.shape
        Visited=numpy.zeros(m,'int')
        Type=numpy.zeros(m)
    #   -1 noise, outlier
    #    0 border
    #    1 core
        ClustersList=[]
        Cluster=[]
        PointClusterNumber=numpy.zeros(m)
        PointClusterNumberIndex=1
        PointNeighbors=[]
        DistanceMatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Dataset, DistanceMethod))
        
        for i in range(m):
            if Visited[i]==0:
                Visited[i]=1
                PointNeighbors=numpy.where(DistanceMatrix[i]<Epsilon)[0]
                print("point neighbors",PointNeighbors)
                if len(PointNeighbors)<MinumumPoints:
                    Type[i]=-1
                else:
                    for k in range(len(Cluster)):
                        Cluster.pop()
                    Cluster.append(i)
                    PointClusterNumber[i]=PointClusterNumberIndex


                    PointNeighbors=set2List(PointNeighbors)    
                    self.ExpandCluster(Dataset[i], PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  )
                    Cluster.append(PointNeighbors[:])
                    ClustersList.append(Cluster[:])
                    PointClusterNumberIndex=PointClusterNumberIndex+1

    #         print("Type",Type)
        return PointClusterNumber 



    def ExpandCluster(self,PointToExapnd, PointNeighbors,Cluster,MinumumPoints,Epsilon,Visited,DistanceMatrix,PointClusterNumber,PointClusterNumberIndex  ):
        Neighbors=[]

        for i in PointNeighbors:
            if Visited[i]==0:
                Visited[i]=1
                Neighbors=numpy.where(DistanceMatrix[i]<Epsilon)[0]
                if len(Neighbors)>=MinumumPoints:
    #                Neighbors merge with PointNeighbors
                    for j in Neighbors:
                        try:
                            PointNeighbors.index(j)
                        except ValueError:
                            PointNeighbors.append(j)

            if PointClusterNumber[i]==0:
                Cluster.append(i)
                PointClusterNumber[i]=PointClusterNumberIndex
        return
 
