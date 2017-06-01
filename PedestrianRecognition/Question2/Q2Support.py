# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:01:07 2017

@author: thautwarm
"""
import numpy as np

def getData(X,F,window=(30,30),N=10):
    Y=[]
    
    windowGen=[]
#    ysets= np.unique(F)
    def toY(i,j):
        windowGen.append([i,j,window[0],window[1]])
        Y.append(np.sum(F[i:i+window[0],j:j+window[1]])/(window[0]*window[1]))
#        Y[-1]= ysets[Y[-1]==max(Y[-1])][0]
        Y[-1]=1 if Y[-1]>0.6 else 0
        return np.atleast_3d([X[i:i+window[0],j:j+window[1]]])
    
    Xs=sum([[ toY(i,j) for j in  range(0,X.shape[1]-window[1]-1,N)] 
     for i in range(0,X.shape[0]-window[0]-1,10)],[])
    return np.array(Xs),np.array(Y),windowGen

def strap(X,y):
    index1=y!=0
    N=sum(index1)*2
    index2=y==0
    index3=np.random.permutation(sum(index2))[:N]
    X=np.vstack((X[index1],X[index2][index3] ))
    y=np.hstack( (y[index1],y[index2][index3]))
    return X,y
def overstrap(X,y):
    index1=y!=0
    n1=sum(index1)
    x=[]
    z=[]
    index2=y==0
    N= int( sum(index2)//n1 )
    for i in range(N):
        x.append(X[index1])
        z.append(np.ones(n1,))
    
    X=np.vstack((X[index2],np.vstack(x)))
    y=np.hstack((y[index2],np.hstack(z)))
    return X,y.astype(np.int)

    
    