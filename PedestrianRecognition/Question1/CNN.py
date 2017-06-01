# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:01:07 2017

@author: thautwarm
"""

from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense,MaxPooling2D,Dropout
import numpy as np

def getData(X,F,window=(30,30),N=10):
    """
    #可根据图片 和 像素级标签，利用滑窗法生成 窗口图片。
    滑窗的步长为 ：N:int
    图片: X:ndarray
    像素级标签: F:ndarray
    窗口大小： window:tuple[int,int]
    """
    Y=[]
    windowGen=[]
    ysets= np.unique(F)
    def toY(i,j):
        windowGen.append([i,j,window[0],window[1]])
        a=list(np.hstack( np.atleast_2d (F[i:i+window[0],j:j+window[1]])) )
        Y.append(np.array( [a.count(i) for i in ysets ]) )
        Y[-1]= ysets[Y[-1]==max(Y[-1])][0]
        return np.atleast_3d([X[i:i+window[0],j:j+window[1]]])
    
    Xs=sum([[ toY(i,j) for j in  range(0,X.shape[1]-window[1]-1,N)] 
     for i in range(0,X.shape[0]-window[0]-1,10)],[])
    return np.array(Xs),np.array(Y),windowGen
def getNN(n):
    """
    定义卷积网络结构。
    具体结构设计参考著名的VGG网络。
    """
    nn=Sequential()
    nn.add(Convolution2D(32,(3,3),input_shape=(30,30,1),activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Convolution2D(16,(3,3),activation='relu'))
    nn.add(Dropout(0.2))
    nn.add(Convolution2D(8,(3,3),activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Convolution2D(8,(3,3),activation='relu'))
    nn.add(Dense(50,activation='tanh'))
    nn.add(Dropout(0.2))
    nn.add(Dense(50,activation='tanh'))
    nn.add(Flatten())
    nn.add(Dense(n,activation='sigmoid'))
    nn.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    return nn
def strap(X,y):
    """
    #采样。这里是降采样
    """
    index1=y!=0
    N=sum(index1)*2
    index2=y==0
    index3=np.random.permutation(sum(index2))[:N]
    X=np.vstack((X[index1],X[index2][index3] ))
    y=np.hstack( (y[index1],y[index2][index3]))
    return X,y

def ADatafunc(X,y):
    """
    #数据预处理函数，作为传入TrainNN的参数。这里是该参数的默认定义。
    """
    trains=X.copy()
    trains=trains/255.0
    trains=trains.swapaxes(1,3)
    trains=trains.swapaxes(1,2)
    tags=y.copy()
    return trains,tags
    
def AFitfunc(f):
    """
    #训练函数。用来定义训练网络的方式，作为传入TrainNN的参数。这里是该参数的默认定义。
    """
    f()
def TrainNN(X,y,batch,epochs,datafunc=ADatafunc,fitfunc=AFitfunc,BDatafunc=strap,dataSplit_window=(30,30),dataSplit_N=10):
    """
    根据图片生成一个CNN模型。
    """
    trains,tags,windowGen=getData(X,y,window=dataSplit_window ,N=dataSplit_N)
    nn=getNN(len(np.unique(tags)))
    def u():
        for i in range(20):
            trains_b,tags_b=strap(trains,tags)
            trains_b,tags_b=ADatafunc(trains_b,tags_b)
            nn.fit(trains_b,tags_b,batch_size=batch,epochs=epochs)
    fitfunc(u)
    return nn
    
    