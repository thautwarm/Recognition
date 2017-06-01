# -*- coding: utf-8 -*-
"""

@author: thautwarm
"""

#question2:
import os,cv2
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['FangSong']
mpl.use('qt4agg')
plt=mpl.pyplot
from util import MLCV
from Q2Support import getData,strap
from sklearn.ensemble import RandomForestClassifier
from functools import reduce
from sklearn.externals import joblib

path=r"./BMPRing"
Paths= list(map( lambda x:"./%s/%s"%('BMPRing',x),os.listdir(path)))
#ims= list(map(lambda x: x*0.5+0.5*GetBinary(x) ,[ cv2.imread(i,0) for i in Paths]))
ims= [ cv2.imread(i,0) for i in Paths]
winsize=(30,30)
blocksize=(10,10)
blockstride=(5,5)
cellsize=(5,5)
hog=cv2.HOGDescriptor(winsize,blocksize,blockstride,cellsize,9)

"""
以灰度格式读入图像，存入ims列表
"""

a=MLCV.toSplit2(ims[0])
shape= ims[0].shape
F1=[np.zeros(shape) for i in ims]
#
F1[0][8:35,6:34]=1
F1[3][170:200,135:165]=1
"""
给数据打标签。
"""




window=(30,30) #默认大小视窗
splitN=10
X_train=np.vstack( (ims[0],ims[3]))
y_train=np.vstack( (F1[0],F1[3]) )

def getF(X,y=None,window=window,N=splitN): #原始特征
    X=X.copy()
    
    if isinstance(y,None.__class__):y=F1[0]
    else:
        y=y.copy()
    X,y,windowGen=getData(X,y,window,N)
    X=np.array([x[0] for x in X])
    X= np.array(list(map(lambda x:hog.compute(x).T[0] ,X)))
    return X,y,windowGen

    
X,y,windowGen=getF(X_train,y_train,window,10)
X,y=strap(X,y)
clf=RandomForestClassifier(n_estimators=30)
clf.fit(X,y)
DimDecressor=clf.feature_importances_>0.005*sum(clf.feature_importances_)
X=X[:,DimDecressor]

clf.fit(X,y)

#模型持久化
joblib.dump(clf,'./model/question2.tree')
joblib.dump(DimDecressor,'./model/DimDecressorForQ2.arr')
clf=joblib.load('./model/question2.tree')
DimDecressor=joblib.load('./model/DimDecressorForQ2.arr')

def getF(X,y=None,window=window,N=splitN):  #降维特征
    X=X.copy()
    
    if isinstance(y,None.__class__):y=F1[0]
    else:
        y=y.copy()
    X,y,windowGen=getData(X,y,window,N)
    X=np.array([x[0] for x in X])
    X= np.array(list(map(lambda x:hog.compute(x).T[0] ,X)))
    return X[:,DimDecressor],y,windowGen


#用来存储两个图像的轨迹
index1=[]
index2=[]


#合并所有图像
imall=reduce(lambda x,y:x.astype(np.int)+y.astype(np.int),ims)
imall=(imall//len(ims)).astype(np.uint8)


"""
识别物体，记录位置
"""
log=[]
for i in ims:
    X,y,windowGen=getF(i,N=splitN)
    y_pred=clf.predict_proba(X)
    y1=y_pred[:,1]
    log.append(sorted(y1[y1>=0.5]))
    obj1= windowGen[np.argmax(y1)]
    index1.append([obj1[0]+window[0]//2,obj1[1]+window[1]//2])

index1=list(map(lambda x: tuple(x[::-1]), index1))

imall1=imall.copy()


#在两张图形上分别描轨迹
for i in range(len(index1)-1):
    cv2.line(imall1,index1[i],index1[i+1],color=0)

#画图存储
this=plt.figure()
plt.title('Trajectory')
plt.imshow(imall1)
this.savefig('./result/question2/q2.png')
cv2.imwrite('./result/question2/q2CV.png',imall1)
