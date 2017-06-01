# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:16:18 2017

@author: thautwarm
"""

#question1:
import os,cv2
import numpy as np 
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
from matplotlib import pyplot as plt  
plt.style.use('ggplot')
sh=plt.imshow



path=r"./JPGRing"
Paths= list(map( lambda x:"./%s/%s"%('JPGRing',x),os.listdir(path)))



ims=[ cv2.imread(i,0) for i in Paths]
"""
以灰度格式读入图像，存入ims列表
"""


shape=(186, 235)
F1=[np.zeros(shape) for i in ims]
#
F1[0][10:35,5:35]=1
F1[0][15:40,65:90]=2
"""
给数据打标签。
圆形视为类别1，菱形为2，其余为0
"""



from CNN import getData,ADatafunc
#CNN文件涉及到具体算法。可见CNN.py
import keras,cv2
from functools import reduce
import numpy as np

window=(30,30) #默认大小视窗


"""
以下部分作为训练。
#==============================================================================
# from CNN import TrainNN
# nn=TrainNN(ims[0],F1[0],200,200)
# nn.save('./model/question1.net')   #训练得到模型并保存
#==============================================================================
"""

nn=keras.models.load_model('./model/question1.net') #载入模型



"""
该部分为meanshift和camshift测试，效果不佳
#==============================================================================
#from Q3Support import Shift
# #camshift 追踪
# window_1=[5,10,30,25]
# for i in range(len(ims)-1):
#     window_1=Shift.cam(ims[i],ims[i+1],window_1)
# 
# window_2=[65,15,25,25]
# for i in range(len(ims)-1):
#     window_1=Shift.cam(ims[i],ims[i+1],window_1)
#     
# #meanshift 追踪
# window_1=[5,10,30,25]
# for i in range(len(ims)-1):
#     window_1=Shift.me(ims[i],ims[i+1],window_1)
# 
# window_2=[65,15,25,25]
# for i in range(len(ims)-1):
#     window_1=Shift.me(ims[i],ims[i+1],window_1)
#==============================================================================
"""

index1=[]
index2=[]
#用来存储两个图像的轨迹

#以下处理用来根据像素分布，依靠过滤提取出两类物体，便于后续画图观察

def filter255_1(x):
    x[x==255]=0
    x[x<50]=0
    x[x>200]=0
    return x
def filter255_2(x):
    x[x==255]=0
    x[x>70]=0
    x[x!=0]=29
    return x
    
imall1=reduce(lambda x,y:filter255_2(x.astype(np.int))+filter255_2(y.astype(np.int)),ims).astype(np.uint8)
imall2=reduce(lambda x,y:filter255_1(x.astype(np.int))+filter255_1(y.astype(np.int)),ims).astype(np.uint8)
imall1[imall1==0]=255
imall2[imall2==0]=255

#imall1中只有所有帧里的圆形,imall2中只有所有帧里的菱形

"""
识别物体，记录位置
"""
for i in ims:
    X,y,windowGen=getData(i,F1[0],window)
    X,y=ADatafunc(X,y)
    y_pred=nn.predict_proba(X)
    y1=y_pred[:,1]
    y2=y_pred[:,2]
    obj1= windowGen[np.argmax(y1)]
    obj2= windowGen[np.argmax(y2)]
    index1.append([obj1[0]+window[0]//2,obj1[1]+window[1]//2])
    index2.append([obj2[0]+window[0]//2,obj2[1]+window[1]//2])

index1=list(map(lambda x: tuple(x[::-1]), index1))
index2=list(map(lambda x: tuple(x[::-1]), index2))

imall11=imall1.copy()
imall22 =imall2.copy()
#在两张图形上分别描轨迹

for i in range(len(index1)-1):
    cv2.line(imall1,index1[i],index1[i+1],color=100)
    cv2.line(imall2,index2[i],index2[i+1],color=30)

#在两张图形上分别描预测的中心，在中心处做圆。

for i in range(len(index1)):
    cv2.circle(imall11,index1[i],3,thickness=3,color=100)
    cv2.circle(imall22,index2[i],3,thickness=3,color=30)

this=plt.figure()
plt.title(u'圆的轨迹')
plt.imshow(imall1)
this.savefig('./result/question1/q11.png')

this=plt.figure()
plt.title(u'圆的预测中心')
plt.imshow(imall11)
this.savefig('./result/question1/q11-center.png')


this=plt.figure()
plt.title(u'菱形的轨迹')
plt.imshow(imall2)
this.savefig('./result/question1/q12.png')

this=plt.figure()
plt.title(u'菱形的预测中心')
plt.imshow(imall22)
this.savefig('./result/question1/q12-center.png')

#写入图像
cv2.imwrite('./result/question1/q11CV.png',imall1)
cv2.imwrite('./result/question1/q12CV.png',imall2)

