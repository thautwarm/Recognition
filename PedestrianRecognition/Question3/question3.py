# -*- coding: utf-8 -*-
"""

@author: thautwarm
"""

#question3:
import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.sans-serif']=['FangSong']
plt=mpl.pyplot
from functools import reduce
plt.style.use('ggplot')
from Q3Support import cv2,peopleIndentifySystem,blur,MatchBlock
shp=plt.imshow
path=r"./Pedestrian"
try:
    os.remove('./Pedestrian/Thumbs.db')
except:
    pass
Paths= list(map( lambda x:"./%s/%s"%('Pedestrian',x),os.listdir(path)))


"""
以灰度格式读入图像，存入ims列表
"""
ims= [ blur(cv2.imread(i,0)) for i in Paths]


sys=peopleIndentifySystem()
last_people=[]
last_img=[]
mm=[]#
mc=[]#

rectangle = lambda people: (people[0]+people[2]//2,people[1]+people[3]//2)


#画箭头
def arrow(img,p1,p2):
    cv2.line(s1,p1,p2,(100,255,100),thickness=2)
    cv2.line(s2,p1,p2,(100,255,100),thickness=2)
    dy,dx= np.array(p2)-np.array(p1)
    theta=  np.arctan(dy/dx) + (0 if dx>0 else np.pi) if dx!=0 else  (1 if dy>0 else -1) * np.pi/2 

    phy1=theta+ np.pi*7/6
    phy2=theta+ np.pi*5/6
    
    R=0.4*np.linalg.norm([dx,dy])
    dx1,dx2= (R*np.cos([phy1,phy2])).astype(np.int)
    dy1,dy2= (R*np.sin([phy1,phy2])).astype(np.int)
    if R<=2:return
    Y1,X1=p1
    Y2,X2=p2
    cv2.line(s1,(dy1+Y2,dx1+X2),p2,(100,255,100),thickness=1)
    cv2.line(s1,(dy2+Y2,dx2+X2),p2,(100,255,100),thickness=1)
    cv2.line(s2,(dy1+Y2,dx1+X2),p2,(100,255,100),thickness=1)
    cv2.line(s2,(dy2+Y2,dx2+X2),p2,(100,255,100),thickness=1)
    
    
#连续图依次识别并按照格式存储结果
for num,image_i in enumerate(ims):
    print(num)
    pair=dict()
    mapp=[]
    people_i= sys.identifyPeople(image_i)
    if len(last_people)!=0:
        matRela,matCount=MatchBlock(last_img,last_people,image_i,people_i)
        mm.append(matRela)
        mc.append(matCount)
        print(matRela)
        for i,matRela_i in enumerate(matRela):
            pair=(i,np.argmax(matRela_i))
            if matCount[i][pair[1]]>=2:
                mapp.append(pair)
        mapp=dict(mapp)
        for personIdx in range(len(last_people)):
            if personIdx in mapp:
                toPersonIdx=mapp[personIdx]
                
                s1=cv2.cvtColor(last_img.copy(), cv2.COLOR_GRAY2BGR)
                param1 = rectangle(last_people[personIdx])
                
                s2=cv2.cvtColor(image_i.copy(), cv2.COLOR_GRAY2BGR)
                param2= rectangle(people_i[toPersonIdx])
                arrow(s1,param1,param2)
                arrow(s2,param1,param2)
                rootfile="./result/question3/pic%dTOpic%d"%(num-1,num)
                try:
                    os.makedirs(rootfile)
                except:
                    pass
                dirfile='%s/person%d'%(rootfile,personIdx)
               
                if  not os.path.exists(dirfile):
                    os.makedirs(dirfile)
                    cv2.imwrite('%s/from.png'%(dirfile),s1)
                    cv2.imwrite('%s/to.png'%(dirfile),s2)
    last_people=people_i
    last_img=image_i
    
         
            
    
#    plotRects(image_i,people_i)
#    imgs_i=getRects(image_i,people_i)
#    if len(last_people)!=0:
#        r=matCorr(last_imgs,imgs_i)
#        for i,r_i in  enumerate(r):
#            #maxi=np.max(r_i)
#            pair[i]=np.argmax(r_i) #if maxi>30 else None
#
#    last_imgs=imgs_i
#    last_people=people_i
#    for key in pair:
#            if pair[key]:
#                begin=last_people[key]
#                begin=(begin[0]+begin[2]//2,begin[1])
#                end  = people_i[pair[key]]
#                end=(end[0]+end[2]//2,end[1])
#                cv2.line(image_i,begin,end,color=2) 
#    
#    cv2.imshow('test',image_i)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#        

        
    
            
        
    
    



    
    
    
    
    
    
    
    
    
    
    
    


#X,y,windowGen = ImagSplit(ims[3],F,window=window,stride=strideDiv)



#def getF(X,y=None,window=window,N=splitN):  #降维特征
#    X=X.copy()
#    
#    if isinstance(y,None.__class__):y=F1[0]
#    else:
#        y=y.copy()
#    X,y,windowGen=getData(X,y,window,N)
#    X=np.array([x[0] for x in X])
#    X= np.array(list(map(lambda x:hog.compute(x).T[0] ,X)))
#    return X[:,DimDecressor],y,windowGen
#
#index1=[]
#index2=[]
##用来存储两个图像的轨迹
#imall=reduce(lambda x,y:x.astype(np.int)+y.astype(np.int),ims)
#imall=(imall//len(ims)).astype(np.uint8)
##合并所有图像
#
#"""
#识别物体，记录位置
#"""
#log=[]
#for i in ims:
#    X,y,windowGen=getF(i,N=splitN)
#    y_pred=clf.predict_proba(X)
#    y1=y_pred[:,1]
#    log.append(sorted(y1[y1>=0.5]))
#    obj1= windowGen[np.argmax(y1)]
#    index1.append([obj1[0]+window[0]//2,obj1[1]+window[1]//2])
#
#index1=list(map(lambda x: tuple(x[::-1]), index1))
#
#imall1=imall.copy()
#
#
##在两张图形上分别描轨迹
#for i in range(len(index1)-1):
#    cv2.line(imall1,index1[i],index1[i+1],color=0)
#
#
#this=plt.figure()
#plt.title('Trajectory')
#plt.imshow(imall1)
#this.savefig('./result/question2/q2.png')
#cv2.imwrite('./result/question2/q2CV.png',imall1)
