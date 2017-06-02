# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:55:29 2017

@author: thautwarm
question3Support

1.确定 人 这一类物体。唯一识别。物体在一些区域内。
2.计算 区域特征，对每个人对象绑定 区域分布特征。加入知识库。
3.加载时序图。
4.唯一识别 人对象，根据区域分布特征比较。确定人对象身份。
5.知识遗忘。

"""
import cv2,re
import numpy as np 
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
from matplotlib import pyplot as plt  
MyCos= lambda vector_i,vector_j:2* np.dot(vector_i,vector_j)/\
            (0.01+np.sum(np.square(vector_i))+np.sum(np.square(vector_j) ))

haskmap=lambda *x:list(map(*x))
haskfilter=lambda *x:list(filter(*x))

#识别行人
window=(60,40)

#Sift

Sift = cv2.xfeatures2d.SIFT_create()

#Hog
winsize=(35, 35)
blocksize=(15, 15)
blockstride=(10,10)
cellsize=(5,5)



hogSVM = cv2.HOGDescriptor()
hogSVM.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
bigBodyClassify=lambda image:hogSVM.detectMultiScale(image,winStride=(2, 2),padding=(0, 0), scale=1.05)

smallBodyClassifyRoot=cv2.CascadeClassifier("./openCvModel/haarcascade_fullbody.xml")
smallBodyClassify=lambda subimage:smallBodyClassifyRoot.detectMultiScale(subimage)

faceClassifyRoot=cv2.CascadeClassifier("./openCvModel/haarcascade_frontalface_default.xml")
faceClassify=lambda subimage:faceClassifyRoot.detectMultiScale(subimage)


upperBodyClassifyRoot=cv2.CascadeClassifier("./openCvModel/haarcascade_upperbody.xml")
upperBodyClassify=lambda subimage :upperBodyClassifyRoot.detectMultiScale(subimage)


eyeClassifyRoot=cv2.CascadeClassifier("./openCvModel/haarcascade_frontalface_alt.xml")
eyeClassify=lambda subimage :eyeClassifyRoot.detectMultiScale(subimage)




Hog=cv2.HOGDescriptor(winsize,blocksize,blockstride,cellsize,9)
getHogFeature=lambda image:Hog.compute(image).T[0]



#Hist
getHistFeature=lambda image:cv2.calcHist([image],[0],None,[16],[0.0,256.0]).T[0]
calcBackProject=lambda image:cv2.calcBackProject([image],[0],None,[16],[0.0,256.0])

mask=lambda image,roi:cv2.calcHist([image],[0],roi,[16],[0.0,256.0]).T[0]
"""
"""
#image::filter

blur=lambda image:cv2.blur(image,(3,3))

GaussianBlur_size=1
GaussianBlur_sigma=1.5
KSIZE = GaussianBlur_size * 2 +3
gauss=lambda image:cv2.GaussianBlur(image,(3,3),GaussianBlur_sigma,GaussianBlur_size)
"""
"""

def strap(X,y):
    index1=y!=0
    N=sum(index1)*2
    index2=y==0
    index3=np.random.permutation(sum(index2))[:N]
    X=np.vstack((X[index1],X[index2][index3] ))
    y=np.hstack( (y[index1],y[index2][index3]))
    return X,y
    
def ImagSplit(img,TagSpace=None,window=window,stride=(10,10)):
    windowGen=[]
    Datas=[]
    dx,dy=window
    Target=[]

        
            
        
        
    if not isinstance( TagSpace, None.__class__):
        def tagF(Tag,i,j):
            subTag=Tag[i:i+dx,j:j+dy]
            m,n=subTag.shape
            if np.count_nonzero(subTag)/(m*n*1.0) <0.6:
                Target.append( 0 )
            else:
                Target.append( 1 )
        passFunc= tagF
    else:
        passFunc=lambda *x: None
    for i in range(0,img.shape[0]-window[0]-1,stride[0]):
        for j in range(0,img.shape[1]-window[1]-1,stride[1]):
            Datas.append(img[i:i+dx,j:j+dy])
            windowGen.append([i,j,dx,dy])
            passFunc(TagSpace,i,j)
            
    return np.array(Datas),np.array(Target),np.array(windowGen)

def featureGet(img):
#    histfea=getHistFeature(img)
    hogfea=getHogFeature(img)
    return hogfea #np.hstack((histfea,hogfea))        

def getPair2(im1,im2):
    r1=MyCos(getHistFeature(im1),getHistFeature(im2))
    return r1*SIFTMATCHCOUNT(im1,im2)
def getPair(im1,im2):
    r1=MyCos(getHistFeature(im1),getHistFeature(im2))
    p1,v1=Sift.detectAndCompute(im1,None)
    p2,v2=Sift.detectAndCompute(im2,None)
    if not p1 or not p2:
        return 0
    count=0
    v2INDEX=np.arange((v2.shape[0]))
#    for vec_i in v1:
#        for vec_j in v2:
#            r=MyCos(vec_i,vec_j)
#            if MyCos(vec_i,vec_j)>0.8:
#                count+=r
                
    
    for i,vector_i in enumerate(v1):
        r=0.0
        for j in v2INDEX:
            if i==j:continue
            if v2INDEX[j]==-1:continue
            vector_j=v2[j]
            rnew=MyCos(vector_i,vector_j)
            if rnew>0.75 and rnew>r :
                remove_index=j
                r=rnew
        try:
            v2INDEX[remove_index]=-1
            count+=1
            del remove_index
        except:
            pass
        
    return np.log(3+count)*r1

    
def getRects(img,rects):
    return [img[x:x+w,y:y+h] for (x,y,w,h) in rects]
def getRect(img,rect):
     (x,y,w,h)= rect 
     return img[x:x+w,y:y+h]

def axis2str(vec):
    return ' '.join(map(lambda x:"%d"%x,vec))
def str2axis(str):
    return np.array(list(map(int, re.findall('\d+',str)) ))
def plotRects(img,rects):
    img=img.copy('C')
    for (x,y,w,h) in rects:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,100,100),2)
    cv2.imshow('ok!',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getRectFeature(rect):
    x,y,w,h=rect
    center=np.array([x+w//2,y+h//2,w,h])
    return center
    
def CombineRects(rects,deg=2,filterMax=False):
#    ss = [ axis2str(i[:2]) for i in deg*(np.round(rects/deg)).astype(int) ]
#    _,index=np.unique(ss,return_index=True)
#    rects=np.array(rects)[index]
    u=[False]
    def setNotEq(i,j):
        u[0]= i!=j
        return i>=j
        return 
    remove_index=np.ones(len(rects))
    for i,rect_i in enumerate(rects):
        x1,y1,w1,h1=rect_i
        if  remove_index[i]==0:
                continue
        for j,rect_j in enumerate(rects):
            if remove_index[j]==0:
                continue
            x2,y2,w2,h2=rect_j
            u=[False]
            if setNotEq(x2,x1) and setNotEq(y2,y1) and setNotEq(x1+w1,x2+w2) and setNotEq(y1+h1,y2+h2) and u[0]:
                if filterMax:
                   remove_index[j]=0
                else: 
                    remove_index[i]=0
    return rects[remove_index.astype(np.bool)]
                
class Shift:
    def __init__(self,frame,rect,method='m'):
        r,c,h,w=rect
        roi = frame[r:r+h, c:c+w]
        mask = cv2.inRange(roi, np.array((0.)), np.array((255.)))
        roi_hist = cv2.calcHist([roi],[0],mask,[16],[0,255])
        roi_hist=cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        plotRects(frame,[rect])
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        self.roi_hist=roi_hist
        self.track_window=tuple(rect)
        self.m=method
        self.frame=frame
    def deal(self,frame):
        frame=frame.copy()
        track_window=self.track_window
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        roi_hist=self.roi_hist 
        dst = cv2.calcBackProject([frame],[0],roi_hist,[0,180],1)
        if self.m=='m':
            ret, track_window_r = cv2.meanShift(dst, track_window, term_crit)
            x,y,w,h = track_window_r
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        elif self.m=='c':
            ret, track_window_r = cv2.CamShift(dst, track_window, term_crit)
            
            
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame,[pts],True, 255,2)
        rectsNew=[]

        center1=(track_window[0]+track_window[2]//2,track_window[1]+track_window[3]//2)
        center2=(track_window_r[0]+track_window_r[2]//2,track_window_r[1]+track_window_r[3]//2)
        img2 = cv2.line(img2,center1,center2,color=0)
        rectsNew=track_window_r
#        x,y,w,h = track_window
#        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        return rectsNew
    @staticmethod
    def cam(frame1,frame2,a):
        clf=Shift(frame1,a,'c')
        return clf.deal(frame2)
    def me(frame1,frame2,a):
        clf=Shift(frame1,a,'m')
        return clf.deal(frame2)
        

    
class Match:
    def __init__(self,frame,rect):
        self.hist=Match.calHist(frame,rect)
    
    
    def matches(self,frame,rects):
        cal=lambda x:Match.calHist(frame,x)
        hists=list(map(cal,rects))
        print(list(map(lambda x:cv2.compareHist(x,self.hist,cv2.HISTCMP_BHATTACHARYYA),hists)))
        
        
    @staticmethod
    def calHist(frame,rect):
        r,c,h,w=rect
        roi = frame[r:r+h, c:c+w]
        mask = cv2.inRange(roi, np.array((0.)), np.array((255.)))
        roi_hist = cv2.calcHist([roi],[0],mask,[255],[0,255])
        roi_hist=cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        return roi_hist
   
class peopleIndentifySystem:
    def __init__(self):
        #mainIndentifier
        self.bigBodyClassify=bigBodyClassify
        self.smallBodyClassify=smallBodyClassify
        
        #extraIdentifier
        self.eyeClassify=eyeClassify
        self.upperBodyClassify=upperBodyClassify
        self.faceClassify=faceClassify
    def identifyPeople(self,img):
        people1,weights = self.bigBodyClassify(img)
        people2 = self.smallBodyClassify(img)
        people3 = self.upperBodyClassify(img)
#        people4 = self.eyeClassify(img)
#        people5 = self.faceClassify(img)
        
#        bigger=lambda x: 40 if x<40 else x
#        maxbit=lambda L: [L[0],L[1],L[2],bigger(L[3])]
#        downabit=lambda L: [L[0],L[1]+10,L[2],L[3]]
#        upabit=lambda L: [L[0],L[1]-10,L[2],L[3]]
#
#        people11 = np.array(list(map(downabit,people3)))
#        people12 = np.array(list(map(upabit,people3)))
#        
#        people31= np.array(list(map(downabit,people3)))
#        people32= np.array(list(map(maxbit,people3)))
#        people51 = np.array(list(map(downabit,people5)))
#        people52 = np.array(list(map(maxbit,people5)))
#        tup=( people1,people11,people12,people2,people31,people32,people4,people51,people52)
#        
#        tup=(people4,people5,people1,people2,people3)
        tup=(people1,people2)
        tup=list(filter(lambda x:not isinstance(x,tuple),tup))
#        people1 = self.eyeClassify(img)
#        people2 = self.upperBodyClassify(img)
#        bigger=lambda x: 40 if x<40 else x
#        maxbit=lambda L: [L[0],L[1],L[2],bigger(L[3])]
#        downabit=lambda L: [L[0],L[1]+10,L[2],L[3]]
#        upabit=lambda L: [L[0],L[1]-10,L[2],L[3]]
#        people= self.eyeClassify(img)
#        adjust=lambda func:np.array(list(map(func, [people_i for people_i in people])))
#        people1=adjust(maxbit);
#        people2=adjust(downabit);
#        people3=adjust(upabit)
#        tup=(people,people1,people2,people3)
#        tup=(people1,people2)
        People=CombineRects(np.vstack(tup))
#        print(people1)
        if not isinstance(people3,tuple):
            People= CombineRects(np.vstack ( (People, people3) ),filterMax=True)
        
        
        
        return People
"""
Following codes were abandoned from peopleIndentifySystem::identifyPeople.
"""
#        PeopleFeature=[]
#        PeopleFilter=[]
#        checkNone= lambda x: isinstance(x,tuple)
#        for people in People:
#            peopleStdImage=getRect(img,people)
#            eye=self.eyeClassify(peopleStdImage)
#            upper=self.upperBodyClassify(peopleStdImage)
#            face=self.faceClassify(peopleStdImage)
#            if checkNone(eye) and checkNone(upper) and  checkNone(face):
#                print(1)
#                continue
#            PeopleFeature.append([peopleStdImage,eye,upper,face])
#            PeopleFilter.append(people)
            
def SIFTMATCH(img1,img2):
    img1=img1.copy()
    img2=img2.copy()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = Sift.detectAndCompute(img1,None)
    kp2, des2 = Sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if 0.55*n.distance<m.distance < 0.80*n.distance:
            matchesMask[i]=[1,0]
            # cv2.drawMatchesKnn expects list of lists as matches.
    img3=None
    draw_params=dict(matchesMask=matchesMask)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2,**draw_params)
#    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2)
    plt.imshow(img3,cmap='gray')

def SIFTMATCHPOINTS(img1,img2):
    img1=img1.copy()
    img2=img2.copy()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = Sift.detectAndCompute(img1,None)
    kp2, des2 = Sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    matchesMask =np.array( [0 for i in range(len(matches))])
    good=[]
    for i,(m,n) in enumerate(matches):
        if 0.50*n.distance<m.distance < 0.85*n.distance:
            good.append(m)
            matchesMask[i]=1
    src_pts = [ tuple([int(pos) for pos in  kp1[m.queryIdx].pt]) for m in good ]
    dst_pts = [ tuple([int(pos) for pos in  kp2[m.trainIdx].pt]) for m in good ]
    return dict(zip(src_pts,dst_pts))
#    kp1=np.array(kp1)[matchesMask==1]
#    kp2=np.array(kp2)[matchesMask==1]
#    kp1pt=list(map(lambda x: tuple([int(posi) for posi in x.pt]),kp1))
#    kp2pt=list(map(lambda x: tuple([int(posi) for posi in x.pt]),kp2))
#    return dict(zip(kp1pt,kp2pt))
def SIFTMATCHCOUNT(img1,img2):
    img1=img1.copy()
    img2=img2.copy()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = Sift.detectAndCompute(img1,None)
    kp2, des2 = Sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    if len(np.array(matches).shape)!=2 or np.array(matches).shape[1]!=2:
        return 0
    # Apply ratio test
    good = []
    for m,n in matches:
        if 0.50*n.distance<m.distance < 0.80*n.distance:
            good.append([m])
    return len(good)
    
def matCorr(ims1,ims2):
    return np.array([ [getPair(i,j) for j in ims2 ] for i in ims1])
    
    
def JudgeIfInRect(rect,point):
    x,y,w,h=rect
    center=(x+w//2,y+h//2)
    x0,y0=point
    influence_factor = 2-(abs(x0-center[0])*1.0/w+ abs(y0-center[1]*1.0)/h)
    return x0>=x and y0>=y and x0<=x+w and y0<=y+h ,influence_factor
    
    
    
def getFeaturePointsMapped(rect,pairs):
    points=[]
    factors=[]
    for img_1_posi in pairs:
        jdg,factor=JudgeIfInRect(rect,img_1_posi)
        if jdg:
            img_2_posi=pairs[img_1_posi]
            points.append(img_2_posi)
            factors.append(factor)
    return points,factors
def getFeaturePointsCurrent(rect,points,factors):
    pointsCurrent=[]
    factorsCurrent=[]
    
    for point,factor_ori in zip(points,factors):
        jdg,factor = JudgeIfInRect(rect,point)
        if jdg:
            pointsCurrent.append(point)
            factorsCurrent.append(factor*factor_ori)
    return pointsCurrent,factorsCurrent
            
    
    

    
    
def ppp(r,im1,ws1,im2,ws2):
    for i,r_i in enumerate(r):
        idx=np.argmax(r_i)
        plotRects(im1,ws1[i:i+1])
        plotRects(im2,ws2[idx:idx+1])


def MatchBlock(img1,people1,img2,people2):
    KeyPointsMapp=SIFTMATCHPOINTS(img1,img2)
    Num1=len(people1)
    Num2=len(people2)
    MappingScore=[ [ 0 for j in range(Num2) ] for i in range(Num1) ]
    MappingCount=[ [ 0 for j in range(Num2) ] for i in range(Num1) ]
    for i,people_i in enumerate(people1):
        pointsMapped ,factorsFrom =getFeaturePointsMapped(people_i,KeyPointsMapp)
        for j,people_j in enumerate(people2):
            points , factors =getFeaturePointsCurrent(people_j,pointsMapped,factorsFrom)
            MappingScore[i][j]+=np.sum(factors)
            MappingCount[i][j]+=len(points)
    return MappingScore,MappingCount
            
    
        
        
    
    
    
    
