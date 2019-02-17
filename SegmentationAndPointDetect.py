# -*- coding: utf-8 -*-
"""
@author: Himanshu Garg
UBID : 50292195
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

s = time.time()

UBID = '50292195'; 
np.random.seed(sum([ord(c) for c in UBID]))

def writeImage(name, img):
    path = "output_imgs/" + name
    cv2.imwrite(path,img)
    print("\n****" + name + " saved****")
    
def display(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, np.array(img,dtype=np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectPoints(image,kernel,T):
    pImg = np.zeros((image.shape[0],image.shape[1]))
    N = len(kernel)
    pd = int(np.floor(N/2))
    for indxr,row in enumerate(image):
        if indxr < pd or indxr > len(image) - pd - 1:
            continue
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            SUM = 0
            for l in range(N):
                r = image[indxr - pd + l][indc - pd:indc + pd + 1]
                SUM+= getProductSum(r,kernel[l],N)
            if abs(SUM) > T :
                #pImg[indxr,indc] = image[indxr,indc]
                pImg[indxr,indc] = 255
    return pImg

def getProductSum(l1,l2,N):
    return sum([l1[k]*l2[k] for k in range(N)])

def createKernel(size,thresh):
    ind = int(size/2)
    krnl = np.ones((size,size)) * -1
    center = (size ** 2) - 1
    krnl[ind,ind] = center
    T = thresh * center * 255
    return krnl,T

def thresholdImage(img,T):
    tImg = np.zeros((img.shape[0],img.shape[1]))
    set1 = []
    set2 = []
    for indr,row in enumerate(img):
        for indc,col in enumerate(row):
            if img[indr,indc] > T:
                tImg[indr,indc] = 255
                set1.append(img[indr,indc])
            else:
                set2.append(img[indr,indc])
    return set1,set2,tImg

def dilate(img,el):
    pd = np.int32(np.floor(len(el)/2))
    dImg = np.copy(img)
    for indr,row in enumerate(img):
        if indr < pd or indr > len(img) - pd - 1:
            continue
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            if el[pd,pd] == img[indr,indc]:
                dImg[indr-pd:indr+pd+1,indc-pd:indc+pd+1] = 255
    return dImg
        
def erode(img,el):
    pd = np.int32(np.floor(len(el)/2))
    dImg = np.copy(img)
    #display('',dImg)
    for indr,row in enumerate(img):
        if indr < pd or indr > len(img) - pd - 1:
            continue
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            if not np.array_equal(el,img[indr-pd:indr+pd+1,indc-pd:indc+pd+1]):
                dImg[indr,indc] = 0
            #else:
            #    dImg[indr,indc] = 255
    return dImg

def doOpening(img,el):
    oimg = erode(img,el)
    oimg = dilate(oimg,el)
    return oimg

def doClosing(img,el):
    cimg = dilate(img,el)
    cimg = erode(cimg,el)
    return cimg

def getBoundingBoxes(opimg):
    pts = np.array(np.where(opimg == 255)).T
    pts = pts[pts[:,1].argsort()]
    temp = []
    rectpts = []
    for ind,pt in enumerate(pts):
        if len(temp) == 0:
            temp.append(pt)
            temp = np.array(temp)
        else:
            diff = abs(pt - temp[len(temp)-1])
            if diff[1] < 10:
                temp = np.append(temp,[pt],axis=0)
                if ind == len(pts)-1:
                    pt1 = [temp[0][1],np.min(temp,axis=0)[0]]
                    pt2 = [temp[len(temp)-1][1],np.max(temp,axis=0)[0]]
                    rectpts.append([pt1,pt2])
            else:
                pt1 = [temp[0][1],np.min(temp,axis=0)[0]]
                pt2 = [temp[len(temp)-1][1],np.max(temp,axis=0)[0]]
                rectpts.append([pt1,pt2])
                temp=[]
                temp.append(pt)
                temp = np.array(temp)
    return rectpts
    
def drawBoundingBoxes(img,pts):
    for pt in pts:
        cv2.rectangle(img,(pt[0][0],pt[0][1]),(pt[1][0],pt[1][1]),(0,0,255),2)


######################################task2.1##############################
pointImg = cv2.imread("original_imgs/point.jpg",0)
#thresh = 0.37
#krnl,T = createKernel(5,thresh)
krnl1 = [[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,24,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]]
krnl2 = np.array([[255,255,255],[255,255,255],[255,255,255]])

T = 2264

pImg = detectPoints(pointImg,krnl1,T)
coord = np.array(np.where(pImg == 255)).T
pImg = dilate(pImg,krnl2)
pImg = cv2.cvtColor(np.asarray(pImg,dtype=np.uint8),cv2.COLOR_GRAY2RGB)
print("Detected point: ",coord[0])
cv2.circle(pImg,(coord[0][1],coord[0][0]),15,(0,0,255),2)
writeImage("point.jpg",pImg)

######################################task2.1##############################
threshObjs = []
segImg = cv2.imread("original_imgs/segment.jpg",0)
segImgCol = cv2.imread("original_imgs/segment.jpg",1)
T = 20
d = T
while d > 15:
    set1,set2,tImg = thresholdImage(segImg,T)
    threshObjs.append(set1)
    m = np.mean(set1)
    d = m-T
    T = m
    
set1,set2,fthImg = thresholdImage(segImg,T)
krnl = np.array([[255,255,255],[255,255,255],[255,255,255]])
fthImg = doClosing(fthImg,krnl)
fthImg = doOpening(fthImg,krnl)

segback2colorimg = cv2.cvtColor(np.asarray(fthImg,dtype=np.uint8),cv2.COLOR_GRAY2RGB)

rectpts = getBoundingBoxes(fthImg)
print("Bounding Boxes: \n",np.array(rectpts))
drawBoundingBoxes(segback2colorimg,rectpts)
drawBoundingBoxes(segImgCol,rectpts)
#display('',segback2colorimg)
writeImage("segment.jpg",segback2colorimg)

print("Elapsed time: ",time.time()-s)

###########################################################################



























