# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 02:02:13 2017

@author: JorgeEmmanuel
"""

import preprocessing
import numpy   as np
import matplotlib.pyplot as plt
import cv2


"""
Finds the biggest in an image and returns the bounding rect
"""
def find_biggest_blob(image):
    image, contours, hierarchy = cv2.findContours(image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if contours != None and len(contours)>0:
        c=max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
    else:
        x,y,w,h=0,0,0,0
    
    return x,y,w,h

"""
    Deletes a portion of a binarized image defined by a rect
"""
def delete_part(image,x,y,w,h):
    newX=image.copy()
    newX[y:y+h, x:x+w] = 0
    
    return newX

"""
    Extracts part of a binarized image defined by a rect by making all the other values 0
"""
def extract(image,x,y,w,h):
    newX=image.copy()
    
    mask = np.ones(image.shape,dtype=bool) #np.ones_like(a,dtype=bool)
    mask[y:y+h, x:x+w] = False
    newX[mask] = 0
    
    return newX


"""
    Extracts the N biggest blobs of an image and returns an array with N images containing only these and one image with the image without the N biggest blobs
"""
def extract_blobs(image, n=3, returnRemaining=True):
    # reshape 
    binImg=preprocessing.binarize(image).astype(np.uint8)
    binImg=binImg.reshape(64, 64)
    
    
    result=[]
    remaining=binImg
    for i in range(0,n):
        x,y,w,h=find_biggest_blob(remaining)
        result.append(extract(remaining,x,y,w,h))
        remaining=delete_part(remaining,x,y,w,h)
        
    #image=cv2.rectangle(cv2.cvtColor(x0,cv2.COLOR_GRAY2RGBA),(x,y),(x+w,y+h),(0,255,0),1)
    #image = cv2.drawContours(cv2.cvtColor(deleted,cv2.COLOR_GRAY2RGBA), contours, c3, (0,0,255), 1)
    if returnRemaining:
        result.append(remaining)
    return result
   
x = np.loadtxt("train_x.csv", delimiter=",")

result=[]
for i in range(0,len(x)):
    print(i)
    blobs=extract_blobs(x[i],3,True)
    result.append(blobs)

result=np.asarray(result)
result=result.reshape(len(x),-1)
preprocessing.write_file("blobs.csv",result)

#cv2.imshow("C1", blobs[0])
#cv2.imshow("C2", blobs[1])
#cv2.imshow("C3", blobs[2])
#cv2.imshow("Result", blobs[-1])
#cv2.waitKey(0)
#cv2.destroyAllWindows()