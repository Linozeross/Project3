# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 19:56:30 2017

@author: JorgeEmmanuel
"""
import preprocessing
import numpy   as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score


x = np.loadtxt("small_train_x.csv", delimiter=",") # load from text 
y = np.loadtxt("small_train_y.csv", delimiter=",") 
image=preprocessing.binarize(x)


X_train, Y_train, X_test, Y_test = preprocessing.spltDataset(x, y, 0.2,15)
log_regresion=linear_model.LogisticRegression(C=1e5)
log_regresion.fit(X_train,Y_train)
prediction=log_regresion.predict(X_test)
mat=accuracy_score(Y_test,prediction)
print(mat)

#x = image.reshape(-1, 64, 64) # reshape 
#y = y.reshape(-1, 1) 
#scipy.misc.imshow(x[0]) # to visualize only 

#plt.imshow(np.uint8(x[0]))
#plt.show()