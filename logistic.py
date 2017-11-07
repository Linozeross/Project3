# -*- coding: utf-8 -*-
"""
Created on Thu Nov 02 19:56:30 2017

@author: JorgeEmmanuel
"""
import preprocessing
import numpy   as np 
from sklearn import linear_model
from sklearn.metrics import accuracy_score

print('started')
#x = np.loadtxt("small_train_x.csv", delimiter=",")
x = np.loadtxt("train_x_bin.csv", delimiter=",") # load from text 
print('loaded X')
#y = np.loadtxt("small_train_y.csv", delimiter=",")
y = np.loadtxt("train_y.csv", delimiter=",") 
print('loaded Y')
test = np.loadtxt("test_x_bin.csv", delimiter=",") 

print('loaded')

#training=preprocessing.binarize(x)
#test=preprocessing.binarize(test)

print('binarized')

#preprocessing.write_file("train_x_bin",training)
#preprocessing.write_file("test_x_bin",test)
#
#print('saved bin')

#X_train, Y_train, X_test, Y_test = preprocessing.spltDataset(x, y, 0.2,15)
log_regresion=linear_model.LogisticRegression(C=1e5)
log_regresion.fit(x,y)
prediction=log_regresion.predict(test)
print('predicted')
#mat=accuracy_score(Y_test,prediction)
labels=np.array(['Id','Label'])
ids=list(range(1,prediction.size+1))
toSave=np.c_[ids,prediction.astype(int).astype(str)]
toSave=np.vstack((labels,toSave))
preprocessing.write_file('prediction.csv',toSave)
#print(mat)