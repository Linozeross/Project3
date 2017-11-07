# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 03:17:47 2017

@author: Jorge Morales
"""
import byhand
import preprocessing
from sklearn.preprocessing import LabelBinarizer
import numpy   as np 
from sklearn.model_selection import KFold
from math import log
from sklearn.metrics import mean_squared_error

def cross_validation(k,x,y):
    #Epochs
    training_epoch=100
    #Input neurons
    inp=x[0].size
    #Hidden layers
    hl=1
    #Hidden Neurons
    hn=int(log(inp))
    #Output classes
    outp=40
    #Output classes
    learning_rate=0.5
    
    #Saves a portion of the data for testing
    X_train, Y_train, X_test, Y_test = preprocessing.spltDataset(x, y, 0.1,15)
    #Splits the data into K folds
    kf=KFold(n_splits=k, shuffle=True, random_state=15)
    
    #Iterates over the sets
    for train_index, validation_index in kf.split(X_train,Y_train):   
            print(X_train[validation_index])
            print(Y_train[validation_index])
            #Creates the neural network
            NN=byhand.FFN(inp,hn,hl,outp)
            NN.train_network(X_train[train_index],Y_train[train_index], learning_rate)
            prediction=NN.predict(X_train[validation_index])
            val_err=mean_squared_error(Y_train[validation_index], prediction)
            print(val_err)
            
        
        

x = np.loadtxt("500_train_x_bin.csv", delimiter=",")
y = np.loadtxt("500_train_y.csv", delimiter=",")

lb=LabelBinarizer()
y_bin=lb.fit_transform(y)
cross_validation(10,x,y_bin)