# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 03:17:47 2017

@author: Jorge Morales
"""
import byhand
import math
import preprocessing
from sklearn.preprocessing import LabelBinarizer
import numpy   as np 
from sklearn.model_selection import KFold
from math import log
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import csv

"""Calculates alpha given a learnin and epoch"""
def learning_decay(epoch, initial_rate, decay=0.0005):
    rate=initial_rate * 1/(1 + decay * epoch)
    return rate

"""Calculates the generalization loss"""
def generalization_loss(epoch_error,min_error, gl_alpha=0.1):
     generalization_loss=100*((epoch_error/min_error)-1)
     print('gl='+str(generalization_loss))
     return  generalization_loss>gl_alpha or (generalization_loss>0 and math.isclose(generalization_loss, 0, abs_tol=1e-5))
         
"""Finds the training epochs that provide the better results"""
def tune_epochs(x, y, k, input_neurons, hidden_neurons, layers, output_neurons, alpha):
    #Epochs
    training_epoch=0
    best_epoch=None
    #Creates the neural networks for each fold
    nets=[]
    learning_rate=alpha
    for i in range(0,k):
        nets.append(byhand.FFN(input_neurons,hidden_neurons,layers,output_neurons))
    
    #Splits the data into K folds
    kf=KFold(n_splits=k, shuffle=True, random_state=15)
    
    min_val_error=np.inf
    #Iterates over epochs
    while True:
        training_epoch+=1
        print(training_epoch)    
        valid_error=[]
        
        for index, (train_index, validation_index) in enumerate(kf.split(x,y)):   
                NN=nets[index]
                #trains the network
                NN.train_network(x[train_index],y[train_index], learning_rate)
                #gets prediction over validation
                prediction=NN.predict_batch(x[validation_index])
                #print(prediction)
                #Calculates validation error
                err=mean_squared_error(y[validation_index], prediction)
                valid_error.append(err)

        #Average error for every fold
        epoch_error=np.average(valid_error)
        #Checks the stopping condition
        stop_condition=generalization_loss(epoch_error,min_val_error)
        #Updates the min values
        if epoch_error < min_val_error:
            min_val_error= epoch_error
            best_epoch=training_epoch
        if stop_condition:
            break
    
        #updates learning rate
        learning_rate=learning_decay(training_epoch,learning_rate)
    
    #Returns the best epoch and the validation error
    return best_epoch, min_val_error

"""Finds the best learning rate among a set of random values"""
def tune_alpha(x, y, k, input_neurons, hidden_neurons, layers, output_neurons):
    #Learning rates
    N=10
    smaller_exp=-2

    #Generates the random numbers
    learning_rates=np.power(10*np.ones(N),smaller_exp*np.random.random_sample((N,)))
    
    #learning rate
    bestalpha=None
    best_alpha_error=np.inf
    best_epoch=None
    
    #Iterates over the values to train the neural network 
    for learning_rate in learning_rates:
        epoch, epoch_error=tune_epochs(x,y,k,input_neurons,hidden_neurons,layers,output_neurons,learning_rate)
        
        #Updates the minimum error
        if epoch_error < best_alpha_error:
            bestalpha=learning_rate
            best_epoch=epoch
            best_alpha_error=epoch_error
    
    return best_epoch, bestalpha, best_alpha_error

"""Finds the best number of neurons for the hidden layer"""
def tune_neurons(x, y, k, input_neurons, layers, output_neurons):
    #learning rate
    best_number_neurons=None
    best_epoch=None
    best_alpha=None
    best_error=np.inf
     #Hidden Neurons
    neurons=int(log(input_neurons))
    
    #While the error keeps on improving, keep increasing the neurons
    while True:
        epoch, alpha, error=tune_alpha(x,y,k,input_neurons,neurons,layers,output_neurons)
        
        if error > best_error:
            break
        
        best_alpha=alpha
        best_epoch=epoch
        best_number_neurons=neurons
        best_error=error

                
        neurons+=1

    
    return best_epoch, best_alpha, best_number_neurons, best_error

"""Finds the best parameters for the NN"""
def cross_validation(k,x,y):  
    #Input neurons
    inp=x[0].size
    #Output classes
    outp=40
    #Saves a portion of the data for testing
    X_train, Y_train, X_test, Y_test = preprocessing.spltDataset(x, y, 0.1,15)
    
    
    #learning rate
    best_layers=None
    best_number_neurons=None
    best_alpha=None
    best_epoch=None
    best_error=np.inf
     #Hidden layers
    layers=1
    
    #While the result keeps on improving add more layers
    while True:
        epoch, alpha, neurons, error=tune_neurons(X_train,Y_train,k,inp,layers,outp)
        
        if error > best_error:
            break
        
        best_layers=layers
        best_alpha=alpha
        best_epoch=epoch
        best_number_neurons=neurons
        best_error=error
        
        layers+=1
    
    return  best_epoch, best_alpha, best_number_neurons, best_layers,best_error
        

"""Train and test a NN for the given parameters"""
def test_NN(X_train, Y_train, X_valid, Y_valid, test, layers, hidden_neurons, alpha, epochs,labels):
    f=open('NNTrainResults.csv','w')
    writer = csv.writer(f)
    
    
    valid_error=[]
    train_error=[]
    learning_rate=alpha
    
    NN=byhand.FFN(len(X_train[0]),hidden_neurons,layers,40)  
    #Train one epoch at a time
    for epoch in range(0,epochs):
        NN.train_network(X_train,Y_train, learning_rate)
        #gets prediction over training
        trainPrediction=NN.predict_batch(X_train)
        #gets prediction over training
        prediction=NN.predict_batch(X_valid)
        #Calculates validation error
        err_train=mean_squared_error(Y_train, trainPrediction)
        err_val=mean_squared_error(Y_valid, prediction)
        train_error.append(err_train)
        valid_error.append(err_val)
        #Checks the stopping condition
        
        #updates learning rate
        learning_rate=learning_decay(epoch,learning_rate)
        results=[err_train,err_val]
        
    #Gets the confussion matrix of the model    
    prediction_valid=NN.predict_batch(X_valid)
    labs= labels.inverse_transform(np.asarray(prediction_valid))
    y_true= labels.inverse_transform(np.asarray(Y_valid))
    test_matrix=confusion_matrix(y_true,labs)
    
    #Predicts the test set labels
    prediction=NN.predict_batch(test)
    prediction= labels.inverse_transform(np.asarray(prediction))

    return prediction,test_matrix,valid_error,train_error
    
    

x = np.loadtxt("train_x.csv", delimiter=",")
y = np.loadtxt("train_y.csv", delimiter=",")
test = np.loadtxt("test_x.csv", delimiter=",")
print("Finished loading")

#Binarize the data
x=preprocessing.binarize(x)
test=preprocessing.binarize(test)

#Binarize the labels
lb=LabelBinarizer()
Y_train=lb.fit_transform(Y_train)
Y_valid=lb.transform(Y_valid)

#Makes the prediction
prediction, confussion, valid_error,train_error=test_NN(X_train, Y_train, X_valid,Y_valid,test,1,8,0.085692543223609377,69,lb)

#Writes the results
preprocessing.write_file("confusion.csv",confussion)
labels=np.array(['Id','Label'])
ids=list(range(1,prediction.size+1))
toSave=np.c_[ids,prediction.astype(int).astype(str)]
toSave=np.vstack((labels,toSave))
preprocessing.write_file('prediction_NN.csv',prediction)


