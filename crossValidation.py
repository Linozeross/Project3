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
import csv


def learning_decay(epoch, initial_rate, decay=0.0005):
    rate=initial_rate * 1/(1 + decay * epoch)
    return rate

def generalization_loss(epoch_error,min_error, gl_alpha=0.1):
     generalization_loss=100*((epoch_error/min_error)-1)
     print('gl='+str(generalization_loss))
     return  generalization_loss>gl_alpha
         

def tune_epochs(x, y, k, input_neurons, hidden_neurons, layers, output_neurons, alpha):
    print("inputs: %d, hidden: %d, layers: %d, output: %d, alpha: %f"% (input_neurons,hidden_neurons,layers,output_neurons,alpha))
    f=open('crossValidationResults.csv','a')
    writer = csv.writer(f)
    
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
        results=[input_neurons,hidden_neurons,layers,output_neurons,alpha,training_epoch,epoch_error]
        writer.writerow(results)
        print('epoch_error='+str(epoch_error))
        
        if epoch_error < min_val_error:
            min_val_error= epoch_error
            best_epoch=training_epoch
        if stop_condition:
            break
    
        #updates learning rate
        learning_rate=learning_decay(training_epoch,learning_rate)
    
    f.close()
    #Returns the best epoch and the validation error
    return best_epoch, min_val_error
    
def tune_alpha(x, y, k, input_neurons, hidden_neurons, layers, output_neurons):
    #Learning rates
    N=2
    smaller_exp=-2
    learning_rates=np.power(10*np.ones(N),smaller_exp*np.random.random_sample((N,)))
    
    #learning rate
    bestalpha=None
    best_alpha_error=np.inf
    best_epoch=None
    
    for learning_rate in learning_rates:
        epoch, epoch_error=tune_epochs(x,y,k,input_neurons,hidden_neurons,layers,output_neurons,learning_rate)
        
        if epoch_error < best_alpha_error:
            bestalpha=learning_rate
            best_epoch=epoch
            best_alpha_error=epoch_error
    
    return best_epoch, bestalpha, best_alpha_error

def tune_neurons(x, y, k, input_neurons, layers, output_neurons):
    #learning rate
    best_number_neurons=None
    best_epoch=None
    best_alpha=None
    best_error=np.inf
     #Hidden Neurons
    neurons=int(log(input_neurons))
    
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
    
def cross_validation(k,x,y):  
    #Input neurons
    inp=x[0].size
    #Output classes
    outp=40
    f=open('crossValidationResults.csv','w')
    results=["inputs","hidden","layers", "output", "alpha","epoch","validation_error"]
    writer = csv.writer(f)
    writer.writerow(results)
    f.close()
    #Saves a portion of the data for testing
    X_train, Y_train, X_test, Y_test = preprocessing.spltDataset(x, y, 0.1,15)
    
    
    #learning rate
    best_layers=None
    best_number_neurons=None
    best_alpha=None
    best_epoch=None
    best_error=np.inf
     #Hidden layers
    layers=5
    
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
        

x = np.loadtxt("500_train_x_bin.csv", delimiter=",")
y = np.loadtxt("500_train_y.csv", delimiter=",")

lb=LabelBinarizer()
y_bin=lb.fit_transform(y)
epoch, alpha, neurons, layers,error=cross_validation(10,x,y_bin)