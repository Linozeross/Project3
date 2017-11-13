README 

This readme is for hand implemented feed forward neural network model. 

Requirement 
This script requires Python 2.7, with installation of Python libraries numpy 

Usage
- initialize the FFN by calling the FFN object with desired number of input neurons, number of neurons per hidden layer, number of hidden layers, number of output layers. 
  -Example:  ffn = FFN(num_in=3, num_hidden=4, num_layer=1, num_out=2)
- train the model by method: train_network with desired learning rate. Learning rate default to 0.1. 
  -Example : ffn.train_network(X_train,y_train,learning_rate=0.05)
  -X_train and y_train are numpy 2d arrays
- predict a list of X_inputs by calling predict_batch function 
  -Example : ffn.predict_batch(X_test)
