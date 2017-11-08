from numpy import * 
import sys


'''
-ask if you train (forward/backward) arbitrarily amount of times 
- is it normal to have seen an example, but still get it wrong 
- ask about the bias term, and how is it handled? 
'''

def sigmoid(activation):
	output = 1.0 / (1.0 + exp(-activation))
	return output
class FFN:
	def __init__(self, num_in, num_hidden, num_layer, num_out):
		self.alpha = 0.1

		# numerical number of nodes and layers documented
		self.nIn= num_in
		self.nHidden=num_hidden
		self.nLayer = num_layer
		self.nOut=num_out

		# initializing weights
		# the length of array represents the number of hidden layers
		# assume number of neurons in each layers is the same
		self.hWeights=[random.random((self.nHidden, self.nIn+1))]
		self.hWeights.extend([random.random((self.nHidden, self.nHidden+1)) for i in range(self.nLayer-1)])

		#self.oWeights=[random.random((self.nOut, self.nHidden)) for i in range(self.nLayer)] # different from tutorial
		self.oWeights= random.random((self.nOut , self.nHidden+1))

		#activation results of neurons 
		self.hActivation=[zeros((self.nHidden,1),dtype=float)]
		self.hActivation.extend ([zeros((self.nHidden,1),dtype=float) for i in range(self.nLayer-1)]) # keep the 1 for dot product later
		
		self.oActivation = zeros((self.nOut,1), dtype=float)

		# outputs of neurons, after sigmoid function
		self.hOutput = [zeros((self.nHidden+1,1), dtype=float) for i in range(self.nLayer)] # differ from tutorial
		self.oOutput= zeros((self.nOut,1),dtype=float)
		self.iOutput= zeros((self.nIn+1, 1), dtype=float) # extra 1 for bias

		# deltas for neurons
		# QUESTION: why not nHidden+1 ??
		self.hDelta = [zeros((self.nHidden,1),dtype=float) for i in range(self.nLayer)]
		self.oDelta = zeros((self.nOut), dtype = float)

	
	def forward(self, inp):

		# feed in the input layer, force it output
		self.iOutput[:-1, 0] = inp
		self.iOutput[-1:, 0] = 1 # the bias term is always set to 1 

		# hidden layer
		for i in range(self.nLayer):
			if i == self.nLayer-1:
				self.hActivation[i]=dot(self.hWeights[i], self.iOutput)
				self.hOutput[i][:-1,:]=sigmoid(self.hActivation[i])
			else:
				self.hActivation[i]=dot(self.hWeights[i], self.hOutput[i-1])
				self.hOutput[i][:-1, :]=sigmoid(self.hActivation[i])
			self.hOutput[i][-1:, :]=1.0

		# output layer
		self.oActivation = dot(self.oWeights, self.hOutput[-1])

		self.oOutput= sigmoid(self.oActivation)
		return self.oOutput


	def backward(self, expected,learning_rate=0.1): 
		self.alpha=learning_rate

		# did not recompute the thing 
		error = self.oOutput - array(expected, dtype=float)

		# calculating delta 
		# outer delta 
		#self.oDelta = (1- self.oOutput)*self.oOutput * error
		self.oDelta = (1- sigmoid(self.oActivation))*sigmoid(self.oActivation)* error

		# adjust hidden layer 
		for i in list(reversed(range( self.nLayer ))):
			if i == self.nLayer-1:
				# print self.oWeights.shape
				# print self.oDelta.shape
				# print self.hActivation[i].shape
				# sys.exit()
				self.hDelta[i]=(1 - sigmoid(self.hActivation[i]))*(sigmoid(self.hActivation[i]))*dot(self.oWeights[0:,:-1].transpose(), self.oDelta)
			elif i != self.nLayer-1:
				self.hDelta[i]=(1 - sigmoid(self.hActivation[i]))*(sigmoid(self.hActivation[i]))*dot(self.hWeights[i+1][0:,:-1].transpose(), self.hDelta[i+1])
		
		#applying delta weight change 
		# Outer output change
		self.oWeights = self.oWeights - self.alpha*dot(self.oDelta, self.hOutput[-1].transpose())

		#hidden output change ... now doing 'forward' changds
		for i in range( self.nLayer ):
			if i== 0:
				# print self.hDelta[i].shape
				# print self.iOutput.transpose().shape
				# print dot(self.hDelta[i], self.iOutput.transpose()).shape
				# print self.hWeights[i].shape
				# sys.exit()

				self.hWeights[i]=self.hWeights[i] - self.alpha * dot(self.hDelta[i], self.iOutput.transpose())

			elif i != 0:
				self.hWeights[i]=self.hWeights[i] - self.alpha * dot(self.hDelta[i], self.hOutput[i-1].transpose())
	
	def train_network(self,X_train,y_train, learning_rate=0.1):
		# assuming x_train  and y_train are arrays of arrays
		for index,row in enumerate(X_train):
			expected = y_train[index][0]
			self.forward(row)
			self.backward(expected,learning_rate)

	def predict(self,X_test):

		def convert_to_true_labels(a):
			true_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] 
			return true_labels[a]

		# assume that x_text is an array of arrays
		y_result=[]
		for index, row in enumerate(X_test):
			outputs=self.forward(row)
			outputs=outputs.flatten().tolist()
			#outputs = outputs.index(max(outputs))
			y_result.append(outputs)
		return y_result

	def getOutput(self):
		return self.oOutput

def convert_to_neural_input(y_labels):
	result=[]
	true_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] 
	for label in y_labels:
		array_label=[]
		for a in true_labels:
			if label == a:
				array_label.append(1)
			else:
				array_label.append(0)
		result.append(array_label)
	return result
	pass
if __name__ == '__main__':
	X_train = [[0,0],[0,1],[0,0],[1,1]]
	y_train=[[0,1],[0,1],[0,1],[1,0]]

	ffn = FFN(num_in=2, num_hidden=4, num_layer=1, num_out=2)
	epochs=60000
	for i in range(epochs):
		ffn.train_network(X_train,y_train,learning_rate=0.05)
	print 'finish training'
	print ffn.predict([[1,1]])

	
	
