from numpy import * 
import sys


'''
The feed-forward neural net model implemented by hand.
'''

def sigmoid(activation):
	'''
	the standard sigmoid function as the activation function
	'''
	output = 1.0 / (1.0 + exp(-activation))
	return output
class FFN:
	def __init__(self, num_in, num_hidden, num_layer, num_out):
		self.alpha = 0.1 # the default learning right

		# numerical number of nodes and layers documented
		self.nIn= num_in
		self.nHidden=num_hidden
		self.nLayer = num_layer
		self.nOut=num_out

		# initializing weights, using default random from -1 to +1 
		
		# the length of array represents the number of hidden layers, each item in the array is the weight for one layer
		# assume number of neurons in each layers is the same
		# hidden layer weights initialization 
		self.hWeights=[random.random((self.nHidden, self.nIn+1))]  # +1 for weights
		self.hWeights.extend([random.random((self.nHidden, self.nHidden+1)) for i in range(self.nLayer-1)])

		# output layer weights initialization 
		self.oWeights= random.random((self.nOut , self.nHidden+1))

		#creating slots for activation results of hidden layer neurons 
		self.hActivation=[zeros((self.nHidden,1),dtype=float)]
		self.hActivation.extend ([zeros((self.nHidden,1),dtype=float) for i in range(self.nLayer-1)]) # keep the 1 for dot product later
		
		#creating slots for activation results of output layer neurons 
		self.oActivation = zeros((self.nOut,1), dtype=float)

		# creating slots for output result for neurons
		self.hOutput = [zeros((self.nHidden+1,1), dtype=float) for i in range(self.nLayer)] # differ from tutorial
		self.oOutput= zeros((self.nOut,1),dtype=float)
		self.iOutput= zeros((self.nIn+1, 1), dtype=float) # extra 1 for bias

		# deltas for neurons,calculated by difference between output and correct answer/propagation
		self.hDelta = [zeros((self.nHidden),dtype=float) for i in range(self.nLayer)]
		self.oDelta = zeros((self.nOut), dtype = float)

	
	def forward(self, inp):

		# feed in the input layer
		self.iOutput[:-1, 0] = inp
		self.iOutput[-1:, 0] = 1 # the bias term is always set to 1 

		# hidden layer
		for i in range(self.nLayer):
			if i == 0:
				self.hActivation[i]=dot(self.hWeights[i], self.iOutput)
				self.hOutput[i][:-1,:]=sigmoid(self.hActivation[i])
			else:

				self.hActivation[i]=dot(self.hWeights[i], self.hOutput[i-1]) # use output from previous layer
				self.hOutput[i][:-1, :]=sigmoid(self.hActivation[i])
			self.hOutput[i][-1:, :]=1.0

		# output layer
		self.oActivation = dot(self.oWeights, self.hOutput[-1]) # getting input from the last hidden layer

		self.oOutput= sigmoid(self.oActivation)
		return self.oOutput


	def backward(self, expected,learning_rate=0.1): 
		self.alpha=learning_rate

		# computing error by substracting
		format_expected = mat(expected)
		expected= format_expected.reshape(1,len(expected)).transpose()
		error=subtract(self.oOutput, expected)


		# calculating delta 
		# outer delta 
		self.oDelta = multiply( multiply((1- sigmoid(self.oActivation)) , sigmoid(self.oActivation))  , error)

		# adjust hidden layer, multiplying to propagate the delta
		for i in list(reversed(range( self.nLayer ))):
			if i == self.nLayer-1:
				self.hDelta[i]=multiply(multiply((1 - sigmoid(self.hActivation[i])), (sigmoid(self.hActivation[i]))) , dot(self.oWeights[0:,:-1].transpose(), self.oDelta))
			elif i != self.nLayer-1:
				self.hDelta[i]=multiply(multiply((1 - sigmoid(self.hActivation[i])), (sigmoid(self.hActivation[i]))) , dot(self.hWeights[i+1][0:,:-1].transpose(), self.hDelta[i+1]) )
		
		#applying delta weight change 
		self.oWeights = self.oWeights - self.alpha*dot(self.oDelta, self.hOutput[-1].transpose())

		#hidden output change ... now doing 'forward' changds
		for i in range( self.nLayer ):
			if i== 0:
				self.hWeights[i]=self.hWeights[i] - self.alpha * dot(self.hDelta[i], self.iOutput.transpose())
			elif i != 0:
				self.hWeights[i]=self.hWeights[i] - self.alpha * dot(self.hDelta[i], self.hOutput[i-1].transpose())
	
	def train_network(self,X_train,y_train, learning_rate=0.1):
		# assuming x_train and y_train are arrays of arrays
		# forward and backward one pass
		for index,row in enumerate(X_train):
			expected = y_train[index]
			self.forward(row)
			self.backward(expected,learning_rate)

	def predict_batch(self,X_test):
		# predicting many X_entries
		def convert_to_true_labels(a):
			true_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81] 
			return true_labels[a]

		# assume that x_text is an array of arrays
		y_result=[]
		for index, row in enumerate(X_test):
			outputs=self.forward(row)
			outputs=outputs.flatten().tolist()[0]
			#outputs = outputs.index(max(outputs)) # comment this out if want to output original probabilities
			#sprint outputs

			y_result.append(outputs)
		return y_result

	def predict(self, inp):
		# predicting a single entry
		outputs=self.forward(inp)
		outputs=outputs.flatten().tolist()

		return outputs

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
	# sample usage
	X_train = list(reversed([[1,0,2],[0,3,1],[0,3,1],[1,2,1]]))
	y_train=[[1,0],[1,0],[0,1],[1,0]]

	ffn = FFN(num_in=3, num_hidden=4, num_layer=1, num_out=2)
	rounds=60
	for i in range(rounds):
		ffn.train_network(X_train,y_train,learning_rate=0.05)
	print 'finish training'
	print ffn.predict_batch(X_train)

	
	
