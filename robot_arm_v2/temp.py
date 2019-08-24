# Inputs: [0,1]
# Activation: Sigmoidal
# Output: [0,1]
# Accuracy: ~80%
import numpy as np
from scipy.special import expit as sigmoid

class NeuronLayer:

	def __init__(self, neuronCount, nextNeuronCount, first=False):
		self.neuronCount = neuronCount
		self.nextNeuronCount = nextNeuronCount
		self.neurons = np.zeros((self.neuronCount, 1))
		if (first):
			self.neuronErrors = np.zeros((self.neuronCount, 1))
		if (nextNeuronCount != 0):
			self.synapseCount = (self.neuronCount) * self.nextNeuronCount
			self.synapses = np.random.rand(self.nextNeuronCount, self.neuronCount)
			self.synapseErrors = np.zeros((self.nextNeuronCount, self.neuronCount))

class NeuralNetwork:

	def __init__(self):
		self.layers = 0
		self.network = []

	def initNetwork(self, networkStructure):
		self.layers = len(networkStructure)
		# Input Layer
		self.network.append(NeuronLayer(networkStructure[0], networkStructure[1], True))
		# Hidden Layer[s]
		for i in range(self.layers-2):
			self.network.append(NeuronLayer(networkStructure[i+1], networkStructure[i+2]))
		# Output Layer
		self.network.append(NeuronLayer(networkStructure[-1], 0))
			
	def feedForward(self, data):
		self.network[0].neurons = data.reshape(self.network[0].neuronCount, 1)
		for i in range(self.layers-1):
			self.network[i+1].neurons = sigmoid(np.dot(self.network[i].synapses, self.network[i].neurons))

	def backpropagation(self, target, learningRate):
		self.network[-1].neuronErrors = target - self.network[-1].neurons
		for j in reversed(range(self.layers-1)):
				if (j != 0):
					self.network[j].neuronErrors = np.dot(self.network[j].synapses.T, self.network[j+1].neuronErrors)
				self.network[j].synapses += learningRate * np.dot(self.network[j+1].neuronErrors, self.network[j].neurons.T)

	def train(self, trainLabels, trainData, epochs, learningRate):
		for n in range(epochs):
			print("\t-- Epoch %i" % (n+1))
			for label, data in zip(trainLabels, trainData):
				target = self.createTargetMatrix(label)
				self.feedForward(data)
				self.backpropagation(target, learningRate)

	def test(self, labels, test_data):
		right = 0
		total = 0
		for i, (label, data) in enumerate(zip(labels, test_data)):
			self.feedForward(data)
			best_neuron = 0
			best_index = 0
			best_index = np.argmax(self.network[-1].neurons)
			if (label == (best_index+1)):
				right += 1
			total += 1
		return (right/total)


	def createTargetMatrix(self, num):
		arr = np.zeros((self.network[-1].neuronCount, 1))
		arr[num-1] = 1
		return arr

def extractDataAndLabels(fileName):
	fname = open(fileName, "r")
	labels = []
	values = fname.readlines()
	fname.close()
	for i, record in enumerate(values):
		data = record.split(",")
		values[i] = (np.asfarray(data[1:])/255)
		labels.append(int(data[0]))
	return labels, values
		
def main():
	# Number of training sessions
	network = [784, 200, 10]
	epochs = 2
	learningRate = 0.0005
	
	# Create neural network
	print("Creating Network")
	snn = NeuralNetwork()
	snn.initNetwork(network)

	# Open file to loop through
	print("Opening Training Data")
	MNIST_Train_Labels, MNIST_Train_Values = extractDataAndLabels("../datasets/MNIST/mnist_train.csv")
	print("Opening Testing Data")
	MNIST_Test_Labels, MNIST_Test_Values = extractDataAndLabels("../datasets/MNIST/mnist_test.csv")

	# Train
	print("Training:")
	snn.train(MNIST_Train_Labels, MNIST_Train_Values, epochs, learningRate)

	# Test
	print("Testing")
	accuracy = snn.test(MNIST_Test_Labels, MNIST_Test_Values)
	
	# Print Accuracy
	print("Accuracy = %.2f%%" % (accuracy * 100))
	
	
if __name__ == "__main__":
	main()
