
# Homecoming (eYRC-2018): Task 1B
# Fruit Classification with a CNN

from model import FNet
# import required modules
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

def train_model(dataset_path, debug=False, destination_path='', save=False):
	"""Trains model with set hyper-parameters and provide an option to save the model.

	This function should contain necessary logic to load fruits dataset and train a CNN model on it. It should
	accept dataset_path which will be path to the dataset directory. You should also specify an option to save the 
	trained model with all parameters. If debug option is specified, it'll print loss and accuracy for all iterations. Returns loss and accuracy for both train and validation sets.

	Args:
		dataset_path (str): Path to the dataset folder. For example, '../Data/fruits/'.
		debug (bool, optional): Prints train, validation loss and accuracy for every iteration. Defaults to False.
		destination_path (str, optional): Destination to save the model file. Defaults to ''.
		save (bool, optional): Saves model if True. Defaults to False.

	Returns:
		loss (torch.tensor): Train loss and validation loss.
		accuracy (torch.tensor): Train accuracy and validation accuracy.
	"""
	# Write your code here
	# The code must follow a similar structure
	# NOTE: Make sure you use torch.device() to use GPU if available

	#CHECH IF GPU IS AVAILABLE
	#gpu_avail=torch.cuda.is_available()
	#if gpu_avail:
    	#net.cuda()

	trainbatch_size=8
	testbatch_size=1
	n_iterations = 3
	
	find_loss = nn.CrossEntropyLoss()
	learning_rate = 0.001

	#ADJUSTING LEARNING RATE
	if n_iterations > 120:
    	learning_rate = learning_rate/10000
	elif n_iterations > 90 and n_iterations < 120:
    	learning_rate = learning_rate/1000
	elif n_iterations > 60 and n_iterations < 90:
    	learning_rate = learning_rate/100
	elif n_iterations > 30 and n_iterations < 60:
    	learning_rate = learning_rate/10
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
	#optimizer=optim.Adam(net.parameters(),lr=0.001)

	average_loss = 0.0	
	print("TRAINING OF IMAGES STARTED")
	losses = []
	for epoch in range(n_iterations):
    	for i, data in enumerate(trainloader, 0):
        	inputs, labels = data
        	#if gpu_avail:
        		#inputs = Variable(inputs.cuda())
        		#labels = Variable(labels.cuda())
        	optimizer.zero_grad()
        	outputs = net(inputs)
        	loss = find_loss(outputs, labels)
        	loss.backward()
        	optimizer.step()
        	average_loss += loss.item()
    	losses.append(average_loss*trainbatch_size/trainset_length)
    	print("EPOCH= ", epoch+1, "    AVERAGE LOSS DURING TRAINING = ",average_loss*trainbatch_size/(trainset_length))
    	average_loss = 0.0
	print("TRAINING FINISHED")
	# plt.plot(losses)
	# plt.savefig("lossplot.png", transparent=True)	


	#TESTING TESTSET THROUGH TRAINED CNN
	correct = 0
	total = 0
	average_loss = 0.0
	losses=[]
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			#if gpu_avail:
				#images = Variable(images.cuda())
				#labels = Variable(labels.cuda())
			outputs = net(images)
			loss = find_loss(outputs, labels)
        	loss.backward()
			average_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			if predicted.item() == labels.item():
				correct += 1
		losses.append(average_loss*trainbatch_size/trainset_length)
	print("TOTAL NUMBER OF IMAGES IN TEST SET=", total)
	print("TOTAL NUMBER OF IMAGES PREDICTED CORRECTLY=", correct)
	print("AVERAGE LOSS DURING TESTING = ",average_loss*trainbatch_size/(trainset_length))
	pass

if __name__ == "__main__":
	train_model('./Data/fruits/', save=True, destination_path='./')
