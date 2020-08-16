import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from Config import *
from numpy import random

#Returns a neural network with an input with units equal to stateSize,
#a hidden layer and an output layer with units equal to the number of possible agent actions
def InitializeDqn(stateSize):
	model = Sequential()

	#Defines input dimension as well as the first hidden layer
	model.add(Dense(units=HIDDEN_SIZE, input_dim=stateSize, activation='relu'))
	#Defines output layer
	model.add(Dense(units=len(agentActions), activation='linear'))
	#Creates model with optimizer and loss function
	model.compile(optimizer=Adam(lr=ALPHA), loss='mse')

	return model

class ReplayBuffer:

	def __init__(self):
		self.memory = []
		self.indexCounter = 0

	#Stores (s,a,r,s') tuples in replay buffer memory
	def StoreTransition(self, state, action, reward, nextState):
		index = self.indexCounter
		if len(self.memory) < MEMORY_CAPACITY:
			self.memory.append((state, action, reward, nextState))
		else:
			self.memory[index] = (state, action, reward, nextState)

		if self.indexCounter == MEMORY_CAPACITY - 1:
			self.indexCounter = 0

		self.indexCounter += 1

	#Gets a number of random samples from the replay buffer memory
	def SampleBatchFromBuffer(self):
		batch = random.choice(self.indexCounter, BATCH_SIZE)

		batchList = []
		for index in batch:
			batchList.append(self.memory[index])

		return batchList
