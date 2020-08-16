from DeepQNetwork import *
from Database import *
import numpy as np

class Agent:

	def __init__(self, stateSize):
		#Replay buffer for storing and sampling experience for learning
		self.memory = ReplayBuffer()
		#Online network for choosing actions
		self.onlineNetwork = InitializeDqn(stateSize)
		#Target network for computing target values in learning
		self.targetNetwork = InitializeDqn(stateSize)
		#The final reservation made to end the dialogue
		self.chosenReservation = None
		#Initialize epsilon with 1 for epsilon-decreasing strategy
		self.epsilon = 1

	#Takes the current state and its size to choose an action
	def PredictNextAction(self, state, stateSize):
		rand = random.random()
		#Determine if random or policy-based action
		if IN_TRAINING and rand < self.epsilon:
			nextAction = random.choice(agentActions)
		else:
			#Network outputs value for each agent action based on the state
			actionValues = self.onlineNetwork.predict(state.reshape(1, stateSize)).flatten()
			#Gets index of action with highest Q-value
			nextActionIndex = np.argmax(actionValues)
			#Returns corresponding action based on the chosen index
			nextAction = self.IndexToAction(nextActionIndex)

		#Decrease the epsilon
		if self.epsilon > EPSILON_MIN:
			self.epsilon *= EPSILON_DECREASE

		return nextAction

	#Takes an index and returns the corresponding agent action
	def IndexToAction(self, index):
		for (i, action) in enumerate(agentActions):
			if index == i:
				return copy.deepcopy(action)

	#Takes an agent action and returns the corresponding index
	def ActionToIndex(self, action):
		for (i, a) in enumerate(agentActions):
				if action == a:
					return i

	#Reset the agent for usage in new dialogue
	def Reset(self):
		self.chosenReservation = None

	#Adapts the network weights by learning from memory
	def Learn(self, stateSize):
		#Only start learning if a complete batch can be sampled from the buffer
		if self.memory.indexCounter < BATCH_SIZE:
			return

		batch = self.memory.SampleBatchFromBuffer()

		#For each tuple in the batch
		for state, action, reward, nextState in batch:
			#Compute Q-values of online network
			qNow = self.onlineNetwork.predict(state.reshape(1, stateSize)).flatten()
			#Initialize target Q-values with online Q-values
			qTarget = qNow.copy()

			#If the tuple has no next state, the Q-value is the reward
			if isinstance(nextState, list):
				qTarget[self.ActionToIndex(action)] = reward
			else:
				#Compute Q-values of the next state usign the target network
				qNext = self.targetNetwork.predict(nextState.reshape(1, stateSize)).flatten()
				#Set target Q-value of the tuple action to the reward plus the discounted maximal Q-value of the next state
				qTarget[self.ActionToIndex(action)] =  reward + GAMMA * max(qNext)
			
			#Adjust the weights according to the difference of online and target values
			self.onlineNetwork.fit(state.reshape(1, stateSize), qTarget.reshape(1, len(agentActions)), epochs=1, verbose=0)

	#Copies weights of the online network to the target network
	def CopyToTargetNetwork(self):
		self.targetNetwork.set_weights(self.onlineNetwork.get_weights())

	#Saves online network
	def SaveModel(self):
		self.onlineNetwork.save(FILE_NAME)

	#Loads online network
	def LoadModel(self):
		self.onlineNetwork = load_model(FILE_NAME) 

	#Choose request utterance based on the slot
	def GenerateRequestResponse(self, nextAction):
		slot = nextAction['requestSlots']

		if slot == 'restaurantname':
			return 'Do you have a specific restaurant in mind?'
		elif slot == 'numberofpeople':
			return 'How many persons are coming?'
		elif slot == 'city':
			return 'In which city do you want to eat?'
		elif slot == 'time':
			return 'At what time do you want to book?'
		elif slot == 'cuisine':
			return 'Which cuisine are you looking for?'
		elif slot == 'pricing':
			return 'How high shall the pricing be?'

	#Compose response to propose a matching restaurant
	def GenerateMatchFoundResponse(self, nextAction):
		responseString = []
		#Missing inform slots indicate that there is no matching database entry
		if nextAction['informSlots']:
			match = nextAction['informSlots']
			responseString.append(f"How about \"{match['restaurantname']}\"? ")
			responseString.append(f"It is located in {match['city']} and has ")
			responseString.append(f"{match['cuisine']} cuisine with {match['pricing'].lower()} pricing.")
		else:
			responseString.append('No restaurant matches the current information.')
		return ''.join(responseString)

	#Compose response for finishing the dialogue and informing about the made reservation
	def GenerateDoneResponse(self, filledSlots):
		if 'match' in filledSlots.keys():
			self.chosenReservation = self.GetEntryFromDb(filledSlots['match'])
			for slot, value in filledSlots.items():
				if slot not in self.chosenReservation.keys():
					self.chosenReservation[slot] = value
		else:
			self.chosenReservation = filledSlots

		if 'restaurantname' in self.chosenReservation.keys() and not self.chosenReservation['restaurantname'] == 'any':
			name = self.chosenReservation['restaurantname']
		else:
			name = 'Unknown'

		if 'city' in self.chosenReservation.keys() and not self.chosenReservation['city'] == 'any':
			city = self.chosenReservation['city']
		else:
			city = 'Unknown'

		if 'numberofpeople' in self.chosenReservation.keys():
			people = self.chosenReservation['numberofpeople']
		else:
			people = 'Unknown'

		if 'time' in self.chosenReservation.keys():
			time = self.chosenReservation['time']
		else:
			time = 'Unknown'

		return f"A reservation has been made at \"{name}\" in {city} for {people} people at {time}."

	#Returns a database entry given a restaurant name
	def GetEntryFromDb(self, restaurantName):
		for entry in copy.deepcopy(database):
			if entry['restaurantname'] == restaurantName:
				return entry