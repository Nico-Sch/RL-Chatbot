from Config import *

class UserSimulator:

	def __init__(self):
		self.Reset()

	def Reset(self):		
		self.ResetUnusedSlotValues()
		self.GenerateUserGoal()

	def ResetUnusedSlotValues(self):
		#Used for keeping track of which slots haven't been choosen yet
		self.unusedSlotValues = copy.deepcopy(slotDictionary)
		self.unusedSlotValues.pop('restaurantname')

	def GenerateUserGoal(self):
		self.goal = {'restaurantname':'any', 'city':'any', 'numberofpeople':'any', 'time':'any', 'cuisine':'any', 'pricing':'any'}
		#Adds informs for necessary slots with random values from the slot dictionary or random numeric
		for slot in necessarySlots:
			#With a probability of 80%, the user does not have a restaurant in mind and will make a request to determine one
			if slot == 'restaurantname':
				if random.random() < 0.8:
					#Fill city with high probability
					if random.random() < 0.9:
						self.FillOptionalSlot('city')
					#Fill cuisine with fairly high probability
					if random.random() < 0.8:
						self.FillOptionalSlot('cuisine')
					#Fill pricing with medium probability
					if random.random() < 0.6:
						self.FillOptionalSlot('pricing')
				else:
					chosenValue = random.choice(slotDictionary['restaurantname'])
					self.goal['restaurantname'] = chosenValue
			else:
				chosenValue = random.randint(1,10)
				self.goal[slot] = chosenValue

		self.SelectFirstAction()

	def FillOptionalSlot(self, slot):
		chosenValue = random.choice(slotDictionary[slot])
		self.goal[slot] = chosenValue
		self.unusedSlotValues[slot].remove(chosenValue)

	#Composes the first actions which consists of randomly but reasonably combined inform slots
	def SelectFirstAction(self):
		#Chance of informing several slots at once
		probability = random.random()
		self.firstAction = None

		if self.goal['restaurantname'] == 'any':
			#Inform all optional slots if they are filled
			if probability < 0.3 and self.goal['city'] != 'any' and self.goal['cuisine'] != 'any' and self.goal['pricing'] != 'any':
				self.firstAction = {'intent':'inform', 'informSlots': \
					{'city' : self.goal['city'], 'cuisine' : self.goal['cuisine'], 'pricing' : self.goal['pricing']}}
			else:
				#Inform city and cuisine if they are filled
				if probability <= 0.4 and self.goal['city'] != 'any' and self.goal['cuisine'] != 'any':
					self.firstAction = {'intent':'inform', 'informSlots': \
						{'city' : self.goal['city'], 'cuisine' : self.goal['cuisine']}}
				#Inform city and pricing if they are filled
				elif probability > 0.4 and probability <= 0.7 and self.goal['city'] != 'any' and self.goal['pricing'] != 'any':
					self.firstAction = {'intent':'inform', 'informSlots': \
						{'city' : self.goal['city'], 'pricing' : self.goal['pricing']}}
				#Inform cuisine and pricing if they are filled
				elif probability > 0.7 and self.goal['cuisine'] != 'any' and self.goal['pricing'] != 'any':
					self.firstAction = {'intent':'inform', 'informSlots': \
						{'cuisine' : self.goal['cuisine'], 'pricing' : self.goal['pricing']}}			
		else:
			#Inform all necessary slots with 40% chance
			if probability < 0.4:
				self.firstAction = {'intent':'inform', 'informSlots': \
					{'restaurantname' : self.goal['restaurantname'], 'time' : self.goal['time'], 'numberofpeople' : self.goal['numberofpeople']}}
			else:
				#Inform restaurant and numberofpeople with 30% chance
				if random.random() < 0.5:
					self.firstAction = {'intent':'inform', 'informSlots': \
						{'restaurantname' : self.goal['restaurantname'], 'numberofpeople' : self.goal['numberofpeople']}}
				#Inform restaurant and time with 30% chance
				else:
					self.firstAction = {'intent':'inform', 'informSlots': \
						{'restaurantname' : self.goal['restaurantname'], 'time' : self.goal['time']}}

		#If not multi-inform, choose random single-inform
		if not self.firstAction:
			if self.goal['city'] != 'any' or self.goal['cuisine'] != 'any' or self.goal['pricing'] != 'any':
				while not self.firstAction:
					chosenSlot = random.choice(optionalSlots)
					if self.goal[chosenSlot] != 'any':
						self.firstAction = {'intent':'inform', 'informSlots': {chosenSlot : self.goal[chosenSlot]}}
			else:
				self.firstAction = {'intent':'inform', 'informSlots': {'time' : self.goal['time']}}

	#Returns a user action based on the last agent action
	def GetNextAction(self, turnCount, chosenReservation, agentAction):
		result = None

		#Immedialtely return a fail if the turn limit has been reached
		if turnCount == TURN_LIMIT - 1:
			nextAction = {'intent':'reject', 'informSlots':{}}
			return nextAction, -TURN_LIMIT, FAIL
		else:
			if agentAction:
				#React to agent request by providing the corresponding information as inform
				if agentAction['intent'] == 'request':
					slot = agentAction['requestSlots']
					nextAction = {'intent':'inform', 'informSlots':{slot:self.goal[slot]}}
				#Evaluate the match proposed by the agent
				elif agentAction['intent'] == 'matchFound':
					#If there is no match, change random requestable slot value (city, pricing or cuisine)
					if not agentAction['informSlots']:
						nextAction = self.ChangeOptionalSlotIfNoMatches()
					#If there is a match and it is fulfilling the goal...
					elif agentAction['informSlots'] and self.IsMatchAcceptable(agentAction):
						#...change a random slot and inform it to represent the uncertainty of a real user
						if random.random() < 0.1:
							nextAction = self.ChangeOptionalSlotRandom()
						#...confirm the match
						else:
							nextAction = {'intent':'confirm', 'informSlots':{}}
					#If there is a match and it does not fulfill the goal, reject it
					else:
						nextAction = {'intent':'reject', 'informSlots':{}}
				#Confirm if the agent has concluded the dialogue (meaning this actions has no further influence on the dialogue)
				else:
					nextAction = {'intent':'confirm', 'informSlots':{}}
			#Send first action if there is no previous agent action
			else:
				nextAction = self.firstAction

		#Check if agent has finished the dialogue by making a reservation
		if chosenReservation:
			result = self.DetermineResult(chosenReservation)
		else:
			result = NO_RESULT

		#Give rewards for the agent
		if result == SUCCESS:
			reward = 2 * TURN_LIMIT
			nextAction = {'intent':'confirm', 'informSlots':{}}
		elif result == FAIL:
			reward = -TURN_LIMIT
			nextAction = {'intent':'reject', 'informSlots':{}}
		else:	
			#Give reward of -3 for a normal dialogue turn with no outcome
			reward = -3

		return nextAction, reward, result

	#Changes an optional slot if no database entries match the user goal
	def ChangeOptionalSlotIfNoMatches(self):
		#Reset unused slots if all possible slots have been tried
		if not self.unusedSlotValues['cuisine'] and not self.unusedSlotValues['city'] and not self.unusedSlotValues['pricing']:
			self.ResetUnusedSlotValues()			
		#Choose random slot to change
		slotToReplace = random.choice(optionalSlots)
		#If all slot values have been used of that slot, choose another slot
		while not self.unusedSlotValues[slotToReplace]:
			slotToReplace = random.choice(optionalSlots)
		#Replace slot value with random value
		self.goal[slotToReplace] = random.choice(self.unusedSlotValues[slotToReplace])
		self.unusedSlotValues[slotToReplace].remove(self.goal[slotToReplace])
		#Inform about changed slot
		return {'intent':'inform', 'informSlots':{slotToReplace:self.goal[slotToReplace]}}

	#Checks if a match proposed by the agent fulfills the user goal
	def IsMatchAcceptable(self, agentAction):
		for slot in agentAction['informSlots'].keys():
			if self.goal[slot] == 'any':
				continue
			if agentAction['informSlots'][slot] != self.goal[slot]:
				return False
		return True
		
	#Changes random optional slot
	def ChangeOptionalSlotRandom(self):
		slotToReplace = random.choice(optionalSlots)
		value = random.choice(slotDictionary[slotToReplace])
		#Choose a new random value for the slot until it is different from the old value
		while value == self.goal[slotToReplace]:
			value = random.choice(slotDictionary[slotToReplace])
		self.goal[slotToReplace] = value
		#Reject the match proposed by the agent and inform about the changed slot
		return {'intent':'reject', 'informSlots':{slotToReplace:value}}

	#Compares the user goals with the reservation chosen by the agent
	#If there are differences in the slots, it's a fail
	#If all slots are as requested, it's a success
	def DetermineResult(self, chosenReservation):
		result = SUCCESS
		for slot in self.goal.keys():
			if slot in chosenReservation.keys():
				#If a slot is filled with any in the agent's chosen reservation, it is a fail
				if chosenReservation[slot] == 'any':
					return FAIL
				elif self.goal[slot] == 'any' or self.goal[slot] == chosenReservation[slot]:
					continue
				#If a slot is filled and does not match the user goal, it is a fail
				else:
					return FAIL
			#If a slot is not filled in the agent's chosen reservation, it is a fail
			else:
				return FAIL
		return result