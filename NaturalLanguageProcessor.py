from Config import *
from gensim import models
from gensim import utils
import gensim.downloader as api

class NaturalLanguageProcessor:

	def __init__(self):
		#Load the gensim model
		self.model = api.load('glove-wiki-gigaword-50')

	#Transform user utterance into semantic frame
	def GetSemanticFrame(self, userInput, lastAgenAction):
		#Get inform slots
		informSlots = self.GetSlots(userInput.lower(), lastAgenAction)

		#Intent if inform if there are inform slots
		if informSlots:
			intent = 'inform'
		else:
			#Intent has to be determined (confirm or deny)
			intent = self.GetIntent(userInput.lower())

		#Return user action
		return {'intent':intent, 'requestSlots':{}, 'informSlots':informSlots}

	#Detects intent using the gensim model
	def GetIntent(self, userInput):
		confirmProbability = 0.0
		rejectProbability = 0.0

		for word in userInput.split():
			if word in self.model.wv.vocab:
				#Averaged similarity to possible confirm words
				confirmProbability += (self.model.wv.similarity(word, 'confirm') \
									 + self.model.wv.similarity(word, 'accept') \
									 + self.model.wv.similarity(word, 'ok') \
									 + self.model.wv.similarity(word, 'fine') \
									 + self.model.wv.similarity(word, 'good') \
									 + self.model.wv.similarity(word, 'yes')) / 6.0
				#Averaged similarity to possible reject words
				rejectProbability += (self.model.wv.similarity(word, 'reject') \
									 + self.model.wv.similarity(word, 'deny') \
									 + self.model.wv.similarity(word, 'not') \
									 + self.model.wv.similarity(word, 'wrong') \
									 + self.model.wv.similarity(word, 'no')) / 5.0

		if confirmProbability > rejectProbability:
			return 'confirm'
		else:
			return 'reject'

	#Returns dictionary of detected inform slots
	def GetSlots(self, userInput, lastAgenAction):
		slots = {}

		#Detect restaurant name
		for restaurant in slotDictionary['restaurantname']:
			matchesRestaurantName = True
			for namePart in restaurant.lower().split():
				if namePart not in userInput:
					matchesRestaurantName = False
					break
			if matchesRestaurantName:
				slots['restaurantname'] = restaurant
				break

		#Detect city and cuisine
		for word in userInput.split():
			if word.capitalize() in slotDictionary['city']:
				slots['city'] = word.capitalize()
			elif word.capitalize() in slotDictionary['cuisine']:
				slots['cuisine'] = word.capitalize()

		if 'cuisine' not in slots.keys():
			for word in userInput.split():
				#Choose randomly Japanese or Chinese if user wants Asian
				if word == 'asian':
					slots['cuisine'] = random.choice(['Japanese', 'Chinese'])
				#Mediterranean cuisine for Spanish, Greek and Southern
				elif word in ['spanish', 'greek', 'southern']:		
					slots['cuisine'] = 'Mediterranean'

		#Detect pricing
		for word in userInput.split():
			if word in ['cheap', 'low', 'inexpensive', 'low-cost', 'low-priced', 'low-budget']:
				slots['pricing'] = 'Cheap'
			elif word in ['average', 'normal', 'standard', 'moderate', 'affordable', 'medium']:
				slots['pricing'] = 'Average'
			elif word in ['expensive', 'costly', 'high', 'high-cost', 'high-priced', 'high-budget']:
				slots['pricing'] = 'Expensive'

		#Detect number of people if it is followed by the words people, persons or guests
		previousWord = ''
		for word in userInput.split():
			if word in ['people', 'persons', 'guests']:
				if self.IsNumber(previousWord):
					slots['numberofpeople'] = previousWord
					break
			previousWord = word

		#Detect time if it contains : or if its previous word is at
		previousWord = ''
		for word in userInput.split():
			if word[0].isnumeric() and ':' in word:
				slots['time'] = word
				break
			elif previousWord == 'at' and self.IsNumber(word):
				slots['time'] = word
				break
			previousWord = word

		#If no slots have been detected yet but the agent's intent was request
		if lastAgenAction and not slots and lastAgenAction['intent'] == 'request':
			#Choose any as value for city, cuisine or pricing if they have been requested
			if lastAgenAction['requestSlots'] not in ['time', 'numberofpeople']:
				slots[lastAgenAction['requestSlots']] = 'any'
			#If only one word which is a number: Fill time or number of people with it
			elif len(userInput.split()) == 1 and self.IsNumber(userInput):
				if lastAgenAction['requestSlots'] == 'time':
					slots['time'] = userInput
				elif lastAgenAction['requestSlots'] == 'numberofpeople':
					slots['numberofpeople'] = userInput

		return slots

	#Check if word is number by numerical value or high similarity to the word number
	def IsNumber(self, word):
		if word.isnumeric() or (word in self.model.wv.vocab and self.model.wv.similarity(word, 'number') > 0.7):
			return True
		else:
			return False

