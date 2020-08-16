import random, copy

###Constants

#Defines whether the program is in training
IN_TRAINING = False
#Defines whether the program is used by a real user or the user sim
REAL_USER = True
#For printing the dialogues in training
PRINTING = False
#For printing the success rate
PRINT_PROGRESS_INTERVAL = 1000
#Minimum epsilon value for epsilon-decreasing exploration
EPSILON_MIN = 0.01
#Value which epsilon is multiplied by to decrease its impact over time
EPSILON_DECREASE = 0.9999
#Learning rate alpha
ALPHA = 0.0002
#Discount factor gamma for value updates
GAMMA = 0.99
#Capacity of the replay buffer
MEMORY_CAPACITY = 100000
#Batch size of sampled tuples from the replay buffer for training
BATCH_SIZE = 16
#File name of the the dqn model saved at the end of training
FILE_NAME = 'dqn_model.h5'
#How many rounds can occur in a conversation at most
TURN_LIMIT = 20
#How many episodes/dialogues to train in total
TRAIN_AMOUNT = 100000
#Number of neurons in the deep q networks hidden layer
HIDDEN_SIZE = 80
#Number of turns between target network updates based on the online network
TARGET_UPDATE_INTERVAL = 100
#Used to check for result of dialogue in user simulator
FAIL = -1
NO_RESULT = 0
SUCCESS = 1

#Slots
allSlots = ['restaurantname', 'numberofpeople', 'city', 'time', 'cuisine', 'pricing']
fillableSlots = ['match', 'restaurantname', 'numberofpeople', 'city', 'time', 'cuisine', 'pricing']
necessarySlots = ['restaurantname', 'numberofpeople', 'time']
optionalSlots = ['city', 'cuisine', 'pricing']
requestableSlots = ['city', 'cuisine', 'pricing']

slotDictionary = { 'restaurantname' : [ 'Pizzeria Napoli', 'Roshis Ramen', 'Bingo Burgers', 'Sallys Seafood',
									    'Pasta House', 'Il Cavallino', 'Leonidas', 'The Mussel Shed', 'Sushi Roma',
									    'Da Gonzo', 'The Wasabi House', 'Buffalo Boys', 'Steaks and Fries', 
										'Don Camillo', 'Happy Sushi', 'Burger Factory', 'Texas Steak House',
									    'El Pescador', 'Akropolis Grill', 'Pizza Place', 'Ichiban', 'Shinjuku Sushi',
									    'La Cantina', 'Bobs Diner', 'Tapas Bar', 'The Jasmin Dragon', 'Hasia Ung',
									    'Wang Jo', 'Soul Food', 'The Golden Dragon', 'The Talking Duck', 'Everything Fried'],
				   'city' : [ 'Rome', 'Berlin', 'London', 'Tokyo' ],
				   'cuisine' : ['American', 'Japanese', 'Italian', 'Mediterranean', 'Chinese' ],
				   'pricing' : [ 'Expensive', 'Cheap', 'Average' ] }

#For one-hot encoding in state representation
userIntents = ['inform', 'reject', 'confirm']
agentIntents = ['done', 'matchFound', 'request']

#Set all possible actions of the agent
agentActions = []
agentActions.append({'intent':'done', 'requestSlots': None})
agentActions.append({'intent':'matchFound', 'requestSlots': None, 'informSlots':{}})

for slot in allSlots:
	agentActions.append({'intent':'request', 'requestSlots': slot})