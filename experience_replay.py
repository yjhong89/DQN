import tensorflow as tf
import numpy as np
import time


'''
Database for minibatch
'''

class experience_replay:
 	def __init__(self, args):
		self.args = args
		# Memory size to store (s,a,r,s')
		# Max experience for experience replay
		# Do not need next state, we use terminals
		# Actions will be one-hot encoded later
		self.size = self.args.db_size
		self.img_scale = self.args.img_scale
		self.states = np.empty([self.size, 84, 84])
		self.actions = np.empty([self.size])
		self.terminals = np.empty([self.size])
		self.rewards = np.empty([self.size])
		# For minibatch from memory
		self.batch_size = self.args.batch_size
		self.batch_states = np.empty([self.batch_size, 84, 84, 4])
		self.batch_actions = np.empty([self.batch_size])
		self.batch_terminals = np.empty([self.batch_size])
		self.batch_rewards = np.empty([self.batch_size])
		self.batch_next_states = np.empty([self.batch_size, 84, 84, 4])

		# Keep track of memory until full 
		self.pointer = 0
		# Flag indicating full or not
		self.flag = False

 	# Get minibatch as batch_size
 	def get_batches(self):
		for i in xrange(self.batch_size):
		# We need to get 4 consecutive 84*84 images
			idx = 0
			while idx < 3 or (idx > self.pointer-2 and idx < self.pointer+3):
				# Getting 'index' until conditions are satisfied
				# if idx < self.pointer + 3, 4 consecutive images are not allowed
				# if idx > self.pointer -2, 4 consecutive images for next states are not allowed
				idx = np.random.randint(3, self.get_size-1)
			# For i th batch
			# 4,84,84 -> 84, 84, 4, and make it grayscale to represent 0~255, (idx-3, idx-2, idx-1, idx)
			self.batch_states[i] = np.transpose(self.states[idx-3:idx+1,:,:], (1,2,0))/self.img_scale
			self.batch_next_states[i] = np.transpose(self.states[idx-2:idx+2,:,:], (1,2,0))/self.img_scale
			self.batch_actions[i] = self.actions[idx]
			self.batch_terminals[i] = self.terminals[idx]
			self.batch_rewards[i] = self.rewards[idx]
		return self.batch_states, self.batch_next_states, self.batch_actions, self.batch_terminals, self.batch_rewards


 	# Insert experience
 	def insert(self, cur_state, action, reward, terminal):
  		self.states[self.pointer] = cur_state
  		self.actions[self.pointer] = action
  		self.rewards[self.pointer] = reward
  		self.terminals[self.pointer] = terminal
  		# Update pointer
  		self.pointer += 1
  		if self.pointer >= self.size:
			print('Buffer is full')
   			self.flag = True
   			self.pointer = 0

 	@property
 	def get_size(self):
  		if self.flag == False:
   			return self.pointer
  		else:
   			return self.size
