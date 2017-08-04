from experience_replay import *
from emulator import *
import tensorflow as tf
import numpy as np
import time, os
import cv2
import utils
from dqn import *

class atari:
 	def __init__(self, args, sess):
  		print('Initializing..')
  		self.args = args
  		self.sess = sess
  		self.database = experience_replay(self.args)
  		self.engine = emulator(rom_name='breakout.bin', vis=self.args.visualize)
		self.args.num_actions = len(self.engine.legal_actions)
 		# Build net
		self.build_model()

 	def build_model(self):
 	 	print('Building QNet and targetnet..')
 	 	# Create 2 dqn to calculate gradient with fixed parameters
 	 	self.qnet = dqn(self.args, name='qnet')
 	 	self.targetnet = dqn(self.args, name='targetnet')
 	 	self.sess.run(tf.global_variables_initializer())
 	 	# Choose which variable to save and restore, save parts of parameters, not all parameters, pass dictionary
 	 	# Before that, need to initialize
 	 	# Make dictionary
 	 	self.saver_dict = dict()
 	 	for i in self.qnet.tr_vrbs:
 	 	 	if i.name.startswith('qnet/conv1/weight'):
 	 	 	 	self.saver_dict['qw1'] = i
 	 	 	elif i.name.startswith('qnet/conv2/weight'):
 	 	 	 	self.saver_dict['qw2'] = i
 	 	 	elif i.name.startswith('qnet/fc3/weight'):
 	 	 	 	self.saver_dict['qw3'] = i
 	 	 	elif i.name.startswith('qnet/fc4/weight'):
 	 	 	 	self.saver_dict['qw4'] = i
 	 	 	elif i.name.startswith('qnet/conv1/bias'):
 	 	 	 	self.saver_dict['qb1'] = i
 	 	 	elif i.name.startswith('qnet/conv2/bias'):
 	 	 	 	self.saver_dict['qb2'] = i
 	 	 	elif i.name.startswith('qnet/fc3/bias'):
 	 	 	 	self.saver_dict['qb3'] = i
 	 	 	elif i.name.startswith('qnet/fc4/bias'):
 	 	 	 	self.saver_dict['qb4'] = i
 		for i in self.targetnet.tr_vrbs:
			if i.name.startswith('targetnet/conv1/weight'):
 	 	 	 	self.saver_dict['tw1'] = i
 	 	 	elif i.name.startswith('targetnet/conv2/weight'):
 	 	 	 	self.saver_dict['tw2'] = i
 	 	 	elif i.name.startswith('targetnet/fc3/weight'):
 	 	 	 	self.saver_dict['tw3'] = i
 	 	 	elif i.name.startswith('targetnet/fc4/weight'):
 	 	 	 	self.saver_dict['tw4'] = i
 	 	 	elif i.name.startswith('targetnet/conv1/bias'):
 	 	 	 	self.saver_dict['tb1'] = i
 	 	 	elif i.name.startswith('targetnet/conv2/bias'):
 	 	 	 	self.saver_dict['tb2'] = i
 	 	 	elif i.name.startswith('targetnet/fc3/bias'):
 	 	 	 	self.saver_dict['tb3'] = i
 	 	 	elif i.name.startswith('targetnet/fc4/bias'):
 	 	 	 	self.saver_dict['tb4'] = i
 	 	self.saver_dict['step'] = self.qnet.global_step
		for k, v in self.saver_dict.items():
			print(v.op.name)
 	 	#print(self.saver_dict)
 	 	self.saver = tf.train.Saver(self.saver_dict)
 	 	# For copy to targetnet
 	 	# Reorder in alphabetical order, can not sort
 	#	 self.qnet.tr_vrbs.sort()
 	#	 self.targetnet.tr_vrbs.sort()
		''' Copy from target network parameters to online network
			17 parameters, qnet parameter/step/targetnet parameters
		'''
 	 	self.sess.run([self.saver_dict['tw1'].assign(self.saver_dict['qw1']), 
 	 	              self.saver_dict['tw2'].assign(self.saver_dict['qw2']),
 	 	              self.saver_dict['tw3'].assign(self.saver_dict['qw3']),
 	 	              self.saver_dict['tw4'].assign(self.saver_dict['qw4'])])
 	 	self.sess.run([self.saver_dict['tb1'].assign(self.saver_dict['qb1']),
 	 	              self.saver_dict['tb2'].assign(self.saver_dict['qb2']),
 	 	              self.saver_dict['tb3'].assign(self.saver_dict['qb3']),
 	 	              self.saver_dict['tb4'].assign(self.saver_dict['qb4'])])
 	 	print('Copy targetnet from qnet!')

 	 	if self.load():
 	 	 	print('Loaded checkpoint..')
 	 	 	# Get global step
 	 	 	print('Continue from %s steps' % str(self.sess.run(self.qnet.global_step)))
 	 	else:
 	 	 	print('Not load checkpoint')


 	def reset_game(self):
  		print('Reset game at : %s ' % str(self.step))
		# Initialize all thing
  		self.state_proc = np.zeros([84,84,4])
  		self.action = -1
  		self.reward = 0
  		self.terminal = False
  		# [screen_height, screen_width, 3]
  		self.state = self.engine.new_game()
  		# Preprocess by first converting RGB representation to gray-scale and down-sampling it to 110*84
  		# cv2.resize(image, (width, height) => 110 * 84 * 3
  		self.state_resized = cv2.resize(self.state, (84,110))
  		# To gray-scale
  		self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
  		# Reset, no previous state
  		self.state_gray_old = None
  		# state_proc[:,:,:3] will remain as zero
  		self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.args.img_scale

 	def train(self):
  		self.step = 0
  		# Reset game
  		print('Reset before train start')
		self.reset_game()
  		self.reset_statistics()

  		# Increment global step as RMSoptimizer run
		utils.initialize_log()

		# Start time
		self.start_time = time.time()
  		print('Start training')
		# Collecting experience before training, composing batch, just observing
		print('Collecting replay memory for %s steps' % (str(self.args.train_start)))
		self.eps = self.args.initial_eps
 
		# total_step + train start : observing game and store state transitions
		while self.step < self.args.num_iterations:
			# self.state_gray_old is None when reset_game()
			if self.state_gray_old is not None:
				# Collect datas
				self.database.insert(self.state_gray_old[26:110,:], self.action_index, self.reward_scaled, self.terminal)
			# Training
			if self.database.get_size > self.args.train_start:
				if self.database.get_size == self.args.train_start+1:
					print('\tStart Training network')
				print('Current training_step : %d' % self.step)
				self.step += 1
				# Get batches
				batch_state, batch_next_state, batch_actions, batch_terminals, batch_rewards = self.database.get_batches()
				batch_actions = self.get_onehot(batch_actions)
				# Target is reward + maxQ(s',a')
				# Caculate it before running optimizer, before doing iteration so getting fixed parameter effect
				feed = {self.targetnet.x : batch_next_state}
				# next_q : batch_size, number of actions
				next_q = self.sess.run(self.targetnet.y, feed_dict = feed)
				next_q_max = np.max(next_q, axis=1)
				# Feed next_q_max to qnet.q to do gradient descent only for q.q_pred 
				feed = {self.qnet.x : batch_state, self.qnet.actions : batch_actions, self.qnet.rewards : batch_rewards, self.qnet.terminals: batch_terminals, self.qnet.q : next_q_max}
				loss_, _ = self.sess.run([self.qnet.loss, self.qnet.train_op], feed_dict = feed)
				self.total_loss += loss_

				# Decaying epsilon
				self.eps = max(self.args.eps_min, self.args.initial_eps - float(self.step)/float(self.args.eps_step))

				# Copy qnet to target net
				if self.step % self.args.copy_interval == 0:
					self.copy_network()				
 				# Save
				if self.step % self.args.save_interval == 0:
					self.save(self.step)
				# Log
				if self.step % self.args.log_interval == 0:
					utils.write_log(self.step, self.total_reward, self.total_Q, self.eps, mode='train')

			# When game is finished(One episode is over)
			if self.terminal:
				print('Reset for episode ends')
				self.reset_game()
				self.num_epi += 1
				self.total_reward += self.epi_reward
				self.epi_reward = 0

			# Get epsilon greedy action from state
			self.action_index, self.action, self.maxQ = self.select_action(self.state_proc)
			# Get reward and next state from environment
			self.state, self.reward, self.terminal = self.engine.next(self.action)
			# Scale rewards, all positive rewards to be 1, all negative rewards to be -1
			self.reward_scaled = self.reward // max(1,abs(self.reward))
			self.epi_reward += self.reward_scaled
			self.total_Q += self.maxQ

			# Change to next 4 consecutive images
			'''
			x = np.array([1,2,3]) ; y = x; z = np.copy(x)
			x[0] = 10 => y[0] == x[0] but z[0] != x[0]
			'''
			self.state_gray_old = np.copy(self.state_gray)
			self.state_proc[:,:,0:3] = self.state_proc[:,:,1:4]
			# Preprocess
			self.state_resized = cv2.resize(self.state, (84,110))
			self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
			self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.args.img_scale

	def evaluation(self):
		self.eval_step = 0
		if self.load():
			print('Loaded checkpoint')
		else:
			raise Exception('No checkpoint')

		self.reset_game()
		self.reset_statistics()
		utils.initialize_log()

		while self.eval_step < self.args.num_iterations:
			self.eval_step += 1
			if self.eval_step % self.log_interval == 0:
				utils.write_log(self.eval_step, self.total_reward, self.total_Q, self.args.eps_min, mode='eval')

			# When game is finished(One episode is over)
			if self.terminal:
				print('Reset since episode ends')
				self.reset_game()
				self.num_epi += 1
				self.total_reward += self.epi_reward
				self.epi_reward = 0

			# Get epsilon greedy action from state
			self.action_index, self.action, self.maxQ = self.select_action(self.state_proc)
			# Get reward and next state from environment
			self.state, self.reward, self.terminal = self.engine.next(self.action)
			# Scale rewards, all positive rewards to be 1, all negative rewards to be -1
			self.reward_scaled = self.reward // max(1,abs(self.reward))
			self.epi_reward += self.reward_scaled
			self.total_Q += self.maxQ
	
			# Change to next 4 consecutive images
			'''
			x = np.array([1,2,3]) ; y = x; z = np.copy(x)
			x[0] = 10 => y[0] == x[0] but z[0] != x[0]
			'''
			self.state_gray_old = np.copy(self.state_gray)
			self.state_proc[:,:,0:3] = self.state_proc[:,:,1:4]
			# Preprocess
			self.state_resized = cv2.resize(self.state, (84,110))
			self.state_gray = cv2.cvtColor(self.state_resized, cv2.COLOR_BGR2GRAY)
			self.state_proc[:,:,3] = self.state_gray[26:110,:]/self.args.img_scale


	def copy_network(self):
		print('Copying qnet to targetnet')
		self.sess.run([self.saver_dict['tw1'].assign(self.saver_dict['qw1']), 
	                  self.saver_dict['tw2'].assign(self.saver_dict['qw2']),
	                  self.saver_dict['tw3'].assign(self.saver_dict['qw3']),
	                  self.saver_dict['tw4'].assign(self.saver_dict['qw4'])])
		self.sess.run([self.saver_dict['tb1'].assign(self.saver_dict['qb1']),
	                  self.saver_dict['tb2'].assign(self.saver_dict['qb2']),
	                  self.saver_dict['tb3'].assign(self.saver_dict['qb3']),
	                  self.saver_dict['tb4'].assign(self.saver_dict['qb4'])])
		print('Copy targetnet from qnet!')
 
	def reset_statistics(self):
   		self.epi_reward = 0
		self.num_epi = 0
		self.total_reward = 0
		self.total_Q = 0
		self.total_loss = 0


	def select_action(self, state):
		# Greedy action
		if np.random.rand() > self.eps:
			print('Greedy action')
			# batch size for 'x' is 1 since we choose action for specific state
			q_prediction = self.sess.run(self.qnet.y, feed_dict={self.qnet.x : np.reshape(state, [1,84,84,4])})[0]
   			# Consider case when there are several same q max value
			# argwhere(if statement), return 2 dim array
			max_action_indices = np.argwhere(q_prediction == np.max(q_prediction))
			# If max_action_indices has more than 1 element, Choose 1 of them
			if len(max_action_indices) > 1:
				action_idx = max_action_indices[np.random.randint(0, len(max_action_indexs))][0]
				return action_idx, self.engine.legal_actions[action_idx], np.max(q_prediction)
			else:
				action_idx = max_action_indices[0][0]
				return action_idx, self.engine.legal_actions[action_idx], np.max(q_prediction)
		# episilon greedy action
		else:
			action_idx = np.random.randint(0,len(self.engine.legal_actions))
			print('Episilon greedy action : %d ' %self.engine.legal_actions[action_idx])
			q_prediction = self.sess.run(self.qnet.y, feed_dict={self.qnet.x : np.reshape(state, [1,84,84,4])})[0]
			return action_idx, self.engine.legal_actions[action_idx], q_prediction[action_idx]


	# action : [batch_size,] and element is integer, environment gives it as an integer
	def get_onehot(self, action):
		one_hot = np.zeros([self.args.batch_size, self.args.num_actions])
		for i in xrange(self.args.batch_size):
			one_hot[i, int(action[i])] = 1
		return one_hot

	@property
	def model_dir(self):
		return '{}_batch'.format(self.args.batch_size)

	def save(self, total_step):
		model_name = 'DQN'
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		if not os.path.exists(chekcpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(chekcpoint_dir, model_name, global_step=total_step))
		print('Model saved at %s in %d steps' % (checkpoint_dir, total_step))

 	def load(self):
  		print('Loading checkpoint..')
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
  		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			print(ckpt_name)
			self.saver.restore(self.sess, ckpt.model_checkpoint_path)
			print('Success to load %s' % (ckpt_name))
			return True
		else:
			print('Failed to find a checkpoint')
			return False   
