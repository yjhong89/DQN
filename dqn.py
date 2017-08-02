import tensorflow as tf
import numpy as np
import time
import utils

'''
Define Q networks
'''


class dqn:
 	def __init__(self, args, name):
		self.args = args
		self.network_name = name
		with tf.variable_scope(self.network_name):
			# Input : batch, 84,84,4(grayscale)
			self.x = tf.placeholder(tf.float32, [None, 84, 84, 4], name='x')
			# Q value : batch(max value among actions)
			self.q = tf.placeholder(tf.float32, [None], name='q')
			# Action : batch, number of action
			self.actions = tf.placeholder(tf.float32, [None, self.args.num_actions], name='actions')
			# Reward : batch, 
			self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
			# Terminal : batch, 
			self.terminals = tf.placeholder(tf.float32, [None], name='terminals')
 
			# Conv1 layer
			# Use 8*8 16 filters, out : batch, 20,20,16
			self.convolution1, conv1_shape = utils.conv2d(self.x, 16, 8, 8, 4, name='conv1')
			print(conv1_shape)
			self.out1 = tf.nn.relu(self.convolution1, name='activation1')
  
			# Conv2 layer
			# Use 4*4 32 filters, out: batch,9,9,32
			self.convolution2, conv2_shape = utils.conv2d(self.out1, 32, 4, 4, 2, name='conv2')
			print(conv2_shape)
			self.out2 = tf.nn.relu(self.convolution2, name='activation2')
   
			out2_shape = self.out2.get_shape().as_list()
 
			# Fully Connected layer
			# 256 hidden units
			self.out2_flat = tf.reshape(self.out2, [self.args.batch_size, -1], name='input_flat')
			self.out3 = utils.linear(self.out2_flat, 256, name='fc3')
			self.activation3 = tf.nn.relu(self.out3, name='activation3')
  
			# Last fully connected layer
			# output dimension is number of actions : [batch , number of actions]
			self.y = utils.linear(self.activation3, self.args.num_actions, name='fc4')

			# Define q value and loss function
			# Also consider terminal state(no next state, so no next q value)
			self.q_target = self.rewards + tf.mul(1-self.terminals, tf.mul(self.args.discount_factor, self.q))
			# [batch,]
			self.q_pred= tf.reduce_sum(tf.mul(self.y, self.actions), reduction_indices=1)

			self.diff_square = tf.mul(tf.constant(0.5), tf.pow(tf.sub(self.q_target - self.q_pred), 2))

			self.loss = tf.reduce_mean(self.diff_square)

			# Check training step
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			# Use RMSpropoptimizer
			self.train_op = tf.train.RMSPropOptimizer(self.args.learning_rate, 0.99, 0, 1e-6).minimize(self.loss, global_step=self.global_step)
			self.tr_vrbs = tf.trainable_variables()
			for i in xrange(len(self.tr_vrbs)):
				print(self.tr_vrbs[i].name)

