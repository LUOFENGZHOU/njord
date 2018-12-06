#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
	"""Critic network
	"""

	def __init__(self, state_dim, action_dim, usecuda=False):
		"""Special method for class initialisation.

		:param state_dim: Dimension of input state.
		:type state_dim: int.
		:param action_dim: Dimension of input action.
		:type action_dim: int.
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden = 128
		self.usecuda = usecuda		

		self.rnn = nn.LSTMCell(self.state_dim, self.hidden, bias=True)

		self.fcs1 = nn.Linear(self.hidden,1)
		self.fcs1.weight.data.uniform_(-EPS,EPS)

		self.fca1 = nn.Linear(self.action_dim,1)
		self.fca1.weight.data.uniform_(-EPS,EPS)
		
		return

	def zero_state(self, batch_size):
		"""Returns the initial network state.

		:param batch_size: the size of the current batch.
		:type batch_size: int.

		:return: the initial network state.
		:rtype: list of torch tensors.
		"""
		h = torch.zeros(batch_size, self.hidden)
		c = torch.zeros(batch_size, self.hidden)
		if self.usecuda and torch.cuda.is_available():
			h = h.cuda()
			c = c.cuda()
		return h, c

	def forward(self, state, action):
		"""Returns Value function Q(s,a) obtained from critic network.

		:param state: Input state.
		:type state: torch tensor.
		:param action: Input Action.
		:type action: Torch tensor. 
		
		:return: Value function Q(S,a) [m, 1]
		:rtype: Torch Variable.
		"""
		action = action.view(-1, 1)
	
		x = state.permute(0,2,1)
		s = self.zero_state(state.size(0))

		for t in range(x.size(2)):
			s = self.rnn(x[:,:,t], s)

		return self.fcs1(s[0]) + self.fca1(action)


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim, usecuda=False):
		"""Special method for class initialisation.

		:param state_dim: Dimension of input state.
		:type state_dim: int.
		:param action_dim: Dimension of output action.
		:type action_dim: int.
		:param action_lim: Used to limit action.
		:type action_lim: float.
		
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.hidden = 128
		self.usecuda = usecuda

		self.rnn = nn.LSTMCell(self.state_dim, self.hidden, bias=True)

		self.fc1 = nn.Linear(self.hidden, action_dim)
		self.fc1.weight.data.uniform_(-EPS,EPS)

		return

	def zero_state(self, batch_size):
		"""Returns the initial network state.

		:param batch_size: the size of the current batch.
		:type batch_size: int.

		:return: the initial network state.
		:rtype: list of torch tensors.
		"""
		h = torch.zeros(batch_size, self.hidden)
		c = torch.zeros(batch_size, self.hidden)
		if self.usecuda and torch.cuda.is_available():
			h = h.cuda()
			c = c.cuda()
		return h, c

	def forward(self, state):
		"""returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""

		# Reshape the batch.
		x = state.permute(0,2,1)

		# Get the initial state.
		s = self.zero_state(state.size(0))

		# Loop on the temporal dimension of the batch.
		for t in range(x.size(2)):
			s = self.rnn(x[:,:,t], s)

		# Forward pass to get the action.
		action = torch.tanh(self.fc1(s[0]))

		return action



