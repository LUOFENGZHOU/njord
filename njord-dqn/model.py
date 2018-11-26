#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 1.0E-3


class Model(nn.Module):

	def __init__(self, state_dim, action_dim, usecuda=False):
		"""Special method for class initialisation.

		:param state_dim: Dimension of input state.
		:type state_dim: int.
		:param action_dim: Dimension of output action.
		:type action_dim: int.
		"""
		super(Model, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden = 64
		self.usecuda = usecuda

		self.rnn1 = nn.GRUCell(self.state_dim, self.hidden, bias=True)
		self.rnn2 = nn.GRUCell(self.hidden, self.hidden, bias=True)

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
		if self.usecuda and torch.cuda.is_available():
			h = h.cuda()
		return h

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
		s1 = self.zero_state(state.size(0))
		s2 = self.zero_state(state.size(0))

		# Loop on the temporal dimension of the batch.
		for t in range(x.size(2)):
			s1 = self.rnn1(x[:,:,t], s1)
			s2 = self.rnn2(s1, s2)

		# Forward pass to get the action.
		action = self.fc1(s2)

		return action



